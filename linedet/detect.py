"""Find text lines on a page with LineNet and put them in Sindhi reading order.

    det = LineDetector()                 # loads model/weights/linedet.pt on the GPU
    lines = det.detect(gray)             # [{"box": (x0,y0,x1,y1), "quad": 4x2, "type": "body", "height": h}]
    lines = reading_order(lines)         # headers, then columns right to left, footnotes, footers
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from linedet.model import N_TYPES, LineNet, page_input

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "model" / "weights" / "linedet.pt"
TYPE_NAMES = ["body", "heading", "header", "footer", "footnote"]
DET_SIDE = 1600  # long side the detector reads the page at
MIN_LINE_H = 13.0  # lines thinner than this at DET_SIDE: read the page again larger
CORE_THRESHOLD = 0.5
# pieces of one row are joined into one line: boxes that overlap side by side (a line found twice,
# or its first/last letters found as a separate sliver), and short pieces of widely spaced poetry
ROW_OVERLAP = 0.5  # share of the shorter box's height the two boxes must share vertically
GAP_LINES = 2.0  # largest gap (in line heights) bridged when one piece is short
SHORT_PIECE = 4.0  # a piece narrower than this many line heights counts as short


class LineDetector:
    def __init__(self, weights: Path = WEIGHTS, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LineNet().to(self.device).eval()
        state = torch.load(weights, map_location=self.device)
        self.net.load_state_dict(state["model"])
        if self.device == "cuda":
            self.net.half()

    @torch.no_grad()
    def _maps(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        ph, pw = math.ceil(h / 64) * 64, math.ceil(w / 64) * 64
        padded = cv2.copyMakeBorder(gray, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=255)
        tile = torch.from_numpy(padded)[None, None].to(self.device)
        x = page_input(tile, 0, 0, w, h)
        if self.device == "cuda":
            x = x.half()
        out = self.net(x).float()[0]
        out[0] = torch.sigmoid(out[0])
        out[2:] = torch.softmax(out[2:], 0)
        return out[:, : (h + 1) // 2, : (w + 1) // 2].cpu().numpy()

    def _lines_at(self, gray: np.ndarray, side: int) -> list[dict]:
        h, w = gray.shape
        scale = side / max(h, w)
        small = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        maps = self._maps(small)
        core = (maps[0] > CORE_THRESHOLD).astype(np.uint8)
        count, labels, stats, _cent = cv2.connectedComponentsWithStats(core, connectivity=8)
        lines = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 3:
                continue
            ys, xs = np.nonzero(labels == index)
            weight = maps[0][ys, xs]
            height = float(np.exp(np.median(maps[1][ys, xs])))  # in detector pixels
            types = maps[2:, ys, xs].mean(1)
            # centre line in detector pixels (maps are half resolution)
            fx = xs * 2.0 + 1.0
            fy = ys * 2.0 + 1.0
            if np.ptp(fx) > 2 * height:
                slope, cut = np.polyfit(fx, fy, 1, w=weight)
                slope = float(np.clip(slope, -0.1, 0.1))
                cut = float(np.average(fy - slope * fx, weights=weight))
            else:
                slope, cut = 0.0, float(np.average(fy, weights=weight))
            reach = 0.25 * height + 2
            left, right = fx.min() - reach, fx.max() + reach
            half = height / 2
            quad = np.array([
                [left, slope * left + cut - half], [right, slope * right + cut - half],
                [right, slope * right + cut + half], [left, slope * left + cut + half],
            ]) / scale
            quad[:, 0] = quad[:, 0].clip(0, w)
            quad[:, 1] = quad[:, 1].clip(0, h)
            lines.append({
                "quad": quad.round(1).tolist(),
                "box": (int(quad[:, 0].min()), int(quad[:, 1].min()), int(math.ceil(quad[:, 0].max())), int(math.ceil(quad[:, 1].max()))),
                "type": TYPE_NAMES[int(types.argmax())],
                "height": height / scale,
                "score": float(weight.mean()),
            })
        return lines

    def detect(self, gray: np.ndarray, side: int = DET_SIDE) -> list[dict]:
        """Lines on a gray page (dark text on light), boxes in the page's own pixels."""
        lines = self._lines_at(gray, side)
        if lines:
            scale = side / max(gray.shape)
            typical = float(np.median([line["height"] for line in lines])) * scale
            if typical < MIN_LINE_H and side < 3200:
                # small print: read again larger so thin lines don't merge
                bigger = min(3200, int(side * MIN_LINE_H * 1.6 / max(typical, 1)))
                lines = self._lines_at(gray, bigger)
        return join_row_pieces(lines)


def join_row_pieces(lines: list[dict]) -> list[dict]:
    """Join boxes that belong to one row (see ROW_OVERLAP / GAP_LINES / SHORT_PIECE).
    Two long lines side by side with a real gap (newspaper columns) are never joined."""
    if len(lines) < 2:
        return lines
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(lines)), key=lambda i: lines[i]["box"][1])
    for a_pos, a in enumerate(order):
        ax0, ay0, ax1, ay1 = lines[a]["box"]
        for b in order[a_pos + 1:]:
            bx0, by0, bx1, by1 = lines[b]["box"]
            if by0 >= ay1:
                break  # sorted by top: nothing further down can share this row
            shared = min(ay1, by1) - max(ay0, by0)
            height = min(ay1 - ay0, by1 - by0)
            if shared < ROW_OVERLAP * height:
                continue
            gap = max(bx0, ax0) - min(bx1, ax1)  # negative when the boxes overlap side by side
            line_h = min(lines[a]["height"], lines[b]["height"])
            short = min(ax1 - ax0, bx1 - bx0) < SHORT_PIECE * line_h
            if gap <= 0 or (short and gap < GAP_LINES * line_h):
                parent[find(a)] = find(b)
    groups: dict[int, list[dict]] = {}
    for i, line in enumerate(lines):
        groups.setdefault(find(i), []).append(line)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        x0 = min(m["box"][0] for m in members)
        y0 = min(m["box"][1] for m in members)
        x1 = max(m["box"][2] for m in members)
        y1 = max(m["box"][3] for m in members)
        widest = max(members, key=lambda m: m["box"][2] - m["box"][0])
        out.append({
            "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "box": (x0, y0, x1, y1),
            "type": widest["type"],
            "height": max(m["height"] for m in members),
            "score": float(np.mean([m["score"] for m in members])),
        })
    return out


# ---------------------------------------------------------------- reading order


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _xy_order(lines: list[dict]) -> list[list[dict]]:
    """Recursive XY cut over line boxes. Returns blocks (lists of lines) in reading order:
    columns right to left, rows top to bottom. Couplets (two half lines side by side)
    stay paired, right half first."""
    if len(lines) <= 1:
        return [lines] if lines else []
    heights = np.median([l["box"][3] - l["box"][1] for l in lines])
    x0 = min(l["box"][0] for l in lines)
    x1 = max(l["box"][2] for l in lines)

    # columns: x gaps no line crosses
    cols = _column_cut(lines, x0, x1, heights, tolerance=0)
    if cols:
        return _columns(cols, heights)

    # rows: cut only beside a line that spans most of this region (a headline, a full line)
    groups = _row_cut(lines, x0, x1, heights)
    if len(groups) > 1:
        out = []
        for group in groups:
            out.extend(_xy_order(group) if len(group) > 1 else [group])
        return _join_rows(out)

    # columns again, now letting a stray line or two (a long last line, a speck) cross a gutter
    cols = _column_cut(lines, x0, x1, heights, tolerance=max(1, len(lines) // 40))
    if cols:
        return _columns(cols, heights)
    return [sorted(lines, key=lambda l: (l["box"][1], -l["box"][2]))]


# a page photographed or scanned slightly tilted makes its gutters lean: try these slants
# (x shift per pixel of height, about +-1.1 degrees) when a straight gutter is not found
SLANTS = (0.0, 0.004, -0.004, 0.008, -0.008, 0.012, -0.012, 0.016, -0.016, 0.02, -0.02)


def _column_cut(lines: list[dict], x0: int, x1: int, heights: float, tolerance: int) -> list[list[dict]] | None:
    """Split at vertical (or slightly leaning) gutters crossed by at most `tolerance` lines;
    columns right to left."""
    middle = (min(l["box"][1] for l in lines) + max(l["box"][3] for l in lines)) / 2
    for slant in SLANTS:
        shift = [slant * ((l["box"][1] + l["box"][3]) / 2 - middle) for l in lines]
        lo = int(np.floor(min(l["box"][0] + d for l, d in zip(lines, shift))))
        hi = int(np.ceil(max(l["box"][2] + d for l, d in zip(lines, shift))))
        cols = _cut_at(lines, shift, lo, hi, heights, tolerance)
        if cols:
            return cols
    return None


def _cut_at(lines: list[dict], shift: list[float], x0: int, x1: int, heights: float, tolerance: int) -> list[list[dict]] | None:
    cover = np.zeros(x1 - x0 + 2, np.int32)
    for l, d in zip(lines, shift):
        cover[int(round(l["box"][0] + d)) - x0 : int(round(l["box"][2] + d)) - x0] += 1
    free = cover <= tolerance
    min_gap = max(4, heights * 0.4)
    cuts = []
    start = None
    for x, ok in enumerate(free):
        if ok and start is None:
            start = x
        elif not ok and start is not None:
            if start > 0 and x - start >= min_gap:
                cuts.append(x0 + (start + x) / 2)
            start = None
    if not cuts:
        return None
    edges = [x0 - 1] + cuts + [x1 + 1]
    centre = [(l["box"][0] + l["box"][2]) / 2 + d for l, d in zip(lines, shift)]
    cols = [[l for l, c in zip(lines, centre) if lo <= c < hi] for lo, hi in zip(edges, edges[1:])]
    cols = [c for c in cols if c][::-1]  # right column first
    return cols if len(cols) > 1 else None


def _columns(cols: list[list[dict]], heights: float) -> list[list[dict]]:
    if len(cols) == 2 and _couplets(cols[0], cols[1], heights):
        pairs = sorted(cols[0] + cols[1], key=lambda l: ((l["box"][1] + l["box"][3]) / 2 // max(heights, 1), -l["box"][2]))
        return [pairs]
    return [block for col in cols for block in _xy_order(col)]


def _row_cut(lines: list[dict], x0: int, x1: int, heights: float) -> list[list[dict]]:
    ordered = sorted(lines, key=lambda l: l["box"][1])
    wide = [(l["box"][2] - l["box"][0]) > 0.6 * (x1 - x0) for l in ordered]
    groups, current = [], [ordered[0]]
    bottom = ordered[0]["box"][3]
    for prev_wide, (line, is_wide) in zip(wide, zip(ordered[1:], wide[1:])):
        gap = line["box"][1] - bottom
        if gap > -heights * 0.2 and (is_wide or prev_wide):
            groups.append(current)
            current = [line]
        else:
            current.append(line)
        bottom = max(bottom, line["box"][3])
    groups.append(current)
    return groups


def _couplets(right: list[dict], left: list[dict], height: float) -> bool:
    if len(right) != len(left) or len(right) > 12:
        return False
    for a, b in zip(sorted(right, key=lambda l: l["box"][1]), sorted(left, key=lambda l: l["box"][1])):
        if _overlap(a["box"][1], a["box"][3], b["box"][1], b["box"][3]) < 0.5 * height:
            return False
    return True


def _join_rows(blocks: list[list[dict]]) -> list[list[dict]]:
    """Consecutive single full-width lines of the same type belong to one paragraph block."""
    out: list[list[dict]] = []
    for block in blocks:
        if out and len(block) == 1 and out[-1] and out[-1][-1]["type"] == block[0]["type"] and len(out[-1]) >= 1 \
                and block[0]["box"][1] - out[-1][-1]["box"][3] < 1.2 * (block[0]["box"][3] - block[0]["box"][1]):
            out[-1].extend(block)
        else:
            out.append(block)
    return out


def reading_order(lines: list[dict]) -> list[list[dict]]:
    """Blocks in reading order: header lines, body (XY cut), footnotes, footer lines."""
    by = {name: [l for l in lines if l["type"] == name] for name in TYPE_NAMES}
    row = lambda ls: sorted(ls, key=lambda l: ((l["box"][1] + l["box"][3]) / 2, -l["box"][2]))
    blocks: list[list[dict]] = []
    if by["header"]:
        blocks.append(row(by["header"]))
    blocks.extend(_xy_order(by["body"] + by["heading"]))
    if by["footnote"]:
        blocks.extend(_xy_order(by["footnote"]))
    if by["footer"]:
        blocks.append(row(by["footer"]))
    return [b for b in blocks if b]
