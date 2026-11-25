"""First-pass layout labels for real newspaper pages, built from cached line-detector output
(layout/news_lines.py) plus the page's ink.

Rules
  - line taller than HEAD x median line height -> headline line, taller than TITLE x -> main headline
  - body lines stacked in one column (overlapping side by side, small vertical gap) -> Text block
  - headline lines stacked together -> Section-header (sub headline) or Title (main headline)
  - lines the detector calls header / footer -> Page-header / Page-footer
  - large ink areas with no text lines in them -> Picture
  - page 1: wide block at the very top (the paper's name) -> Masthead

Run:  python -m layout.news_labels --show 8        (draw 8 pages to layout/news_out to check the rules)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGES = ROOT / "assets" / "news_pages"
CACHE = PAGES / "lines.jsonl"
HEAD, TITLE, MASTHEAD = 1.6, 2.6, 10.0
NAMES = ["Text", "Section-header", "Title", "Page-header", "Page-footer", "Picture", "Masthead"]
COLORS = {"Text": (60, 160, 60), "Section-header": (220, 120, 0), "Title": (220, 0, 0), "Page-header": (160, 0, 160),
          "Page-footer": (160, 0, 160), "Picture": (0, 120, 220), "Masthead": (0, 200, 200), "Jump-line": (200, 0, 200)}


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def group(lines, side_share, gap):
    """Union lines that sit in one column: side-by-side overlap >= side_share of the narrower one and
    vertical gap <= gap x the taller one's height."""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: lines[i][1])
    for a_i, i in enumerate(order):
        a = lines[i]
        for j in order[a_i + 1:]:
            b = lines[j]
            if b[1] - a[3] > gap * max(a[4], b[4]):
                break
            ov = overlap(a[0], a[2], b[0], b[2])
            narrow, wide = sorted((a[2] - a[0], b[2] - b[0]))
            # same column: they share most of the wider line, or the shorter one starts at the same right
            # edge (a paragraph's short last line; Sindhi lines start on the right)
            same = ov >= side_share * wide or (ov >= 0.8 * narrow and abs(a[2] - b[2]) <= 1.5 * max(a[4], b[4]))
            if same and b[1] - a[3] <= gap * max(a[4], b[4]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def join_pieces(lines, med):
    """Join pieces of one row split at a word gap. A gap is a word gap when body lines just above or below
    run through it; a column gutter has none crossing it (only wide headlines, left out here)."""
    body = np.array([l[:4] for l in lines if l[4] < HEAD * med], float).reshape(-1, 4)
    bcy = (body[:, 1] + body[:, 3]) / 2
    lines = sorted(lines, key=lambda l: l[0])
    used = [False] * len(lines)
    out = []
    for i, a in enumerate(lines):
        if used[i]:
            continue
        cur = list(a)
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(lines)):
                b = lines[j]
                if used[j] or b[0] < cur[0]:
                    continue
                h = max(cur[4], b[4])
                gap = b[0] - cur[2]
                if gap > 2 * h or gap < -h:
                    continue
                if overlap(cur[1], cur[3], b[1], b[3]) < 0.6 * min(cur[3] - cur[1], b[3] - b[1]):
                    continue
                if max(cur[4], b[4]) > 1.4 * min(cur[4], b[4]) or min(cur[4], b[4]) >= MASTHEAD * med:
                    continue
                xc, cy = (cur[2] + b[0]) / 2, (cur[1] + cur[3] + b[1] + b[3]) / 4
                dy = np.abs(bcy - cy)  # body lines just above / below this row
                near = (dy < h / 2 + 3 * med) & (dy > 0.6 * h)
                cross = int(((body[:, 0] < xc - med) & (body[:, 2] > xc + med) & near).sum())
                if cross >= 3:
                    cur = [min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3]),
                           max(cur[4], b[4]), cur[5]]
                    used[j] = True
                    changed = True
        out.append(cur)
    return out


def box_of(ls):
    return [min(l[0] for l in ls), min(l[1] for l in ls), max(l[2] for l in ls), max(l[3] for l in ls)]


def pictures(gray, lines, med):
    """Big ink areas the line detector found no text in."""
    ink = (gray < 140).astype(np.uint8)
    pad = int(med * 0.4)
    for x0, y0, x1, y1, *_ in lines:
        ink[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 0
    # column rules and box frames are long thin ink: drop them before joining
    k = max(3, int(med * 0.5))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    ink = cv2.dilate(ink, np.ones((int(med), int(med)), np.uint8))
    count, _lab, stats, _c = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w > 5 * med and h > 4 * med and area > 0.35 * w * h:
            out.append([int(x), int(y), int(x + w), int(y + h)])
    return out


def tidy_boxes(boxes):
    """Merge boxes of one class that overlap a lot (the network paints one class per pixel, so
    overlapping truths fight), and drop small boxes lying inside a bigger box of any class."""
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    changed = True
    while changed:
        changed = False
        boxes.sort(key=lambda b: -area(b))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if not a[4] or not b[4]:
                    continue
                inter = overlap(a[0], a[2], b[0], b[2]) * overlap(a[1], a[3], b[1], b[3])
                if inter <= 0:
                    continue
                if inter >= 0.6 * area(b) and (a[4] == b[4] or a[4] in ("Picture", "Masthead") or area(b) < 0.25 * area(a)):
                    b[4] = None  # inside a bigger box
                elif a[4] == b[4] and inter >= 0.5 * min(area(a), area(b)):
                    a[:4] = box_of([a, b])
                    b[4] = None
                    changed = True
        boxes = [b for b in boxes if b[4]]
    return boxes


def label_page(row, gray):
    """-> boxes [x0, y0, x1, y1, name]; row["lines"] are replaced by the joined lines (jump_lines reads them)."""
    lines = [l for l in row["lines"] if l[2] - l[0] > 2 and l[3] - l[1] > 2]
    if len(lines) < 10:
        return []
    med = float(np.median([l[4] for l in lines]))
    lines = join_pieces(lines, med)
    row["lines"] = lines
    W, H = row["w"], row["h"]
    boxes = []
    edge = [l for l in lines if l[5] in ("header", "footer")]
    rest = [l for l in lines if l[5] not in ("header", "footer")]
    for l in edge:
        boxes.append([*l[:4], "Page-header" if l[5] == "header" else "Page-footer"])
    body = [l for l in rest if l[4] < HEAD * med]
    head = [l for l in rest if l[4] >= HEAD * med]
    for g in group(body, 0.6, 0.9):
        boxes.append([*box_of(g), "Text"])
    for g in group(head, 0.3, 0.7):
        tall = np.median([l[4] for l in g]) >= TITLE * med
        boxes.append([*box_of(g), "Title" if tall else "Section-header"])
    for b in pictures(gray, lines, med):
        x0, y0, x1, y1 = b
        dark = float((gray[y0:y1, x0:x1] < 100).mean())
        # a wide, mostly black band is a white-on-black headline bar (the detector reads dark-on-light only)
        bar = (x1 - x0) > 3.5 * (y1 - y0) and (y1 - y0) < 8 * med and dark > 0.45
        boxes.append([*b, "Section-header" if bar else "Picture"])
    if row["path"].endswith("page_01.jpg"):  # the paper's name: a giant "line" near the top of the front page
        giant = [l for l in lines if l[4] >= MASTHEAD * med and l[1] < 0.2 * H]
        if giant:
            m = list(box_of(giant))
            # the logo is often white letters on a black plate: take the plate around the giant line
            ink = cv2.morphologyEx((gray < 110).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((int(2 * med),) * 2, np.uint8))
            _n, lab, st, _c = cv2.connectedComponentsWithStats(ink, connectivity=8)
            cx, cy = (m[0] + m[2]) // 2, (m[1] + m[3]) // 2
            k = lab[cy, cx]
            if k:
                x, y, w, h, _a = st[k]
                if w * h < 0.15 * W * H and y < 0.2 * H:
                    m = box_of([m, [x, y, x + w, y + h]])
            for b in boxes:  # take in the logo's frame / pieces that touch it
                if b[4] == "Picture" and overlap(b[0], b[2], m[0], m[2]) > 0 and overlap(b[1], b[3], m[1], m[3]) > 0:
                    m = box_of([m, b])
            boxes = [b for b in boxes if not (b[0] >= m[0] - 2 and b[2] <= m[2] + 2 and b[1] >= m[1] - 2 and b[3] <= m[3] + 2)]
            boxes.append([*m, "Masthead"])
    return tidy_boxes(boxes)


class Reader:
    """The fine-tuned CRNN, only to spot jump lines ("ڏسو صفحو 8 ڪالم 3") among short lines."""
    JUMP = re.compile(r"صفح|ڪالم|كالم")

    def __init__(self):
        import torch
        from model.charset import Charset
        from model.crnn import CRNN
        self.torch, self.cs = torch, Charset()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ROOT / "model" / "weights" / "crnn_finetune_best.pt", map_location="cpu", weights_only=False)
        self.m = CRNN(ck["classes"]).eval().to(self.dev)
        self.m.load_state_dict(ck["state"])

    def read(self, gray, boxes):
        from PIL import Image
        from model.decode import greedy
        from model.order import to_logical
        from model.overfit import to_tensor
        torch = self.torch
        ts = []
        for x0, y0, x1, y1 in boxes:
            pad = max(2, (y1 - y0) // 4)
            crop = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            ts.append(to_tensor(Image.fromarray(crop)))
        out = []
        for i in range(0, len(ts), 64):
            chunk = ts[i:i + 64]
            W = max(t.shape[-1] for t in chunk)
            x = torch.ones(len(chunk), 1, chunk[0].shape[1], W)
            x = torch.zeros_like(x)
            for j, t in enumerate(chunk):
                x[j, :, :, :t.shape[-1]] = t
            with torch.no_grad():
                lp = self.m(x.to(self.dev)).log_softmax(-1).cpu()
            for j, t in enumerate(chunk):
                T = max(1, lp.shape[1] * t.shape[-1] // W)
                out.append(to_logical(greedy(lp[j, :T], self.cs), self.cs))
        return out


def jump_lines(row, gray, boxes, reader):
    """Short lines (last line of a Text block, or a block of one or two lines) whose text names a page /
    column become Jump-line boxes cut out of their Text block."""
    lines = [l for l in row["lines"] if l[5] not in ("header", "footer")]
    cands = []
    for b in boxes:
        if b[4] != "Text":
            continue
        inside = sorted([l for l in lines if l[0] >= b[0] - 1 and l[2] <= b[2] + 1 and l[1] >= b[1] - 1 and l[3] <= b[3] + 1],
                        key=lambda l: l[1])
        if inside and (len(inside) <= 2 or inside[-1][2] - inside[-1][0] < 0.8 * (b[2] - b[0])):
            cands.append((b, inside))
    if not cands:
        return boxes, []
    texts = reader.read(gray, [c[1][-1][:4] for c in cands])
    hits = []
    for (b, inside), text in zip(cands, texts):
        if Reader.JUMP.search(text):
            j = inside[-1]
            hits.append(text)
            boxes.append([*j[:4], "Jump-line"])
            rest = inside[:-1]
            if rest:
                b[:4] = box_of(rest)
            else:
                b[4] = None
    return [b for b in boxes if b[4]], hits


def draw(gray, boxes):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1, name in boxes:
        c = COLORS[name][::-1]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 3)
        cv2.putText(img, name, (int(x0) + 3, int(y0) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    reader = None if a.no_ocr else Reader()
    rows = [json.loads(l) for l in CACHE.open(encoding="utf-8")]
    out = Path(__file__).resolve().parent / "news_out"
    out.mkdir(exist_ok=True)
    for row in random.Random(a.seed).sample(rows, min(a.show, len(rows))):
        gray = cv2.imread(str(PAGES / row["path"]), cv2.IMREAD_GRAYSCALE)
        boxes = label_page(row, gray)
        hits = []
        if reader:
            boxes, hits = jump_lines(row, gray, boxes, reader)
        name = row["path"].replace("/", "_")
        cv2.imwrite(str(out / name), draw(gray, boxes))
        print(name, len(boxes), {k: sum(b[4] == k for b in boxes) for k in NAMES + ["Jump-line"] if any(b[4] == k for b in boxes)}, hits[:3])


if __name__ == "__main__":
    main()
