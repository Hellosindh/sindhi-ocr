"""Page layout network: finds regions (title, heading, text, table, picture, ...) on a whole page.

Same small U-Net family as linedet.LineNet, output at 1/4 resolution:
    0 .. N_CLASSES      class logits per pixel (0 = background), painted only on the central
                        "core" of each region so touching regions stay apart
    N_CLASSES+1 .. +4   distances from the pixel to the region's left, top, right and bottom edge,
                        as a fraction of the input side (read on core pixels)
Boxes come from grouping core pixels of one class and averaging their edge predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from linedet.model import Down, Up, conv

CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
           "Picture", "Section-header", "Table", "Text", "Title",
           "Contents-entry",  # one line of a contents / index page: title ... page number (Sindhi pages only)
           "Masthead",        # newspaper name plate on the front page
           "Jump-line"]       # "ڏسو صفحو 8 ڪالم 3": the story goes on at page 8, column 3
N_CLASSES = len(CLASSES)
STRIDE = 4
CONTENTS = CLASSES.index("Contents-entry")  # decoded as one block, then cut into lines by split_lines
PER_LINE = {CONTENTS}
GROUP_BY_TOP = {CLASSES.index("List-item")}  # blob split where pixels disagree on the item's top edge
SIZE = 640  # pages are letterboxed to SIZE x SIZE


class LayoutNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        c = [width, width * 2, width * 4, width * 6, width * 8, width * 8]
        self.stem = nn.Sequential(conv(3, c[0], 2), conv(c[0], c[0]))  # /2
        self.d1 = Down(c[0], c[1])  # /4
        self.d2 = Down(c[1], c[2])  # /8
        self.d3 = Down(c[2], c[3])  # /16
        self.d4 = Down(c[3], c[4])  # /32
        self.d5 = Down(c[4], c[5])  # /64: sees the whole page column at once
        self.u4 = Up(c[5], c[4], c[4])
        self.u3 = Up(c[4], c[3], c[3])
        self.u2 = Up(c[3], c[2], c[2])
        self.u1 = Up(c[2], c[1], c[1])
        self.n_classes = N_CLASSES
        self.cls = nn.Sequential(conv(c[1], c[1]), nn.Conv2d(c[1], N_CLASSES + 1, 1))
        self.box = nn.Sequential(conv(c[1], c[1]), nn.Conv2d(c[1], 4, 1))

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        # weights trained before classes were added (e.g. 12 classes before Masthead / Jump-line): shrink
        # the class head to match, so old and new layout weights both load; class ids keep their names
        head = state.get("cls.1.weight")
        if type(self) is LayoutNet and head is not None and head.shape[0] != self.cls[1].out_channels:
            self.cls[1] = nn.Conv2d(self.cls[1].in_channels, head.shape[0], 1)
            self.n_classes = head.shape[0] - 1
        return super().load_state_dict(state, strict=strict, assign=assign)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        y = self.u4(s5, s4)
        y = self.u3(y, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        return torch.cat([self.cls(y), F.softplus(self.box(y))], 1)


def net_input(gray: torch.Tensor) -> torch.Tensor:
    """(B,1,S,S) 0..255 gray -> (B,3,S,S): ink-high image plus x/y coordinates in -1..1."""
    b, _c, h, w = gray.shape
    xs = torch.linspace(-1, 1, w, device=gray.device).view(1, 1, 1, w).expand(b, 1, h, w)
    ys = torch.linspace(-1, 1, h, device=gray.device).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([1.0 - gray.float() / 255.0, xs, ys], 1)


def _overlap(a, b) -> tuple[float, float]:
    """(IoU, share of the smaller box covered by the other)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    small = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0), (inter / small if small > 0 else 0.0)


def merge_duplicates(boxes: list[tuple], iou: float = 0.5, inside: float = 0.85) -> list[tuple]:
    """Keep one box where boxes of the same class overlap a lot or one sits inside another
    (bigger, surer boxes win)."""
    kept: list[tuple] = []
    for box in sorted(boxes, key=lambda b: -b[5] * ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5):
        clash = False
        for k in kept:
            if k[4] == box[4]:
                o_iou, o_inside = _overlap(box, k)
                if o_iou >= iou or o_inside >= inside:
                    clash = True
                    break
        if not clash:
            kept.append(box)
    return kept


def decode(out: torch.Tensor, min_score: float = 0.4, min_cells: int = 2) -> list[list[tuple]]:
    """Network output (B, C, H, W) -> per image [(x0, y0, x1, y1, class, score)] in input pixels."""
    import numpy as np
    from scipy import ndimage

    results = []
    n = out.shape[1] - 5  # class count of the weights that made `out`
    prob = out[:, : n + 1].float().softmax(1)
    edges = out[:, n + 1 :].float()
    side = out.shape[-1] * STRIDE
    for b in range(out.shape[0]):
        p = prob[b].cpu().numpy()
        e = edges[b].cpu().numpy() * side
        label = p.argmax(0)
        boxes = []
        for k in range(1, n + 1):
            mask = (label == k) & (p[k] > min_score)
            if mask.sum() < min_cells:
                continue
            comps, n = ndimage.label(mask)
            for idx in range(1, n + 1):
                ys, xs = np.nonzero(comps == idx)
                if len(ys) < min_cells:
                    continue
                w = p[k, ys, xs]
                cx = (xs + 0.5) * STRIDE
                cy = (ys + 0.5) * STRIDE
                if k - 1 == CONTENTS:  # each pixel knows only its own line -> union, split later
                    groups = [None]
                elif k - 1 in GROUP_BY_TOP and len(ys) >= 2 * min_cells:
                    # a run of entries painted as one blob: pixels of one entry agree on its top edge
                    top = cy - e[1, ys, xs]
                    order = np.argsort(top)
                    hgt = np.median(e[1, ys, xs] + e[3, ys, xs])
                    cut = np.nonzero(np.diff(top[order]) > max(2.0, hgt * 0.3))[0] + 1
                    groups = [g for g in np.split(order, cut) if len(g) >= min_cells]
                else:
                    groups = [np.arange(len(ys))]
                for gi in groups:
                    if gi is None:
                        boxes.append((max(0.0, float(np.percentile(cx - e[0, ys, xs], 2))),
                                      max(0.0, float(np.percentile(cy - e[1, ys, xs], 1))),
                                      min(side, float(np.percentile(cx + e[2, ys, xs], 98))),
                                      min(side, float(np.percentile(cy + e[3, ys, xs], 99))), k - 1, float(w.mean())))
                        continue
                    gw = w[gi]
                    gx, gy = cx[gi], cy[gi]
                    x0 = float(np.average(gx - e[0, ys[gi], xs[gi]], weights=gw))
                    y0 = float(np.average(gy - e[1, ys[gi], xs[gi]], weights=gw))
                    x1 = float(np.average(gx + e[2, ys[gi], xs[gi]], weights=gw))
                    y1 = float(np.average(gy + e[3, ys[gi], xs[gi]], weights=gw))
                    boxes.append((max(0.0, x0), max(0.0, y0), min(side, x1), min(side, y1), k - 1, float(gw.mean())))
        results.append(merge_duplicates(boxes))
    return results



PAD = 0.2


def split_lines(boxes: list[tuple], gray, classes=PER_LINE, min_h: int = 3) -> list[tuple]:
    """The net paints a run of contents entries / list items as one blob; cut each such box into
    its text lines by the row ink profile of the image (gray: HxW uint8, same pixels as the boxes)."""
    import numpy as np

    g = np.asarray(gray, np.uint8)
    H, W = g.shape[:2]
    out = []
    for box in boxes:
        if box[4] not in classes:
            out.append(box)
            continue
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]) - 2)
        x1, y1 = min(W, int(np.ceil(box[2]))), min(H, int(np.ceil(box[3])) + 2)
        if x1 - x0 < 4 or y1 - y0 < 2 * min_h:
            out.append(box)
            continue
        import cv2

        crop = g[y0:y1, x0:x1]
        k = max(5, int(min(crop.shape) * 0.15) | 1)
        ink = cv2.dilate(crop, np.ones((k, k), np.uint8)) - crop  # ink against local paper (faded pages too)
        cut = max(10.0, float(np.percentile(ink, 99.5)) * 0.35)
        ink = np.where(ink > cut, ink, 0).astype(np.float32)
        prof = np.convolve(ink.sum(1), np.ones(3) / 3, mode="same")
        if prof.max() <= 0:
            out.append(box)
            continue
        on = prof > float(np.percentile(prof, 80)) * 0.15  # not max: a dark rule or stain would hide faint lines
        runs, start = [], None
        for i, v in enumerate(list(on) + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_h:
                    runs.append([start, i])
                start = None
        if len(runs) < 2:
            out.append(box)
            continue
        # glue thin runs (dots / marks above or below a line) to their nearest neighbour
        heights = sorted(b - a for a, b in runs)
        typical = heights[len(heights) // 2]
        merged = []
        for r in runs:
            if merged and (r[1] - r[0] < typical * 0.45 or merged[-1][1] - merged[-1][0] < typical * 0.45) \
                    and r[0] - merged[-1][1] < typical * 0.5:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        if len(merged) < 2:
            out.append(box)
            continue
        # pad each ink run a little (labels include the full line height), never past the gap middle
        gaps = [(merged[i][1] + merged[i + 1][0]) / 2 for i in range(len(merged) - 1)]
        for i, (a, b) in enumerate(merged):
            pad = (b - a) * PAD
            top = max(a - pad, gaps[i - 1] if i else 0.0)
            bot = min(b + pad, gaps[i] if i < len(gaps) else float(y1 - y0))
            # an entry runs from its title to its page number: the block's full width
            out.append((float(box[0]), float(y0 + top), float(box[2]), float(y0 + bot), box[4], box[5]))
    return out


def _covered(box, others) -> float:
    """Share of box's area covered by the union of others (pixel grid at 2 px)."""
    import numpy as np

    x0, y0, x1, y1 = box[:4]
    if x1 - x0 < 1 or y1 - y0 < 1:
        return 0.0
    w, h = max(1, int((x1 - x0) / 2)), max(1, int((y1 - y0) / 2))
    m = np.zeros((h, w), bool)
    for o in others:
        a, b = int((o[0] - x0) / 2), int((o[1] - y0) / 2)
        c, d = int(np.ceil((o[2] - x0) / 2)), int(np.ceil((o[3] - y0) / 2))
        m[max(0, b):max(0, min(h, d)), max(0, a):max(0, min(w, c))] = True
    return float(m.mean())


CONTAINERS = {CLASSES.index("Table"), CLASSES.index("Picture")}


def cross_class(boxes: list[tuple], iou: float = 0.5, inside: float = 1.01) -> list[tuple]:
    """One region, one label: where boxes of different classes cover the same place (cover titles come
    out as Title + Section-header + Picture), keep the surer one. A text box inside a table / picture
    is kept (captions on art, cells), and so is a container around text."""
    kept: list[tuple] = []
    table = CLASSES.index("Table")
    # tables first: a table read as Text / List-item too keeps its grid (one-row false tables die later)
    for box in sorted(boxes, key=lambda b: (b[4] != table, -b[5])):
        clash = False
        for k in kept:
            if {box[4], k[4]} <= {CONTENTS, *CONTAINERS} and box[4] != k[4]:
                continue  # contents lines vs table / picture: settled in tidy
            o_iou, o_inside = _overlap(box, k)
            if o_iou >= iou:
                clash = True
            elif o_inside >= inside and box[4] not in CONTAINERS and k[4] not in CONTAINERS:
                clash = True  # a small text box inside a bigger text box of another label
            if clash:
                break
        if not clash:
            kept.append(box)
    return kept


def tidy(boxes: list[tuple]) -> list[tuple]:
    """Known false alarms: contents lines inside a picture (striped / textured cover art), one or two
    lone "entries", duplicate labels on one region, and a Table box on a contents list."""
    pic = [b for b in boxes if b[4] == CLASSES.index("Picture")]
    boxes = [b for b in boxes if not (b[4] == CONTENTS and pic and _covered(b, pic) > 0.5)]
    if sum(b[4] == CONTENTS for b in boxes) < 3:  # one or two "entries" alone are not a contents list
        boxes = [(*b[:4], CLASSES.index("Text"), b[5]) if b[4] == CONTENTS else b for b in boxes]
    boxes = cross_class(boxes)
    lines = [b for b in boxes if b[4] == CONTENTS]
    table = CLASSES.index("Table")
    return [b for b in boxes if not (b[4] == table and len(lines) >= 3 and _covered(b, lines) > 0.6)]
