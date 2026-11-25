"""Check contents entries with the line reader: a real entry ends in a page number at its left end
(right-to-left: title ... number). Imprint lines ("key : value") and play dialogue ("name : speech")
look like entries to the layout net but mostly don't end in a number.

    boxes = confirm_contents(gray, boxes)   # gray: page pixels the boxes refer to
A run of entries where under half end in a number becomes one Text box.
"""

from __future__ import annotations

import numpy as np

from layout.model import CLASSES, CONTENTS

DIGITS = set("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
TEXT = CLASSES.index("Text")
_charset = None


def _runs(entries: list[tuple]) -> list[list[tuple]]:
    """Entries stacked one under another (overlapping in x, small vertical gaps) form one run."""
    runs: list[list[tuple]] = []
    for e in sorted(entries, key=lambda b: b[1]):
        h = e[3] - e[1]
        for run in runs:
            last = run[-1]
            overlap = min(e[2], last[2]) - max(e[0], last[0])
            if overlap > 0.5 * min(e[2] - e[0], last[2] - last[0]) and e[1] - last[3] < 2.0 * h:
                run.append(e)
                break
        else:
            runs.append([e])
    return runs


def _left_end(gray: np.ndarray, box: tuple, narrow: bool = False) -> np.ndarray | None:
    """The leftmost piece of ink of an entry (where its page number sits), about a quarter wide."""
    x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None
    crop = gray[y0:y1, x0:x1]
    ink = (crop < min(200, float(np.percentile(crop, 50)) - 40)).sum(0)
    cols = np.nonzero(ink > 0)[0]
    if len(cols) == 0:
        return None
    left = cols[0]
    width = int((y1 - y0) * 1.8) if narrow else max(int((y1 - y0) * 3), int((x1 - x0) * 0.22))
    return crop[:, left : min(crop.shape[1], left + width)]


def _has_number(crops: list[np.ndarray]) -> list[bool]:
    import torch
    from PIL import Image

    from layout.orient import _load_reader
    from model.charset import Charset
    from model.decode import greedy
    from model.overfit import to_tensor

    global _charset
    if _charset is None:
        _charset = Charset()
    net, device = _load_reader()
    tensors = [to_tensor(Image.fromarray(c)) for c in crops]
    width = max(t.shape[-1] for t in tensors)
    batch = torch.zeros(len(tensors), 1, tensors[0].shape[1], width)
    for i, t in enumerate(tensors):
        batch[i, :, :, : t.shape[-1]] = t
    with torch.no_grad():
        lp = net(batch.to(device)).log_softmax(-1).cpu()
    out = []
    for i, t in enumerate(tensors):
        frames = max(1, lp.shape[1] * t.shape[-1] // width)
        out.append(any(ch in DIGITS for ch in greedy(lp[i, :frames], _charset)))
    return out


def confirm_contents(gray: np.ndarray, boxes: list[tuple]) -> list[tuple]:
    """Keep a run of entries as contents when enough of them end in a number: a quarter of a long
    run (numbers are small and often faint), two thirds of a short one (two or three lines of
    "key : 500" are not a contents list)."""
    entries = [b for b in boxes if b[4] == CONTENTS]
    if not entries:
        return boxes
    others = [b for b in boxes if b[4] != CONTENTS]
    for run in _runs(entries):
        hits = 0
        pairs = [(_left_end(gray, e), _left_end(gray, e, narrow=True)) for e in run]
        flat = [c for pair in pairs for c in pair if c is not None and c.size]
        votes = iter(_has_number(flat)) if flat else iter(())
        for pair in pairs:
            found = False
            for c in pair:
                if c is not None and c.size:
                    found = next(votes) or found
            hits += found
        need = 0.25 if len(run) >= 4 else 0.67
        if hits / len(run) >= need:
            others.extend(run)
        else:  # not a contents list: one Text block
            others.append((min(e[0] for e in run), min(e[1] for e in run), max(e[2] for e in run),
                           max(e[3] for e in run), TEXT, float(np.mean([e[5] for e in run]))))
    return others
