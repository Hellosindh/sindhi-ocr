"""Synthetic right-to-left Sindhi table crops with exact row / column gaps, for layout.tables.

Each table is drawn by layout.synth_layout.table() on a blank canvas; the cell text boxes it places
give the blank bands between rows and between columns. The image gets paper tone, faded ink,
noise, blur and JPEG loss, but no geometric warp (the gaps must stay straight).

Run:  python -m layout.synth_tables [--count 8000] [--test 400]
Out:  assets/layout_data/sindhi_tables/{train,test}/NNNNNNN.jpg + labels.jsonl
      {"image", "w", "h", "table": [x0, y0, x1, y1], "rows": [[a, b], ...], "cols": [[a, b], ...], "header": [a, b]}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import linedet.synth_pages as sp
from layout.synth_layout import REGION, table

OUT = ROOT / "assets" / "layout_data" / "sindhi_tables"


def bands(ext: dict[int, list[float]]) -> list[list[float]]:
    keys = sorted(ext)
    out = []
    for k0, k1 in zip(keys, keys[1:]):
        a, b = ext[k0][1], ext[k1][0]
        if b - a < 2:
            m = (a + b) / 2
            a, b = m - 1, m + 1
        out.append([round(a, 1), round(b, 1)])
    return out


def make(seed: int):
    rng = random.Random(seed)
    font = rng.choice(sp.usable_fonts())
    src = sp.WordSource(rng)
    width = rng.randint(700, 1400)
    page = sp.Page(width, 1400, rng)
    size = rng.randint(13, 26)
    x0, x1 = 30, width - 30
    start = len(page.lines)
    table(page, src, font, size, x0, x1, 30, 1370)
    geo = getattr(page, "last_table", None)
    if geo is None:
        return None
    quads = [np.array(q) for q, k in page.lines[start:] if k < REGION]
    if len(quads) < 4:
        return None
    rows_ext: dict[int, list[float]] = {}
    cols_ext: dict[int, list[float]] = {}
    edges = [x1]
    for w in geo["col_w"]:
        edges.append(edges[-1] - w)  # right-to-left: column 0 is on the right
    for q in quads:
        qx0, qy0, qx1, qy1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        r = int(((qy0 + qy1) / 2 - geo["top"]) // geo["row_h"])
        cx = (qx0 + qx1) / 2
        c = next((i for i in range(len(edges) - 1) if edges[i + 1] <= cx <= edges[i]), None)
        e = rows_ext.setdefault(r, [qy0, qy1])
        e[0], e[1] = min(e[0], qy0), max(e[1], qy1)
        if c is not None:
            # columns listed left to right in image space so the gaps come out in order
            col_key = len(edges) - 2 - c
            e = cols_ext.setdefault(col_key, [qx0, qx1])
            e[0], e[1] = min(e[0], qx0), max(e[1], qx1)
    header = rows_ext.get(0)
    # photometric look only
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    ink = page.ink
    if rng.random() < 0.3:
        ink = ink * cv2.resize(nrng.uniform(0.55, 1.0, (6, 6)).astype(np.float32), (page.w, page.h))
    paper = rng.uniform(205, 252)
    img = paper * (1 - ink) + rng.uniform(0, 70) * ink
    img += nrng.normal(0, rng.uniform(0, 9), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.1))
    top, bottom = geo["top"] - 2, geo["bottom"] + 2
    m = rng.randint(4, 30)
    cy0, cy1 = max(0, int(top - m)), min(page.h, int(bottom + m))
    cx0, cx1 = max(0, int(x0 - m)), min(page.w, int(x1 + m))
    crop = img[cy0:cy1, cx0:cx1]
    shift_y = lambda v: round(float(v) - cy0, 1)  # noqa: E731  (plain floats for JSON)
    shift_x = lambda v: round(float(v) - cx0, 1)  # noqa: E731
    label = {
        "w": int(crop.shape[1]), "h": int(crop.shape[0]),
        "table": [shift_x(x0 - 2), shift_y(top), shift_x(x1 + 2), shift_y(bottom)],
        "rows": [[shift_y(a), shift_y(b)] for a, b in bands(rows_ext)],
        "cols": [[shift_x(a), shift_x(b)] for a, b in bands(cols_ext)],
        "header": [shift_y(header[0]), shift_y(header[1])] if header else None,
    }
    return crop, label


def write(task) -> str | None:
    index, split = task
    made = make(index * 104729 + (5 if split == "train" else 777_001))
    if made is None:
        return None
    crop, label = made
    name = f"{index:07d}.jpg"
    cv2.imwrite(str(OUT / split / name), crop, [cv2.IMWRITE_JPEG_QUALITY, random.Random(index).randint(55, 92)])
    label["image"] = name
    return json.dumps(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()
    for split, n in (("test", args.test), ("train", args.count)):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        kept = 0
        with ProcessPoolExecutor(args.workers) as pool, (OUT / split / "labels.jsonl").open("w", encoding="utf-8") as out:
            for line in pool.map(write, [(i, split) for i in range(n)], chunksize=16):
                if line:
                    out.write(line + "\n")
                    kept += 1
        print(f"{split}: {kept}/{n} tables", flush=True)


if __name__ == "__main__":
    main()
