"""Run the line detector over every newspaper page once and cache the lines, so the newspaper
layout labeller (layout/news_labels.py) can be tuned without re-detecting.

Writes assets/news_pages/lines.jsonl: {"path", "w", "h", "lines": [[x0,y0,x1,y1,height,type], ...]}
Resumes: pages already in the file are skipped.

Run:  python -m layout.news_lines
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import math

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from linedet.detect import CORE_THRESHOLD, TYPE_NAMES, LineDetector  # noqa: E402

PAGES = ROOT / "assets" / "news_pages"
OUT = PAGES / "lines.jsonl"


def fast_lines(det: LineDetector, gray: np.ndarray, side: int = 1600) -> list[dict]:
    """LineDetector._lines_at in one pass at 1600 px (no second, larger read for small print: blocks are
    what matter here), reading each component inside its own bounding box instead of the whole map."""
    h, w = gray.shape
    scale = side / max(h, w)
    small = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    maps = det._maps(small)
    core = (maps[0] > CORE_THRESHOLD).astype(np.uint8)
    count, labels, stats, _c = cv2.connectedComponentsWithStats(core, connectivity=8)
    lines = []
    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        if area < 3:
            continue
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == index)
        ys, xs = ys + y, xs + x
        weight = maps[0][ys, xs]
        height = float(np.exp(np.median(maps[1][ys, xs])))
        types = maps[2:, ys, xs].mean(1)
        fx, fy = xs * 2.0 + 1.0, ys * 2.0 + 1.0
        reach = 0.25 * height + 2
        x0, x1 = (fx.min() - reach) / scale, (fx.max() + reach) / scale
        cy = float(np.average(fy, weights=weight))
        y0, y1 = (fy.min() - height / 2) / scale, (fy.max() + height / 2) / scale
        lines.append({"box": (int(max(0, x0)), int(max(0, y0)), int(min(w, math.ceil(x1))), int(min(h, math.ceil(y1)))),
                      "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                      "type": TYPE_NAMES[int(types.argmax())], "height": height / scale, "score": float(weight.mean())})
    return lines  # no join_row_pieces: on narrow-gutter news pages it glues neighbouring columns' lines


def main() -> None:
    done = set()
    if OUT.exists():
        done = {json.loads(l)["path"] for l in OUT.open(encoding="utf-8")}
    paths = sorted(p.relative_to(PAGES).as_posix() for p in PAGES.glob("*/*/page_*.jpg"))
    todo = [p for p in paths if p not in done]
    __import__("random").Random(0).shuffle(todo)  # mixed years first, so the rules can be tuned while it runs
    print(f"{len(paths)} pages, {len(done)} cached, {len(todo)} to do", flush=True)
    det = LineDetector()
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, rel in enumerate(todo):
            gray = cv2.imread(str(PAGES / rel), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            lines = fast_lines(det, gray)
            f.write(json.dumps({"path": rel, "w": gray.shape[1], "h": gray.shape[0],
                                "lines": [[*l["box"], round(l["height"], 1), l["type"]] for l in lines]}) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s ({(time.time() - t0) / (i + 1):.2f}s/page)", flush=True)
    print(f"done {len(todo)} pages in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
