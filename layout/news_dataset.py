"""Turn the auto-labelled newspaper pages into a layout training set: assets/layout_data/news/{train,test}.

Whole issues (days) are held out for test, spread across the years, so the test pages come from papers the
model never trained on. Images stay in assets/news_pages (labels point there), nothing is copied.

The test labels are the same automatic labels, not hand-drawn: the test score says how well the model learned
the rules, and the drawn pages show whether that looks right.

Run:  python -m layout.news_dataset
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from layout.model import CLASSES  # noqa: E402
from layout.news_labels import CACHE, PAGES, Reader, jump_lines, label_page  # noqa: E402

OUT = ROOT / "assets" / "layout_data" / "news"
TEST_DAYS = 8  # one issue from each era bucket below


def main() -> None:
    rows = [json.loads(l) for l in CACHE.open(encoding="utf-8")]
    days = sorted({r["path"].split("/")[1] for r in rows})
    eras = [(1990, 1995), (1996, 2000), (2001, 2005), (2006, 2010), (2011, 2015), (2016, 2019), (2020, 2023), (2024, 2026)]
    rng = random.Random(7)
    test = set()
    for lo, hi in eras[:TEST_DAYS]:
        pool = [d for d in days if lo <= int(d[:4]) <= hi]
        if pool:
            test.add(rng.choice(pool))
    print("test days:", sorted(test), flush=True)
    reader = Reader()
    for split in ("train", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
    files = {s: (OUT / s / "labels.jsonl").open("w", encoding="utf-8") for s in ("train", "test")}
    counts = {s: 0 for s in files}
    kinds = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        gray = cv2.imread(str(PAGES / row["path"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        boxes = label_page(row, gray)
        if len(boxes) < 5:
            continue
        boxes, _hits = jump_lines(row, gray, boxes, reader)
        split = "test" if row["path"].split("/")[1] in test else "train"
        out = []
        for x0, y0, x1, y1, name in boxes:
            if x1 - x0 >= 3 and y1 - y0 >= 3:
                out.append([float(x0), float(y0), float(x1), float(y1), CLASSES.index(name)])
                kinds[name] = kinds.get(name, 0) + 1
        files[split].write(json.dumps({"image": "../../../news_pages/" + row["path"], "w": row["w"], "h": row["h"],
                                       "boxes": out}) + "\n")
        counts[split] += 1
        if i % 500 == 0:
            print(f"{i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    print(f"done in {time.time() - t0:.0f}s: {counts} | boxes {kinds}", flush=True)


if __name__ == "__main__":
    main()
