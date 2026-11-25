"""Download parts of DocLayNet v1.2 and PubTables-1M (OTSL v1.1) one parquet file at a time,
keep compact training copies, and delete each parquet file after converting it (disk is tight).

DocLayNet -> assets/layout_data/doclaynet/{train,test}/NNNNNNN.jpg + labels.jsonl
    one JSON line per page: {"image": name, "w": W, "h": H, "boxes": [[x0, y0, x1, y1, class], ...]}
    classes (0-based): 0 Caption, 1 Footnote, 2 Formula, 3 List-item, 4 Page-footer, 5 Page-header,
                       6 Picture, 7 Section-header, 8 Table, 9 Text, 10 Title
PubTables -> assets/layout_data/pubtables/{train,test}/NNNNNNN.jpg + labels.jsonl
    one JSON line per table image: {"image": name, "w": W, "h": H, "table": [x0, y0, x1, y1],
                                    "rows": R, "cols": C, "cells": [[x0, y0, x1, y1, header], ...], "otsl": [...]}

Run:  python -m layout.prepare [--doclaynet-train 12] [--doclaynet-test 1] [--pubtables-train 2] [--pubtables-test 1]
Already converted parts are skipped, so the command can be re-run to add more.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "layout_data"
RAW = ROOT / "assets" / "layout_raw"
DOCLAYNET = ("docling-project/DocLayNet-v1.2", {"train": 72, "test": 6})
PUBTABLES = ("docling-project/PubTables-1M_OTSL-v1.1", {"train": 48, "test": 9})
DOC_CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
               "Picture", "Section-header", "Table", "Text", "Title"]


def parts(split: str, total: int, count: int) -> list[str]:
    return [f"data/{split}-{i:05d}-of-{total:05d}.parquet" for i in range(min(count, total))]


def fetch(repo: str, filename: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(RAW)))


def save_gray(data: bytes, path: Path) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
        gray.save(path, format="JPEG", quality=88)
        return gray.size


def convert_doclaynet(split: str, filename: str, index_base: int) -> int:
    out = OUT / "doclaynet" / split
    out.mkdir(parents=True, exist_ok=True)
    path = fetch(DOCLAYNET[0], filename)
    rows = 0
    with (out / "labels.jsonl").open("a", encoding="utf-8") as labels:
        with pq.ParquetFile(path) as table:  # closed before the file is deleted
          for group in range(table.num_row_groups):
            for row in table.read_row_group(group, columns=["image", "bboxes", "category_id"]).to_pylist():
                name = f"{index_base + rows:07d}.jpg"
                w, h = save_gray(row["image"]["bytes"], out / name)
                boxes = [[round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1), int(c) - 1]
                         for (x, y, bw, bh), c in zip(row["bboxes"], row["category_id"])]
                labels.write(json.dumps({"image": name, "w": w, "h": h, "boxes": boxes}) + "\n")
                rows += 1
    path.unlink()
    return rows


def convert_pubtables(split: str, filename: str, index_base: int) -> int:
    out = OUT / "pubtables" / split
    out.mkdir(parents=True, exist_ok=True)
    path = fetch(PUBTABLES[0], filename)
    rows = 0
    with (out / "labels.jsonl").open("a", encoding="utf-8") as labels:
        with pq.ParquetFile(path) as table:  # closed before the file is deleted
          for group in range(table.num_row_groups):
            for row in table.read_row_group(group, columns=["image", "table_bbox", "cells", "otsl", "rows", "cols"]).to_pylist():
                name = f"{index_base + rows:07d}.jpg"
                w, h = save_gray(row["image"]["bytes"], out / name)
                cells = [[*c["bbox"][:4], int(len(c["bbox"]) > 4 and c["bbox"][4] == 2)] for c in (row["cells"] or [[]])[0]]
                labels.write(json.dumps({"image": name, "w": w, "h": h, "table": row["table_bbox"], "rows": row["rows"],
                                         "cols": row["cols"], "cells": cells, "otsl": row["otsl"]}) + "\n")
                rows += 1
    path.unlink()
    return rows


def run(name: str, repo_info, convert, split: str, count: int) -> None:
    done_file = OUT / name / split / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    total_rows = sum(1 for _ in (OUT / name / split / "labels.jsonl").open()) if (OUT / name / split / "labels.jsonl").exists() else 0
    for filename in parts(split, repo_info[1][split], count):
        if filename in done:
            continue
        started = time.time()
        rows = convert(split, filename, total_rows)
        total_rows += rows
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with done_file.open("a") as handle:
            handle.write(filename + "\n")
        print(f"{name} {split} {filename}: {rows} rows ({total_rows} total) in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doclaynet-train", type=int, default=12)
    parser.add_argument("--doclaynet-test", type=int, default=1)
    parser.add_argument("--pubtables-train", type=int, default=2)
    parser.add_argument("--pubtables-test", type=int, default=1)
    args = parser.parse_args()
    run("doclaynet", DOCLAYNET, convert_doclaynet, "test", args.doclaynet_test)
    run("pubtables", PUBTABLES, convert_pubtables, "test", args.pubtables_test)
    run("pubtables", PUBTABLES, convert_pubtables, "train", args.pubtables_train)
    run("doclaynet", DOCLAYNET, convert_doclaynet, "train", args.doclaynet_train)


if __name__ == "__main__":
    main()
