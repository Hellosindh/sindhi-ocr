"""Download the trained weights from this repository's GitHub release into model/weights/.

    python download_weights.py            # all weights (about 200 MB)
    python download_weights.py reader     # only what ocr_page.py needs: reader + line detector
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = "Hellosindh/sindhi-ocr"
TAG = "weights-v1"
FILES = {
    "crnn_finetune_best.pt": "reader",  # line reader, fine-tuned (use this)
    "linedet.pt": "reader",             # line detector
    "crnn_best_v2.pt": "all",           # line reader, base model (start point for your own fine-tune)
    "layout.pt": "all",                 # page layout regions (books / documents)
    "layout_news.pt": "all",            # page layout regions, also trained on newspapers
    "table.pt": "all",                  # table rows / columns / header
}


def main() -> None:
    only_reader = "reader" in sys.argv[1:]
    out = Path(__file__).resolve().parent / "model" / "weights"
    out.mkdir(parents=True, exist_ok=True)
    for name, group in FILES.items():
        if only_reader and group != "reader":
            continue
        dst = out / name
        if dst.exists():
            print(f"have {name}")
            continue
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
        print(f"downloading {name} ...", flush=True)
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    print("weights in", out)


if __name__ == "__main__":
    main()
