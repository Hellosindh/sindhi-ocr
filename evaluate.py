"""Score a reader on any folder of line images with a label file - use it before and after fine-tuning.

    python evaluate.py --lines my_test/lines --labels my_test/labels.tsv [--weights a.pt b.pt ...]

labels.tsv: filename<TAB>text (a header row "filename<TAB>text" is allowed), text in logical (typing) order.
Scoring, the same for every reader: heh forms made canonical, then
    CER       character error rate, all characters
    WER       word error rate
    base CER  letters and digits only (vowel marks, punctuation and spaces ignored)
Greedy decoding, no language model, so only the reader is measured.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.charset import Charset  # noqa: E402
from model.crnn import CRNN  # noqa: E402
from model.decode import greedy  # noqa: E402
from model.heh_normalize import normalize_heh  # noqa: E402
from model.layers import edit_distance  # noqa: E402
from model.order import to_logical  # noqa: E402
from model.overfit import to_tensor  # noqa: E402

KEEP = re.compile(r"[\w]", re.UNICODE)
MARKS = re.compile("[ً-ٰٟ]")


def base(text: str) -> str:
    return "".join(c for c in MARKS.sub("", text) if KEEP.match(c) and c != "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", nargs="+", default=[str(ROOT / "model/weights/crnn_finetune_best.pt"),
                                                      str(ROOT / "model/weights/crnn_best_v2.pt")])
    a = ap.parse_args()
    rows = []
    for l in Path(a.labels).read_text(encoding="utf-8").lstrip("﻿").splitlines():
        if "\t" in l and not l.lower().startswith("filename\t"):
            n, t = l.split("\t", 1)
            rows.append((n.strip(), unicodedata.normalize("NFC", normalize_heh(t.strip()))))
    cs = Charset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = []
    for n, _t in rows:
        with Image.open(Path(a.lines) / n) as im:
            tensors.append(to_tensor(im.convert("L")))
    for w in a.weights:
        ck = torch.load(w, map_location="cpu", weights_only=False)
        model = CRNN(ck["classes"]).eval().to(device)
        model.load_state_dict(ck["state"])
        t0 = time.time()
        ce = cn = we = wn = be = bn = 0
        with torch.no_grad():
            for (n, truth), x in zip(rows, tensors):
                read = to_logical(greedy(model(x.unsqueeze(0).to(device))[0].log_softmax(-1).cpu(), cs), cs)
                read = unicodedata.normalize("NFC", normalize_heh(read))
                ce += edit_distance(list(truth), list(read)); cn += len(truth)
                we += edit_distance(truth.split(), read.split()); wn += len(truth.split())
                be += edit_distance(list(base(truth)), list(base(read))); bn += len(base(truth))
        print(f"{Path(w).name}: {len(rows)} lines  CER {ce / max(1, cn):.2%}  WER {we / max(1, wn):.2%}  "
              f"base CER {be / max(1, bn):.2%}  ({time.time() - t0:.1f}s, {device})")


if __name__ == "__main__":
    main()
