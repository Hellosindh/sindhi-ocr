"""Read a page image end to end: straighten -> find lines -> reading order -> read each line.

    python ocr_page.py page.jpg [more.jpg ...] [--weights model/weights/crnn_finetune_best.pt]
                       [--no-straighten] [--out text_folder]

Prints the text of each page (Sindhi in logical order, blocks separated by a blank line) and the time
each step took. With --out, also writes <name>.txt there.

To flatten curled / bent photos first, run an unwarping model (for example UVDoc) on the image before
calling this script; see "Unwarping" in README.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from linedet.detect import LineDetector, reading_order  # noqa: E402
from model.charset import Charset  # noqa: E402
from model.crnn import CRNN  # noqa: E402
from model.decode import greedy  # noqa: E402
from model.order import to_logical  # noqa: E402
from model.overfit import HEIGHT  # noqa: E402

LINE_PAD = 8          # white border around each line crop: keeps dots and vowel marks near the edge
MIN_LINE_CROP_H = 48  # short line bands are enlarged to this height before the final resize to HEIGHT


def load_reader(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = CRNN(ck["classes"]).eval()
    model.load_state_dict(ck["state"])
    return model.to(device)


def crops(page: np.ndarray, lines: list[dict]) -> list[np.ndarray]:
    h, w = page.shape
    out = []
    for line in lines:
        x0, y0, x1, y1 = line["box"]
        gy, gx = max(2, int(line["height"] * 0.12)), max(2, int(line["height"] * 0.3))
        y0, y1, x0, x1 = max(0, y0 - gy), min(h, y1 + gy), max(0, x0 - gx), min(w, x1 + gx)
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            out.append(np.full((HEIGHT, 32), 255, np.uint8))
            continue
        out.append(cv2.copyMakeBorder(page[y0:y1, x0:x1], LINE_PAD, LINE_PAD, LINE_PAD, LINE_PAD, cv2.BORDER_CONSTANT, value=255))
    return out


def line_tensor(crop: np.ndarray) -> torch.Tensor:
    if crop.shape[0] < MIN_LINE_CROP_H:
        s = MIN_LINE_CROP_H / max(1, crop.shape[0])
        crop = cv2.resize(crop, (max(8, round(crop.shape[1] * s)), MIN_LINE_CROP_H), interpolation=cv2.INTER_CUBIC)
    h, w = crop.shape
    nw = (max(8, round(w * HEIGHT / max(1, h))) + 3) // 4 * 4
    small = cv2.resize(crop, (nw, HEIGHT), interpolation=cv2.INTER_AREA if h > HEIGHT else cv2.INTER_CUBIC)
    return torch.from_numpy(1.0 - small.astype(np.float32) / 255.0).view(1, 1, HEIGHT, nw)


@torch.no_grad()
def read_lines(model, cs, images: list[np.ndarray], device: str) -> list[str]:
    tensors = [line_tensor(c) for c in images]
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[-1])  # similar widths batch well
    texts = [""] * len(tensors)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        W = max(tensors[i].shape[-1] for i in idx)
        x = torch.zeros(len(idx), 1, HEIGHT, W)
        for j, i in enumerate(idx):
            x[j, :, :, :tensors[i].shape[-1]] = tensors[i]
        lp = model(x.to(device)).log_softmax(-1).cpu()
        for j, i in enumerate(idx):
            t = max(1, lp.shape[1] * tensors[i].shape[-1] // W)
            texts[i] = to_logical(greedy(lp[j, :t], cs), cs)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=str(ROOT / "model" / "weights" / "crnn_finetune_best.pt"))
    ap.add_argument("--no-straighten", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cs = Charset()
    model = load_reader(Path(a.weights), device)
    detector = LineDetector()
    for f in map(Path, a.images):
        gray = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{f}: cannot open", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        if not a.no_straighten:
            from layout.orient import straighten
            gray, _info = straighten(gray)
        t1 = time.perf_counter()
        blocks = reading_order(detector.detect(gray))
        t2 = time.perf_counter()
        lines = [l for b in blocks for l in b]
        texts = read_lines(model, cs, crops(gray, lines), device)
        t3 = time.perf_counter()
        out, k = [], 0
        for b in blocks:
            out.append("\n".join(texts[k:k + len(b)]))
            k += len(b)
        text = "\n\n".join(out)
        print(f"== {f.name}: {len(lines)} lines | straighten {1000 * (t1 - t0):.0f} ms, lines {1000 * (t2 - t1):.0f} ms, "
              f"read {1000 * (t3 - t2):.0f} ms ({device})", flush=True)
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.flush()
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / (f.stem + ".txt")).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
