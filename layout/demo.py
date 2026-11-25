"""Run the layout model (and the table model on each Table box) on real pages; draw the result.

Run:  python -m layout.demo PAGE_OR_FOLDER [...] [--out layout/demo_out]
Each page -> OUT/<name>.jpg with coloured region boxes; tables get their row / column lines.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layout.model import CLASSES, SIZE, STRIDE, LayoutNet, decode, net_input, split_lines, tidy
from layout.contents import confirm_contents
from layout.orient import straighten
from layout.tables import TableNet, structure

COLORS = [(200, 120, 0), (120, 120, 120), (0, 150, 150), (0, 160, 0), (150, 0, 150), (150, 0, 150),
          (0, 120, 255), (0, 0, 220), (255, 0, 0), (60, 160, 60), (0, 0, 160), (0, 200, 255),
          (200, 200, 0), (200, 0, 200)]  # Masthead, Jump-line


def square(image: Image.Image):
    scale = SIZE / max(image.size)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("L", (SIZE, SIZE), 255)
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), (0, 0))
    return torch.from_numpy(np.asarray(canvas, np.uint8).copy())[None, None], scale, w, h


def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state"])
    return model.to(device).eval()


@torch.no_grad()
def page_layout(model, table_model, gray: Image.Image, device):
    x, scale, _w, _h = square(gray)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        out = model(net_input(x.to(device)))
    boxes = [(x0 / scale, y0 / scale, x1 / scale, y1 / scale, c, s) for x0, y0, x1, y1, c, s in tidy(split_lines(decode(out)[0], x[0, 0].numpy()))
             if CLASSES[c] != "Jump-line"]  # not wanted
    boxes = confirm_contents(np.asarray(gray), boxes)
    tables = []
    for b in list(boxes):
        if CLASSES[b[4]] != "Table":
            continue
        x0, y0 = max(0, int(b[0]) - 8), max(0, int(b[1]) - 8)
        x1, y1 = min(gray.width, int(b[2]) + 8), min(gray.height, int(b[3]) + 8)
        crop = gray.crop((x0, y0, x1, y1))
        t, s, w, h = square(crop)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            prob = torch.sigmoid(table_model(net_input(t.to(device))).float())[0].cpu().numpy()
        rows, cols, _hdr = structure(prob, int(np.ceil(w / STRIDE)), int(np.ceil(h / STRIDE)))
        if not rows:  # one row is not a table (a running footer with a page number between rules)
            boxes = [o for o in boxes if o is not b]
            continue
        tables.append(((x0, y0, x1, y1), [y0 + r / s for r in rows], [x0 + c / s for c in cols]))
    return boxes, tables


def draw(bgr, boxes, tables):
    img = bgr.copy()
    lw = max(2, img.shape[1] // 500)
    for x0, y0, x1, y1, c, s in boxes:
        col = COLORS[c]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, lw)
        cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (int(x0), max(12, int(y0) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * lw, col, max(1, lw // 2))
    for (x0, y0, x1, y1), rows, cols in tables:
        for r in rows:
            cv2.line(img, (x0, int(r)), (x1, int(r)), (0, 0, 255), lw)
        for c in cols:
            cv2.line(img, (int(c), y0), (int(c), y1), (255, 0, 255), lw)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default=str(ROOT / "layout" / "demo_out"))
    parser.add_argument("--layout", default=str(ROOT / "model" / "weights" / "layout.pt"))
    parser.add_argument("--table", default=str(ROOT / "model" / "weights" / "table.pt"))
    parser.add_argument("--no-straighten", action="store_true", help="skip tilt / sideways / upside-down fix")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(LayoutNet(), args.layout, device)
    table_model = load(TableNet(), args.table, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for p in map(Path, args.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}) if p.is_dir() else [p]
    for f in files:
        bgr = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        turn = ""
        t0 = time.perf_counter()
        if not args.no_straighten:
            g, info = straighten(g)
            if info["turns"] or info["flipped"] or abs(info["degrees"]) >= 0.3:
                turn = f" | turned {(info['turns'] * 90 + (180 if info['flipped'] else 0)) % 360} deg, tilt {info['degrees']:+.1f}"
                bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        straight_ms = (time.perf_counter() - t0) * 1000
        gray = Image.fromarray(g)
        t = time.perf_counter()
        boxes, tables = page_layout(model, table_model, gray, device)
        ms = (time.perf_counter() - t) * 1000
        kinds = {}
        for b in boxes:
            kinds[CLASSES[b[4]]] = kinds.get(CLASSES[b[4]], 0) + 1
        grid = ", ".join(f"{len(r) + 1}x{len(c) + 1}" for _b, r, c in tables)
        print(f"{f.name}: straighten {straight_ms:.0f} ms + layout {ms:.0f} ms{turn} | {kinds} | tables {grid or '-'}", flush=True)
        cv2.imwrite(str(out / (f.stem + ".jpg")), draw(bgr, boxes, tables), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
