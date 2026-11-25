"""Notebook-style "handwritten" Sindhi lines in a pen-like font (pass the one closest to Sindhi handwriting),
made to look like phone photos of school notebooks (assets/rough).

Bold (thick pen), italic (strong slant) and kashida-stretched words (ـ in the drawing only) add variety.
Each word is shaped on its own and placed right to left with uneven gaps, its own baseline shift and tilt;
the whole line gets a slant and a gentle wave; strokes get a random pen width; ink is blue / black / green
on off-white ruled paper with uneven light; pieces of the lines above and below show at the edges;
then blur, low resolution and JPEG damage.

Texts: short pieces of assets/synthetic_plain/transcript.tsv lines (checked Sindhi text).
Output: assets/synthetic_notebook/notebook/NNNN/NNNNNNN.jpg + transcript.tsv (filename, font, kind, text).

Run:  python -m model.generate_notebook --font Fonts/your_handwriting_font.ttf [--count 25000] [--workers 6] [--preview 24]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FONT = Path(os.environ.get("NOTEBOOK_FONT", ROOT / "Fonts" / "handwriting.ttf"))  # set with --font
SOURCE = ROOT / "assets" / "synthetic_plain" / "transcript.tsv"
OUT = ROOT / "assets" / "synthetic_notebook"
INKS = [(140, 30, 20), (120, 40, 10), (110, 20, 40), (30, 30, 30), (15, 15, 20), (50, 110, 20), (90, 120, 20)]  # BGR

_F = {}


def _font(size):
    if size not in _F:
        import freetype
        import uharfbuzz as hb
        font = hb.Font(hb.Face(hb.Blob.from_file_path(str(FONT))))
        font.scale = (size * 64, size * 64)
        face = freetype.Face(str(FONT))
        face.set_char_size(size * 64)
        _F[size] = (font, face)
    return _F[size]


def word_mask(word, size):
    """(ink mask 0..1, baseline row) of one word, or None when a glyph is missing."""
    import freetype
    import uharfbuzz as hb
    from make_synth import bidi_runs
    font, face = _font(size)
    placed, pen = [], 0
    for piece, direction in bidi_runs(word):
        buf = hb.Buffer()
        buf.add_str(piece)
        buf.direction = direction
        buf.script = "Arab" if direction == "rtl" else "Latn"
        buf.language = "sd" if direction == "rtl" else "en"
        hb.shape(font, buf)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint == 0:
                return None
            placed.append((info.codepoint, pen + pos.x_offset, pos.y_offset))
            pen += pos.x_advance
    base = size * 2
    stamps = []
    for gid, x, y in placed:
        face.load_glyph(gid, freetype.FT_LOAD_RENDER)
        bm = face.glyph.bitmap
        if bm.width and bm.rows:
            g = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width]
            stamps.append((g, face.glyph.bitmap_left + x // 64, base - face.glyph.bitmap_top - y // 64))
    if not stamps:
        return None
    x0 = min(l for _g, l, _t in stamps); y0 = min(t for _g, _l, t in stamps)
    x1 = max(l + g.shape[1] for g, l, _t in stamps); y1 = max(t + g.shape[0] for g, _l, t in stamps)
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    for g, l, t in stamps:
        r = m[t - y0:t - y0 + g.shape[0], l - x0:l - x0 + g.shape[1]]
        np.maximum(r, g / 255.0, out=r)
    return m, base - y0


def line_mask(text, size, rng):
    """Words placed right to left with hand-made unevenness. -> (mask, baseline row) or None."""
    from model.generate_color import stretch
    words = text.split()
    parts = []
    kashida = rng.random() < 0.35  # stretched joins (ـ ــ ـــ) drawn only; the label keeps the plain word
    for w in words:
        r = word_mask(stretch(w, rng), size) if kashida else None
        r = r or word_mask(w, size)
        if r is None:
            return None
        m, b = r
        ang = rng.uniform(-3, 3)  # each word tilts a little
        if abs(ang) > 0.5:
            h, w_ = m.shape
            M = cv2.getRotationMatrix2D((w_ / 2, b), ang, 1.0)
            pad = int(h * 0.15) + 2
            m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
            M[0, 2] += pad; M[1, 2] += pad
            m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]))
            b += pad
        parts.append((m, b, rng.uniform(-0.08, 0.08) * size))  # baseline drift
    above = max(b for _m, b, _d in parts) + int(0.15 * size)
    below = max(m.shape[0] - b for m, b, _d in parts) + int(0.15 * size)
    gaps = [rng.uniform(0.12, 0.45) * size for _ in parts]
    W = int(sum(m.shape[1] for m, _b, _d in parts) + sum(gaps[:-1])) + 4
    H = above + below
    out = np.zeros((H, W), np.float32)
    x = W - 2
    for (m, b, d), g in zip(parts, gaps):  # right to left
        x -= m.shape[1]
        y = int(above - b + d)
        y = max(0, min(H - m.shape[0], y))
        r = out[y:y + m.shape[0], max(0, x):x + m.shape[1]]
        np.maximum(r, m[:, m.shape[1] - r.shape[1]:], out=r)
        x -= int(g)
    return out, above


def hand(mask, size, rng):
    """Pen width, slant and a gentle wave over the whole line."""
    k = rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6]) * size // 48  # thin pen .. bold marker
    if k > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1)))
    h, w = mask.shape
    shear = rng.uniform(-0.18, 0.12) if rng.random() < 0.7 else rng.choice([-1, 1]) * rng.uniform(0.2, 0.35)  # italic
    pad = int(abs(shear) * h) + 2
    mask = cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0)
    M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
    mask = cv2.warpAffine(mask, M, (mask.shape[1], h))
    amp, period, phase = rng.uniform(0, 0.06) * size, rng.uniform(3, 9) * size, rng.uniform(0, 6.3)
    xs = np.arange(mask.shape[1], dtype=np.float32)
    mx = np.tile(xs, (h, 1))
    my = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, mask.shape[1])) + amp * np.sin(xs / period * 6.283 + phase)
    return cv2.remap(mask, mx, my.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


def paper(h, w, rng):
    base = np.array([rng.uniform(205, 245)] * 3, np.float32) + np.array([rng.uniform(-8, 8) for _ in range(3)])
    img = np.ones((h, w, 3), np.float32) * base
    gx = np.linspace(rng.uniform(-30, 10), rng.uniform(-30, 10), w)[None, :, None]  # uneven phone light
    gy = np.linspace(rng.uniform(-15, 5), rng.uniform(-15, 5), h)[:, None, None]
    img += gx + gy
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 5), (h, w, 1))
    return img


def render(task):
    index, text, out_dir = task
    rng = random.Random(index * 7919 + 3)
    size = rng.choice([40, 48, 56, 64])
    r = line_mask(text, size, rng)
    if r is None:
        return None
    mask, base = r
    mask = hand(mask, size, rng)
    mh, mw = mask.shape
    gap = int(size * rng.uniform(1.4, 2.0))  # ruled-line spacing
    top, bottom = int(size * rng.uniform(0.05, 0.35)), int(size * rng.uniform(0.05, 0.35))
    side = int(size * rng.uniform(0.1, 0.6))
    H, W = mh + top + bottom, mw + 2 * side
    img = paper(H, W, rng)
    ink_over = np.zeros((H, W), np.float32)
    ink_over[top:top + mh, side:side + mw] = mask
    # bits of the neighbouring lines peeking in at the top / bottom edge
    for edge in ("top", "bottom"):
        if rng.random() < 0.6:
            other = SAMPLES[rng.randrange(len(SAMPLES))]
            o = line_mask(other, size, rng)
            if o is not None:
                om = hand(o[0], size, rng)
                show = int(size * rng.uniform(0.15, 0.45))
                piece = om[-show:] if edge == "top" else om[:show]
                x = rng.randint(0, max(0, W - piece.shape[1])) if piece.shape[1] < W else 0
                piece = piece[:, :W - x]
                if edge == "top":
                    r_ = ink_over[:show, x:x + piece.shape[1]]
                else:
                    r_ = ink_over[H - show:, x:x + piece.shape[1]]
                np.maximum(r_, piece[:r_.shape[0], :r_.shape[1]], out=r_)
    rule = (rng.uniform(150, 200), rng.uniform(160, 200), rng.uniform(170, 215))  # pale blue / grey rules
    y = top + base + int(size * rng.uniform(0.05, 0.25))
    for ry in range(y % gap, H, gap):
        t = rng.choice([1, 1, 2])
        img[ry:ry + t] = img[ry:ry + t] * 0.5 + np.array(rule) * 0.5
    ink = np.array(INKS[rng.randrange(len(INKS))], np.float32)
    strength = 1.0
    a = (ink_over * strength)[..., None]
    img = img * (1 - a) + ink * a
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.5:  # low-resolution phone crop
        s = rng.uniform(0.35, 0.8)
        small = cv2.resize(img, (max(8, int(W * s)), max(8, int(H * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    rel = f"notebook/{index // 10000:04d}/{index:07d}.jpg"
    path = Path(out_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 90)])
    return f"{rel}\t{FONT.name}\tnotebook\t{text}"


def load_texts():
    rng = random.Random(5)
    lines = [l.split("\t")[3].strip() for l in SOURCE.read_text(encoding="utf-8").splitlines()[1:] if l.count("\t") >= 3]
    rng.shuffle(lines)
    out = []
    for t in lines[:200000]:
        w = t.split()
        n = rng.choice([2, 3, 4, 5, 5, 6, 7])  # notebook lines are short
        if len(w) >= 2:
            s = rng.randint(0, max(0, len(w) - n))
            out.append(" ".join(w[s:s + n]))
    return out


SAMPLES = []


def init():
    global SAMPLES
    sys.path.insert(0, str(ROOT))
    SAMPLES = load_texts()[:5000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--font", default="", help="the pen-like font to draw with (default: Fonts/handwriting.ttf)")
    a = ap.parse_args()
    global FONT
    if a.font:
        os.environ["NOTEBOOK_FONT"] = a.font  # worker processes read it when they import this module
        FONT = Path(a.font)
    texts = load_texts()
    out = OUT / "preview" if a.preview else OUT
    n = a.preview or a.count
    tasks = [(i, texts[(i * 37) % len(texts)], str(out)) for i in range(n)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=init) as pool:
        for i, row in enumerate(pool.map(render, tasks, chunksize=32)):
            if row:
                rows.append(row)
            if i and i % 5000 == 0:
                print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.tsv").write_text("filename\tfont\tkind\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"done {len(rows)}/{n} lines in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
