"""Synthetic lines for what the reader still gets wrong on newspaper test lines (2026-11-24 benchmark):

  inverted   white-on-black / white-on-grey headline bars (newspaper), with the plate's edge cut off
  numbers    land-record and table text: 145/10, (00_20) 6A ايڪڙ, phone numbers, dates, prices, percentages
  urdu       real Urdu lines (training_data/urdu) in every font that has the Urdu letters
  marks      Sindhi lines full of zabar / zer / pesh / shad: words get the vowelled forms seen in real labels
  joined     two or three column pieces in one row with " | " between them (lines the detector joins)

Fonts: Fonts/ (except the held-out test font, SINDHI_HELD_OUT_FONT) + the readable fonts in Fonts_extra.
Every line gets random size, pen weight, blur, low resolution, noise and JPEG damage.
Output: assets/synthetic_gaps/<kind>/NNNN/NNNNNNN.jpg + transcript.tsv (filename, font, kind, text).

Run:  python -m model.generate_gaps [--scale 1.0] [--workers 6] [--preview 6]
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "assets" / "synthetic_gaps"
DATA = ROOT / "assets" / "training_data"
SYN = ROOT / "assets" / "synthetic_plain" / "transcript.tsv"
COUNTS = {"inverted": 15000, "numbers": 10000, "urdu": 10000, "marks": 12000, "joined": 8000}
MARKS = re.compile("[ً-ْٰ]")
URDU_LETTERS = set("ےہٹڈڑںۓ")


def urdu_fonts(names: list[str]) -> list[str]:
    """Fonts that have every Urdu-only letter (many Sindhi fonts lack ے ہ ٹ ڈ ڑ ں)."""
    from fontTools.ttLib import TTFont
    keep = []
    for n in names:
        cmap = TTFont(ROOT / "Fonts" / n, fontNumber=0, lazy=True).getBestCmap() or {}
        if all(ord(c) in cmap for c in URDU_LETTERS):
            keep.append(n)
    return keep


def fonts() -> list[str]:
    held = os.environ.get("SINDHI_HELD_OUT_FONT", "")
    own = [p.name for p in (ROOT / "Fonts").glob("*.ttf") if p.name != held]
    from model.generate_newfonts import good_fonts
    return own + ["../Fonts_extra/" + n for n in good_fonts()]


# ---------- texts ----------

def sindhi_texts(rng, n=200000):
    rows = [l.split("\t")[3].strip() for l in SYN.read_text(encoding="utf-8").splitlines()[1:n] if l.count("\t") >= 3]
    rng.shuffle(rows)
    return rows


def vowelled_forms():
    """plain word -> vowelled spellings seen in real labels."""
    forms = {}
    for l in (DATA / "sindhi" / "transcripts.tsv").read_text(encoding="utf-8").splitlines():
        if "\t" not in l:
            continue
        for w in l.split("\t", 1)[1].split():
            if MARKS.search(w) and len(w) > 1:
                forms.setdefault(MARKS.sub("", w), set()).add(w)
    return {k: sorted(v) for k, v in forms.items()}


def numbers_text(rng):
    d = lambda a, b: str(rng.randint(a, b))
    unit = ["ايڪڙ", "گهٽ", "کان", "سروي نمبر", "ديھ", "تعلقو", "رپيا", "سيڪڙو", "نمبر", "صفحو", "ڪالم", "تاريخ", "فون", "قيمت"]
    pats = [
        lambda: f"{d(1,400)}/{d(1,40)}, ({d(0,9):0>2}_{d(0,59):0>2}) {d(1,400)}/{d(1,40)}",
        lambda: f"{d(1,400)}/{d(1,12)},({d(0,9):0>2}_{d(0,40):0>2}){d(1,400)}/{d(1,9)} {rng.choice(unit)} {d(1,9)}{rng.choice('ABCD')},({d(0,9):0>2}_{d(0,40):0>2}){d(1,9)}{rng.choice('ABCD')}",
        lambda: f"{rng.choice(unit)} {d(1,999)}/{d(1,99)} {rng.choice(unit)} {d(1,99)}-{d(1,39)}",
        lambda: f"0{d(300,349)}-{d(1000000,9999999)}",
        lambda: f"{d(1,31):0>2}-{d(1,12):0>2}-{d(1990,2026)}",
        lambda: f"{rng.choice(unit)}: {d(10,99999)} {rng.choice(unit)}",
        lambda: f"{d(1,100)}% | {d(1,100)}% | {d(1,100)}%",
        lambda: f"({d(1,999)}) {rng.choice(unit)} {d(1,99)}.{d(0,99):0>2} {rng.choice(unit)}",
        lambda: " ".join(d(1, 9999) for _ in range(rng.randint(3, 7))),
    ]
    return rng.choice(pats)()


def make_text(kind, rng, ctx):
    if kind == "numbers":
        t = numbers_text(rng)
        if rng.random() < 0.4:  # numbers inside a Sindhi line
            w = rng.choice(ctx["sindhi"]).split()
            t = " ".join(w[:rng.randint(1, 4)]) + " " + t
        return t
    if kind == "urdu":
        return rng.choice(ctx["urdu"])
    if kind == "marks":
        words = rng.choice(ctx["sindhi"]).split()
        out = [rng.choice(ctx["forms"][w]) if w in ctx["forms"] and rng.random() < 0.8 else w for w in words]
        if rng.random() < 0.15 and ctx["shad"]:
            out.insert(rng.randint(0, len(out)), rng.choice(ctx["shad"]))
        return " ".join(out)
    if kind == "joined":
        parts = []
        for _ in range(rng.randint(2, 3)):
            w = rng.choice(ctx["sindhi"]).split()
            s = rng.randint(0, max(0, len(w) - 3))
            parts.append(" ".join(w[s:s + rng.randint(2, 5)]))
        return " | ".join(parts)
    w = rng.choice(ctx["sindhi"]).split()  # inverted headline: short
    return " ".join(w[:rng.randint(2, 8)])


# ---------- drawing ----------

def damage(img, rng):
    h, w = img.shape[:2]
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    if rng.random() < 0.5:
        s = rng.uniform(0.35, 0.85)
        img = cv2.resize(cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.4:
        img = np.clip(img.astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 14), img.shape), 0, 255).astype(np.uint8)
    return img


def render(task):
    index, kind, font, text, out_dir = task
    rng = random.Random(index * 104729 + 17)
    from model.generate_color import text_mask
    size = rng.choice([28, 32, 40, 48, 56])
    m = text_mask(text, font, size)
    if m is None:
        return None
    if rng.random() < (0.6 if kind == "inverted" else 0.3):  # heavier pen
        k = rng.randint(2, 3 if size < 40 else 4)
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    mh, mw = m.shape
    py, px = int(size * rng.uniform(0.15, 0.5)), int(size * rng.uniform(0.2, 0.8))
    H, W = mh + 2 * py, mw + 2 * px
    a = np.zeros((H, W), np.float32)
    a[py:py + mh, px:px + mw] = m
    if kind == "inverted":
        plate = rng.uniform(0, 70)
        ink = rng.uniform(200, 255)
        img = plate + (ink - plate) * a
        if rng.random() < 0.5:  # crop cuts into the plate: white paper shows at one or both ends
            for side in rng.sample(["l", "r"], rng.randint(1, 2)):
                cut = rng.randint(2, max(3, px))
                if side == "l":
                    img[:, :cut] = 245
                else:
                    img[:, W - cut:] = 245
        if rng.random() < 0.3:  # halftone grey plate instead of black
            img = np.where(a > 0.5, img, img + (np.indices((H, W)).sum(0) % 3 == 0) * rng.uniform(30, 90))
    else:
        paper = rng.uniform(215, 255)
        ink = rng.uniform(0, 60)
        img = paper - (paper - ink) * a
        if kind in ("joined", "numbers") and rng.random() < 0.4:  # column rules / table borders
            for side in rng.sample(["l", "r"], rng.randint(1, 2)):  # only in the margins, never through the text
                x = rng.randint(0, max(0, px - 4)) if side == "l" else rng.randint(min(W - 4, px + mw + 1), W - 3)
                img[:, x:x + rng.randint(1, 3)] = rng.uniform(0, 80)
    img = damage(np.clip(img, 0, 255).astype(np.uint8), rng)
    rel = f"{kind}/{index // 10000:04d}/{index:07d}.jpg"
    path = Path(out_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 90)])
    return f"{rel}\t{font}\t{kind}\t{text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--preview", type=int, default=0, help="this many per kind into OUT/preview")
    a = ap.parse_args()
    rng = random.Random(23)
    forms = vowelled_forms()
    ctx = {"sindhi": sindhi_texts(rng),
           "urdu": [l.split("\t", 1)[1].strip() for l in (DATA / "urdu" / "transcripts.tsv").read_text(encoding="utf-8").splitlines() if "\t" in l],
           "forms": forms,
           "shad": sorted({v for vs in forms.values() for v in vs if "ّ" in v})}
    print(f"vowelled word forms {len(forms)}, words with shad {len(ctx['shad'])}", flush=True)
    pool_fonts = fonts()
    urdu_pool = urdu_fonts(pool_fonts)
    print(f"fonts {len(pool_fonts)}, with Urdu letters {len(urdu_pool)}", flush=True)
    out = OUT / "preview" if a.preview else OUT
    tasks = []
    for kind, n in COUNTS.items():
        n = a.preview or int(n * a.scale)
        for _ in range(n):
            text = make_text(kind, rng, ctx)
            fnt = rng.choice(urdu_pool if kind == "urdu" else pool_fonts)
            tasks.append((len(tasks), kind, fnt, text, str(out)))
    t0 = time.time()
    rows = []
    # a font without a glyph returns None: retry that line in another font, up to 4 times
    with ProcessPoolExecutor(a.workers) as pool:
        todo = tasks
        for attempt in range(5):
            results = list(pool.map(render, todo, chunksize=64))
            rows += [r for r in results if r]
            todo = [(t[0], t[1], rng.choice(urdu_pool if t[1] == "urdu" else pool_fonts), t[3], t[4])
                    for t, r in zip(todo, results) if not r]
            if not todo:
                break
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcript.tsv").write_text("filename\tfont\tkind\ttext\n" + "\n".join(sorted(rows)) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"done {len(rows)}/{len(tasks)} in {time.time() - t0:.0f}s {dict(Counter(r.split(chr(9))[2] for r in rows))} -> {out}", flush=True)


if __name__ == "__main__":
    main()
