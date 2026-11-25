"""Synthetic Sindhi lines in a second folder of fonts (Fonts_extra/), for example freely downloadable Sindhi fonts.

Uses only fonts that have every Sindhi letter, are not byte-identical to a font already in Fonts/, and
looked readable on the contact sheets (outline / dotted / ornamental / broken fonts are in SKIP).
Texts come from assets/synthetic_plain/transcript.tsv (already heh-normalised, checked Sindhi lines).
Each font gets --per-font lines: 60% clean, 20% blurred, 20% noisy (make_synth's own damage).

Output: assets/synthetic_fonts2/<kind>/NNNN/NNNNNNN.png|jpg + transcript.tsv
(filename, font, kind, text) - the same columns as synthetic_plain.

Run:  python -m model.generate_newfonts [--per-font 150] [--workers 6]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "Fonts_extra"
OUT = ROOT / "assets" / "synthetic_fonts2"
SOURCE = ROOT / "assets" / "synthetic_plain" / "transcript.tsv"
# Fonts_extra/skip.txt: 26 outline / dotted / ornamental / broken fonts (checked by eye on the
# contact sheets, 2026-11-24) a reader should not learn from
SKIP: set[str] = set()


def init() -> None:
    sys.path.insert(0, str(ROOT))
    import make_synth
    make_synth.FONT_DIR = FONTS


def render(task):
    index, font, kind, text = task
    import make_synth
    image, missing = make_synth.draw_line(text, font)
    if image is None or missing:
        return None
    ext = "png" if kind == "clean" else "jpg"
    rel = f"{kind}/{index // 10000:04d}/{index:07d}.{ext}"
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "clean":
        image.convert("L").save(path, optimize=True)
    elif kind == "blur":
        make_synth.save_blur(image, index, path)
    else:
        make_synth.save_noise(image, index, path)
    return f"{rel}\t{font}\t{kind}\t{text}"


def good_fonts() -> list[str]:
    old = {hashlib.md5(p.read_bytes()).hexdigest() for p in (ROOT / "Fonts").glob("*.tt[fc]")}
    cov = {l.split("\t")[0]: l.split("\t") for l in (FONTS / "coverage.tsv").read_text(encoding="utf-8").splitlines()[1:]}
    skip_file = FONTS / "skip.txt"
    skip = set(skip_file.read_text(encoding="utf-8").split()) if skip_file.exists() else SKIP
    return [n for n, r in sorted(cov.items()) if r[1] == "0" and n not in skip
            and hashlib.md5((FONTS / n).read_bytes()).hexdigest() not in old]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-font", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    fonts = good_fonts()
    texts = [l.split("\t")[3] for l in SOURCE.read_text(encoding="utf-8").splitlines()[1:] if l.count("\t") >= 3]
    rng = random.Random(11)
    tasks = []
    for f in fonts:
        for k in range(a.per_font):
            r = k / a.per_font
            kind = "clean" if r < 0.6 else ("blur" if r < 0.8 else "noise")
            tasks.append((len(tasks), f, kind, rng.choice(texts).strip()))
    print(f"{len(fonts)} fonts x {a.per_font} = {len(tasks)} lines", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=init) as pool:
        for i, row in enumerate(pool.map(render, tasks, chunksize=64)):
            if row:
                rows.append(row)
            if i % 5000 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    (OUT / "transcript.tsv").write_text("filename\tfont\tkind\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"done {len(rows)} lines in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
