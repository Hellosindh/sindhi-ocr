"""Synthetic lines full of numbers, the reader's weak spot: years with ع / ھ, dates, list markers,
money, phone numbers, times, percentages, ranges, page numbers, and short Sindhi phrases around
them. Digits come in all three styles the labels use: 0-9, ٠-٩ (U+0660) and ۰-۹ (U+06F0).

Each line is drawn with make_synth.draw_line in a random font (the held-out test font,
SINDHI_HELD_OUT_FONT, is never used), then kept clean or given blur / noise / faded ink like synthetic_plain.

Writes assets/synthetic_numbers/lines/num_NNNNNN.png|jpg and transcript.tsv (filename<TAB>text).
Run:  python -m model.generate_numbers [--count 20000] [--sample 40]   (--sample: a few lines + a sheet only)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "synthetic_numbers"
FONT_DIR = ROOT / "Fonts"
HELD_OUT = os.environ.get("SINDHI_HELD_OUT_FONT", "")  # file name of one font kept out of all synthetic data, to test on
SKIP_FONTS = {f for f in os.environ.get("SINDHI_SKIP_FONTS", "").split(",") if f}  # decorative fonts to leave out (file names, comma-separated)

DIGITS = {"en": "0123456789", "ar": "٠١٢٣٤٥٦٧٨٩", "fa": "۰۱۲۳۴۵۶۷۸۹"}
STYLE_WEIGHTS = [("en", 0.5), ("ar", 0.3), ("fa", 0.2)]
MONTHS = ["جنوري", "فيبروري", "مارچ", "اپريل", "مئي", "جون", "جولاءِ", "آگسٽ", "سيپٽمبر", "آڪٽوبر", "نومبر", "ڊسمبر"]
WORDS = ["سال", "رپيا", "رپين", "صفحو", "صفحي", "تاريخ", "سن", "نمبر", "ڪلو", "ميل", "ڪلوميٽر", "ماڻهو", "ڏينهن",
         "مهينا", "ڪلاڪ", "منٽ", "ٻار", "ڪتاب", "ووٽ", "سيٽون", "ملڪ", "صوبا", "ضلعا", "ڳوٺ", "ايڪڙ", "فوٽ"]
TAILS = ["تي", "۾", "کان", "تائين", "جو", "جي", "آهي", "آهن", "ٿيو", "ٿي"]


def num(style: str, value: int | str) -> str:
    return str(value).translate(str.maketrans("0123456789", DIGITS[style]))


def group(value: int, rng: random.Random) -> str:
    text = f"{value:,}"
    if rng.random() < 0.3 and value >= 100000:  # South Asian grouping: 1,25,000
        s = str(value)
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        text = ",".join(([head] if head else []) + parts + [tail])
    return text


def make_text(rng: random.Random) -> str:
    style = rng.choices([s for s, _ in STYLE_WEIGHTS], [w for _, w in STYLE_WEIGHTS])[0]
    n = lambda v: num(style, v)  # noqa: E731
    year = rng.randint(1800, 2030)
    hijri = rng.randint(1250, 1450)
    day, month = rng.randint(1, 31), rng.randint(1, 12)
    era = rng.choice(["ع", "ع", " ع", "ء"])
    kind = rng.randrange(16)
    if kind == 0:
        return n(rng.choice([rng.randint(0, 99), rng.randint(100, 9999), rng.randint(10000, 9999999)]))
    if kind == 1:
        return n(year) + era + rng.choice(["", " ۾", " کان", " تائين", " تي"])
    if kind == 2:
        return n(hijri) + rng.choice(["ھ", "ھ", " ھ", "ه"]) + rng.choice(["", " ۾", " مطابق " + n(year) + "ع"])
    if kind == 3:
        sep = rng.choice(["-", "/", "."])
        return f"{n(f'{day:02d}' if rng.random() < 0.5 else day)}{sep}{n(f'{month:02d}')}{sep}{n(year)}" + rng.choice(["", "ع", " تي"])
    if kind == 4:
        return f"{n(day)} {rng.choice(MONTHS)} {n(year)}{era}" + rng.choice(["", " تي", " تي آزاد ٿيو", " جو"])
    if kind == 5:
        k = rng.randint(1, 60)
        return rng.choice([f"({n(k)})", f"{n(k)}.", f"{n(k)}-", f"{n(k)})", f"({n(k)}) " + rng.choice(WORDS)])
    if kind == 6:
        v = rng.choice([rng.randint(5, 999), rng.randint(1000, 99999), rng.randint(100000, 9999999)])
        return rng.choice([f"{n(group(v, rng))} رپيا", f"{n(group(v, rng))}/- رپيا", f"Rs. {n(group(v, rng))}/-",
                           f"رقم {n(group(v, rng))} رپين", f"{n(group(v, rng))} ڊالر"])
    if kind == 7:
        return rng.choice([f"{n('0300')}-{n(rng.randint(1000000, 9999999))}", f"{n('022')}-{n(rng.randint(2000000, 2999999))}",
                           f"فون: {n(rng.choice(['021', '022', '0300', '0333']))}-{n(rng.randint(1000000, 9999999))}"])
    if kind == 8:
        return f"{n(rng.randint(1, 12))}:{n(f'{rng.randint(0, 59):02d}')}" + rng.choice(["", " وڳي", " بجي", f":{n(f'{rng.randint(0, 59):02d}')}"])
    if kind == 9:
        pct = rng.choice([str(rng.randint(1, 100)), f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}"])
        return n(pct) + rng.choice(["%", "٪", " سيڪڙو", "% ووٽ"])
    if kind == 10:
        a = rng.randint(1900, 2020)
        return rng.choice([f"{n(a)}-{n(str(a + rng.randint(1, 9))[-2:])}", f"صفحو {n(rng.randint(1, 400))}-{n(rng.randint(401, 800))}",
                           f"{n(a)} کان {n(a + rng.randint(1, 30))}{era} تائين"])
    if kind == 11:
        start = rng.randint(1, 50)
        return ", ".join(n(start + i) for i in range(rng.randint(3, 9)))
    if kind == 12:
        return rng.choice(["- ", "", "( "]) + n(rng.randint(1, 500)) + rng.choice([" -", "", " )"])
    if kind == 13:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        op = rng.choice(["+", "-", "*", "/", "="])  # × is not in the charset
        return f"{n(a)} {op} {n(b)}" + (f" = {n(a + b)}" if op == "+" and rng.random() < 0.5 else "")
    if kind == 14:
        v = rng.choice([rng.randint(2, 99), rng.randint(100, 9999), group(rng.randint(10000, 9999999), rng)])
        return f"{n(v)} {rng.choice(WORDS)}" + rng.choice(["", " " + rng.choice(TAILS)])
    v = rng.choice([rng.randint(1, 999), year])
    return f"{rng.choice(WORDS)} {n(v)}" + (era if v == year else "") + rng.choice(["", " " + rng.choice(TAILS)])


def degrade(image, index: int, rng: random.Random):
    from PIL import Image, ImageFilter

    gray = image.convert("L")
    kind = rng.random()
    if kind < 0.45:
        return gray, "png"
    if kind < 0.7:
        radius = rng.uniform(0.4, 1.6)
        gray = gray.filter(ImageFilter.GaussianBlur(radius))
        if rng.random() < 0.4:
            small = gray.resize((max(8, gray.width // 2), max(8, gray.height // 2)))
            gray = small.resize(gray.size, Image.Resampling.BILINEAR)
    elif kind < 0.9:
        noise = Image.effect_noise(gray.size, rng.uniform(15, 45)).convert("L")
        gray = Image.blend(gray, noise, rng.uniform(0.08, 0.22))
    else:  # faded or bold ink
        gray = gray.filter(ImageFilter.MaxFilter(3) if rng.random() < 0.5 else ImageFilter.MinFilter(3))
        gray = gray.point(lambda v: int(max(0, min(255, (v - 128) * 0.6 + 150))))
    return gray, "jpg"


def render(task: tuple[int, int]) -> str | None:
    index, seed = task
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from make_synth import draw_line

    rng = random.Random(seed)
    fonts = [f.name for f in FONT_DIR.glob("*.tt[fc]") if f.name != HELD_OUT and f.name not in SKIP_FONTS]
    for _attempt in range(6):
        text = make_text(rng)
        image, missing = draw_line(text, rng.choice(fonts))
        if image is not None and not missing:
            break
    else:
        return None
    gray, ext = degrade(image, index, rng)
    from PIL import ImageStat

    if ImageStat.Stat(gray).stddev[0] < 8:  # degraded to nearly blank: keep the clean drawing instead
        gray, ext = image.convert("L"), "png"
    name = f"num_{index:06d}.{ext}"
    if ext == "png":
        gray.save(OUT / "lines" / name, optimize=True)
    else:
        gray.save(OUT / "lines" / name, format="JPEG", quality=rng.randint(30, 75))
    return f"{name}\t{text}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--sample", type=int, default=0, help="only draw this many lines and a contact sheet")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    started = time.time()
    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    count = args.sample or args.count
    tasks = [(i, 9000 + i) for i in range(count)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(render, tasks, chunksize=64) if r]
    (OUT / "transcript.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)}/{count} lines in {time.time() - started:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
