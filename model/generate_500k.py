"""5M lines into synthetic_plain: 3M clean_new, 1M blur_new, 1M noise_new."""

from __future__ import annotations

import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import freetype
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "synthetic_plain"
FONT_DIR = ROOT / "Fonts"
CORPUS = ROOT / "sindhi_corpus.txt"
CLEAN = 3_000_000
BLUR = 1_000_000
NOISE = 1_000_000
KIND_CLEAN = "clean_new"
KIND_BLUR = "blur_new"
KIND_NOISE = "noise_new"
WORKERS = 6
HELD_OUT = os.environ.get("SINDHI_HELD_OUT_FONT", "")  # file name of one font kept out of all synthetic data, to test on


def rel_path(kind: str, index: int, ext: str) -> str:
    return f"{kind}/{index // 10000:04d}/{index:07d}.{ext}"


def safe_text(text: str) -> str:
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _root() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def render_clean(task: tuple) -> str | None:
    _root()
    from make_synth import draw_line

    index, font_name, text = task
    image, missing = draw_line(text, font_name)
    if image is None or missing:
        return None
    path = OUT / rel_path(KIND_CLEAN, index, "png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("L").save(path, optimize=True)
    return f"{rel_path(KIND_CLEAN, index, 'png')}\t{font_name}\tclean\t{safe_text(text)}"


def render_blur(task: tuple) -> str | None:
    index, font_name, text = task
    source = OUT / rel_path(KIND_CLEAN, index, "png")
    if not source.exists():
        return None
    gray = Image.open(source).convert("L")
    kind = index % 5
    if kind == 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=0.4 + (index % 7) * 0.15))
    elif kind == 1:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=1.2 + (index % 4) * 0.4))
    elif kind == 2:
        gray = gray.filter(ImageFilter.BoxBlur(radius=1 + index % 3))
    elif kind == 3:
        gray = gray.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
    else:
        small = gray.resize((max(8, gray.width // 2), max(8, gray.height // 2)))
        gray = small.resize(gray.size, Image.Resampling.BILINEAR)
    path = OUT / rel_path(KIND_BLUR, index, "jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    gray.save(path, format="JPEG", quality=28 + (index % 6) * 8)
    gray.close()
    return f"{rel_path(KIND_BLUR, index, 'jpg')}\t{font_name}\tblur\t{safe_text(text)}"


def render_noise(task: tuple) -> str | None:
    index, font_name, text = task
    source_index = CLEAN // 3 + index
    source = OUT / rel_path(KIND_CLEAN, source_index, "png")
    if not source.exists():
        return None
    gray = Image.open(source).convert("L")
    kind = index % 5
    if kind == 0:
        noise = Image.effect_noise(gray.size, 18 + (index % 6) * 6).convert("L")
        gray = Image.blend(gray, noise, 0.08 + (index % 5) * 0.04)
    elif kind == 1:
        noise = Image.effect_noise(gray.size, 40 + (index % 4) * 10).convert("L")
        gray = Image.blend(gray, noise, 0.22)
    elif kind == 2:
        pixels = gray.point(lambda value: int(max(0, min(255, (value - 128) * 0.55 + 150))))
        gray = pixels
    elif kind == 3:
        gray = gray.filter(ImageFilter.MedianFilter(3))
        noise = Image.effect_noise(gray.size, 12).convert("L")
        gray = Image.blend(gray, noise, 0.1)
    else:
        gray = gray.filter(ImageFilter.MaxFilter(3))
        noise = Image.effect_noise(gray.size, 28).convert("L")
        gray = Image.blend(gray, noise, 0.18)
    path = OUT / rel_path(KIND_NOISE, index, "jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    gray.save(path, format="JPEG", quality=32 + (index % 5) * 7)
    gray.close()
    return f"{rel_path(KIND_NOISE, index, 'jpg')}\t{font_name}\tnoise\t{safe_text(text)}"


def font_supports(font_name: str, text: str, faces: dict[str, freetype.Face]) -> bool:
    face = faces.get(font_name)
    if face is None:
        face = freetype.Face(str(FONT_DIR / font_name))
        faces[font_name] = face
    return all(face.get_char_index(ord(char)) != 0 for char in text)


COLLECT_CACHE = OUT / "collected_lines.tsv"


def collect_lines() -> list[tuple[str, str]]:
    """Build (text, font) pairs. Text is raw from corpus; heh normalize is a later pass."""
    from make_synth import eligible, load_font_sets, word_ok

    char_fonts = load_font_sets()
    known = {name for names in char_fonts.values() for name in names}
    pool = sorted(name for name in known if name != HELD_OUT)
    faces: dict[str, freetype.Face] = {}
    chosen: list[tuple[str, str]] = []
    words: list[str] = []
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            for word in line.split():
                if word_ok(word, char_fonts):
                    words.append(word)
    if not words:
        raise SystemExit("no corpus words")
    index = 0
    rng = random.Random(7)
    COLLECT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with COLLECT_CACHE.open("w", encoding="utf-8") as cache:
        cache.write("text\tfont\n")
        while len(chosen) < CLEAN:
            pieces: list[str] = []
            while True:
                word = words[index % len(words)]
                index += 1
                trial = " ".join(pieces + [word])
                if pieces and (len(trial) > 40 or (len(" ".join(pieces)) >= 18 and rng.random() < 0.3)):
                    break
                pieces.append(word)
                if index > len(words) * 50 and not pieces:
                    break
            if not pieces:
                continue
            text = " ".join(pieces)
            options = []
            for font_name in pool:
                if font_name in known and font_name in eligible(text, char_fonts):
                    options.append(font_name)
                elif font_name not in known and font_supports(font_name, text, faces):
                    options.append(font_name)
            if options:
                font = options[len(chosen) % len(options)]
                chosen.append((text, font))
                cache.write(f"{safe_text(text)}\t{font}\n")
            if len(chosen) % 100_000 == 0 and len(chosen):
                cache.flush()
                print(f"collect {len(chosen)}/{CLEAN}", flush=True)
    return chosen


def load_collected() -> list[tuple[str, str]] | None:
    if not COLLECT_CACHE.exists():
        return None
    rows: list[tuple[str, str]] = []
    with COLLECT_CACHE.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            text, font = line.rstrip("\n").split("\t", 1)
            rows.append((text, font))
    if len(rows) < CLEAN:
        print(f"cache incomplete {len(rows)}/{CLEAN} — will recollect", flush=True)
        return None
    print(f"loaded_cache={len(rows)} from {COLLECT_CACHE}", flush=True)
    return rows[:CLEAN]


def apply_heh(lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Replace-rule pass on already collected text (no recollect)."""
    from model.heh_normalize import normalize_heh

    out = [(normalize_heh(text), font) for text, font in lines]
    print(f"heh_normalized={len(out)}", flush=True)
    return out


def run_pool(fn, tasks: list[tuple], transcript) -> int:
    written = 0
    chunk = 200
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(tasks), chunk):
            rows = [row for row in pool.map(fn, tasks[start : start + chunk]) if row]
            if rows:
                transcript.write("\n".join(rows) + "\n")
                transcript.flush()
            written += len(rows)
            print(f"{fn.__name__} {start + len(tasks[start:start+chunk])}/{len(tasks)} kept={written}", flush=True)
    return written


def main() -> None:
    os.chdir(ROOT)
    import sys

    sys.path.insert(0, str(ROOT))
    free = shutil.disk_usage(ROOT).free
    if free < 15 * 1024**3:
        raise SystemExit(f"only {free / 1024**3:.1f} GB free; need at least ~15 GB to start")
    print(f"out={OUT} free_gb={free / 1024**3:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / KIND_CLEAN).mkdir(parents=True, exist_ok=True)
    (OUT / KIND_BLUR).mkdir(parents=True, exist_ok=True)
    (OUT / KIND_NOISE).mkdir(parents=True, exist_ok=True)
    lines = load_collected()
    if lines is None:
        print("collecting lines", flush=True)
        lines = collect_lines()
    print(f"texts={len(lines)}", flush=True)
    if len(lines) < CLEAN:
        raise SystemExit(f"only collected {len(lines)} lines")
    lines = apply_heh(lines)
    transcript_path = OUT / "transcript.tsv"
    with transcript_path.open("w", encoding="utf-8") as transcript:
        transcript.write("filename\tfont\tkind\ttext\n")
        clean_tasks = [(index, font, text) for index, (text, font) in enumerate(lines[:CLEAN])]
        run_pool(render_clean, clean_tasks, transcript)
        blur_tasks = clean_tasks[:BLUR]
        run_pool(render_blur, blur_tasks, transcript)
        noise_tasks = [
            (index, lines[CLEAN // 3 + index][1], lines[CLEAN // 3 + index][0]) for index in range(NOISE)
        ]
        run_pool(render_noise, noise_tasks, transcript)
    print("done", flush=True)


if __name__ == "__main__":
    main()
