"""Synthetic Sindhi line images: 5M clean, 1M blur, 1M noise."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import freetype
import uharfbuzz as hb
from bidi.algorithm import (
    apply_mirroring,
    explicit_embed_and_overrides,
    get_base_level,
    get_embedding_levels,
    get_empty_storage,
    resolve_implicit_levels,
    resolve_neutral_types,
    resolve_weak_types,
)
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "sindhi_corpus.txt"
FONT_DIR = ROOT / "Fonts"
MAP_PATH = ROOT / "font_glyph_map.json"
OUT = ROOT / "assets" / "synthetic_raw"
CLEAN_TARGET = 5_000_000
BLUR_COUNT = 1_000_000
NOISE_COUNT = 1_000_000
MIN_FREE = 6 * 1024**3
FONT_SIZE = 32

HB_CACHE: dict[str, hb.Font] = {}
FT_CACHE: dict[str, freetype.Face] = {}


def font_map() -> dict:
    """{"fonts": {font file: [code points it has]}} for every font in Fonts/, limited to the reader's
    characters. Built on first use from the fonts themselves and cached in font_glyph_map.json."""
    fonts = sorted(p.name for p in FONT_DIR.glob("*") if p.suffix.lower() in {".ttf", ".otf"})
    if MAP_PATH.exists():
        raw = json.loads(MAP_PATH.read_text(encoding="utf-8"))
        if sorted(raw["fonts"]) == fonts:
            return raw
    from fontTools.ttLib import TTFont
    wanted = set()
    for line in (ROOT / "character_dictionary.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) > 3 and parts[3].startswith("U+"):
            wanted.add(int(parts[3][2:], 16))
    raw = {"fonts": {}}
    for name in fonts:
        cmap = TTFont(FONT_DIR / name, fontNumber=0, lazy=True).getBestCmap() or {}
        raw["fonts"][name] = sorted(cp for cp in wanted if cp in cmap)
    MAP_PATH.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def load_font_sets() -> dict[str, set[str]]:
    raw = font_map()
    char_fonts: dict[str, set[str]] = {}
    for name, codepoints in raw["fonts"].items():
        for cp in codepoints:
            char_fonts.setdefault(chr(cp), set()).add(name)
    return char_fonts


def eligible(text: str, char_fonts: dict[str, set[str]]) -> set[str]:
    chosen: set[str] | None = None
    for ch in text:
        if ch == " ":
            continue
        fonts = char_fonts.get(ch)
        if not fonts:
            return set()
        chosen = fonts if chosen is None else chosen & fonts
        if not chosen:
            return set()
    return chosen or set()


def word_ok(word: str, char_fonts: dict[str, set[str]]) -> bool:
    if not any(ch.isalpha() or ch.isdigit() for ch in word):
        return False
    return bool(eligible(word, char_fonts))


def iter_words(char_fonts: dict[str, set[str]]):
    with CORPUS.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            for word in line.split():
                if word_ok(word, char_fonts):
                    yield word


def iter_lines(char_fonts: dict[str, set[str]], rng: random.Random):
    buffer: list[str] = []
    target = rng.randint(10, 15)
    for word in iter_words(char_fonts):
        trial = buffer + [word]
        if eligible(" ".join(trial), char_fonts):
            buffer.append(word)
            if len(buffer) >= target:
                yield " ".join(buffer)
                buffer = []
                target = rng.randint(10, 15)
            continue
        if len(buffer) >= 10:
            yield " ".join(buffer)
        buffer = [word] if eligible(word, char_fonts) else []
        target = rng.randint(10, 15)
    if len(buffer) >= 10:
        yield " ".join(buffer)


def rel_path(kind: str, index: int, ext: str) -> str:
    return f"{kind}/{index // 10000:04d}/{index:07d}.{ext}"


def abs_path(rel: str) -> Path:
    return OUT / rel


def font_faces(name: str) -> tuple[hb.Font, freetype.Face]:
    cached = HB_CACHE.get(name)
    if cached is None:
        path = str(FONT_DIR / name)
        blob = hb.Blob.from_file_path(path)
        face = hb.Face(blob)
        cached = hb.Font(face)
        cached.scale = (FONT_SIZE * 64, FONT_SIZE * 64)
        HB_CACHE[name] = cached
        ft_face = freetype.Face(path)
        ft_face.set_char_size(FONT_SIZE * 64)
        FT_CACHE[name] = ft_face
    return cached, FT_CACHE[name]


def bidi_runs(text: str) -> list[tuple[str, str]]:
    """Directional runs in left-to-right visual order, matching python-bidi."""
    storage = get_empty_storage()
    base = get_base_level(text)
    storage["base_level"] = base
    storage["base_dir"] = ("L", "R")[base]
    get_embedding_levels(text, storage, False, False)
    explicit_embed_and_overrides(storage, False)
    resolve_weak_types(storage, False)
    resolve_neutral_types(storage, False)
    resolve_implicit_levels(storage, False)
    apply_mirroring(storage, False)
    chars = storage["chars"]
    if not chars:
        return []
    logical = [
        {"index": index, "level": item["level"], "ch": item["ch"]}
        for index, item in enumerate(chars)
    ]
    highest = max(item["level"] for item in logical)
    for level in range(highest, 0, -1):
        start = None
        for index, item in enumerate(logical + [None]):
            if item is not None and item["level"] >= level:
                if start is None:
                    start = index
            elif start is not None:
                logical[start:index] = reversed(logical[start:index])
                start = None
    visual_pos = {item["index"]: pos for pos, item in enumerate(logical)}
    runs: list[tuple[int, str, str]] = []
    start = 0
    for index in range(1, len(chars) + 1):
        if index == len(chars) or (chars[index]["level"] % 2) != (chars[start]["level"] % 2):
            piece = "".join(item["ch"] for item in chars[start:index])
            direction = "rtl" if chars[start]["level"] % 2 else "ltr"
            runs.append((min(visual_pos[i] for i in range(start, index)), piece, direction))
            start = index
    runs.sort(key=lambda item: item[0])
    return [(piece, direction) for _pos, piece, direction in runs]


def visual_from_image_order(text: str, font_name: str) -> str:
    """Characters in the same left-to-right order as the drawn glyphs."""
    font, _ft_face = font_faces(font_name)
    placed = []
    pen = 0
    for piece_index, (piece, direction) in enumerate(bidi_runs(text)):
        buf = hb.Buffer()
        buf.add_str(piece)
        buf.direction = direction
        buf.script = "Arab" if direction == "rtl" else "Latn"
        buf.language = "sd" if direction == "rtl" else "en"
        hb.shape(font, buf)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            placed.append((pen + pos.x_offset, piece_index, info.cluster, piece))
            pen += pos.x_advance
    placed.sort()
    chars = []
    seen = set()
    for _x, piece_index, cluster, piece in placed:
        key = (piece_index, cluster)
        if key in seen or cluster < 0 or cluster >= len(piece):
            continue
        seen.add(key)
        chars.append(piece[cluster])
    return "".join(chars)


def draw_line(text: str, font_name: str) -> tuple[Image.Image | None, int]:
    font, ft_face = font_faces(font_name)
    notdef = 0
    placed: list[tuple[int, int, int]] = []
    pen = 0
    for piece, direction in bidi_runs(text):
        buf = hb.Buffer()
        buf.add_str(piece)
        buf.direction = direction
        buf.script = "Arab" if direction == "rtl" else "Latn"
        buf.language = "sd" if direction == "rtl" else "en"
        hb.shape(font, buf)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint == 0:
                notdef += 1
            placed.append((info.codepoint, pen + pos.x_offset, pos.y_offset))
            pen += pos.x_advance
    if notdef:
        return None, notdef
    baseline = FONT_SIZE * 2
    stamps = []
    for glyph_id, x_pos, y_pos in placed:
        ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)
        bitmap = ft_face.glyph.bitmap
        if bitmap.width == 0 or bitmap.rows == 0:
            continue
        glyph = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
        left = ft_face.glyph.bitmap_left + (x_pos // 64)
        top = baseline - ft_face.glyph.bitmap_top - (y_pos // 64)
        stamps.append((glyph, left, top))
    if not stamps:
        return None, notdef
    pad = 4
    min_x = min(left for _glyph, left, _top in stamps)
    min_y = min(top for _glyph, _left, top in stamps)
    max_x = max(left + glyph.width for glyph, left, _top in stamps)
    max_y = max(top + glyph.height for glyph, _left, top in stamps)
    image = Image.new("L", (max_x - min_x + pad * 2, max_y - min_y + pad * 2), 255)
    for glyph, left, top in stamps:
        image.paste(0, (left - min_x + pad, top - min_y + pad), glyph)
    return image.convert("1", dither=Image.Dither.NONE), 0


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def save_blur(image: Image.Image, index: int, path: Path) -> None:
    radius = 0.7 + (index % 8) * 0.15
    gray = image.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))
    path.parent.mkdir(parents=True, exist_ok=True)
    gray.save(path, format="JPEG", quality=35, optimize=True)


def save_noise(image: Image.Image, index: int, path: Path) -> None:
    gray = image.convert("L")
    noise = Image.effect_noise(gray.size, 22 + (index % 5) * 4).convert("L")
    mixed = Image.blend(gray, noise, 0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    mixed.save(path, format="JPEG", quality=40, optimize=True)


def safe_text(text: str) -> str:
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def render_batch(tasks: list[tuple]) -> list[str]:
    rows: list[str] = []
    for task in tasks:
        rows.extend(render_task(task))
    return rows


def render_task(task: tuple) -> list[str]:
    index, font_name, text, do_blur, do_noise = task
    rows = []
    try:
        image, notdef = draw_line(text, font_name)
        if image is None or notdef:
            return [f"ERROR\t{index}\t{font_name}\tmissing glyph"]
        clean_rel = rel_path("clean", index, "png")
        save_png(image, abs_path(clean_rel))
        rows.append(f"{clean_rel}\t{font_name}\tclean\t{safe_text(text)}")
        if do_blur:
            blur_rel = rel_path("blur", index, "jpg")
            save_blur(image, index, abs_path(blur_rel))
            rows.append(f"{blur_rel}\t{font_name}\tblur\t{safe_text(text)}")
        if do_noise:
            noise_rel = rel_path("noise", index, "jpg")
            save_noise(image, index, abs_path(noise_rel))
            rows.append(f"{noise_rel}\t{font_name}\tnoise\t{safe_text(text)}")
    except Exception as exc:
        rows.append(f"ERROR\t{index}\t{font_name}\t{exc}")
    return rows


def clean_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "\tclean\t" in line and not line.startswith("ERROR"):
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=CLEAN_TARGET)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    limit = min(args.limit, CLEAN_TARGET)
    sample = OUT / "_sample"
    if sample.exists():
        shutil.rmtree(sample)
    OUT.mkdir(parents=True, exist_ok=True)
    char_fonts = load_font_sets()
    rng = random.Random(7)
    transcript = OUT / "transcript.tsv"
    log_path = OUT / "progress.log"
    workers = 1 if limit < 100 else min(6, os.cpu_count() or 1)
    started = time.perf_counter()

    start = clean_rows(transcript) if args.resume else 0
    if start >= limit:
        print(f"already_done images={start}", flush=True)
        return

    def tasks():
        made = 0
        for text in iter_lines(char_fonts, rng):
            fonts = sorted(eligible(text, char_fonts))
            if not fonts:
                continue
            for font_name in fonts:
                if made >= limit:
                    return
                if made < start:
                    made += 1
                    continue
                do_blur = made < BLUR_COUNT
                do_noise = BLUR_COUNT <= made < BLUR_COUNT + NOISE_COUNT
                yield (made, font_name, text, do_blur, do_noise)
                made += 1

    def batches():
        batch = []
        for task in tasks():
            batch.append(task)
            if len(batch) >= 64:
                yield batch
                batch = []
        if batch:
            yield batch

    written = start
    reported = start
    stop = False
    mode = "a" if args.resume and transcript.exists() else "w"
    with transcript.open(mode, encoding="utf-8", newline="\n") as handle, log_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as log:
        if mode == "w":
            handle.write("filename\tfont\tkind\ttext\n")
        print(f"resume_from={start} target={limit}", flush=True)

        def consume(rows: list[str]) -> None:
            nonlocal written, reported, stop
            for row in rows:
                handle.write(row + "\n")
                if "\tclean\t" in row:
                    written += 1
            if written - reported >= 2000:
                reported = written
                elapsed = time.perf_counter() - started
                free = shutil.disk_usage(ROOT).free
                message = f"images={written} per_sec={written / elapsed:.1f} free_gb={free / 1024**3:.1f}"
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
                handle.flush()
                if free < MIN_FREE:
                    print("stopped_low_disk", flush=True)
                    log.write("stopped_low_disk\n")
                    stop = True

        if workers == 1:
            batch = []
            for task in tasks():
                batch.append(task)
                if len(batch) >= 64:
                    consume(render_batch(batch))
                    batch = []
                    if stop:
                        break
            if batch and not stop:
                consume(render_batch(batch))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                inflight = []
                for batch in batches():
                    inflight.append(pool.submit(render_batch, batch))
                    if len(inflight) >= workers * 2:
                        consume(inflight.pop(0).result())
                        if stop:
                            break
                for future in inflight:
                    if stop:
                        break
                    consume(future.result())
    elapsed = time.perf_counter() - started
    print(f"done images={written} seconds={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
