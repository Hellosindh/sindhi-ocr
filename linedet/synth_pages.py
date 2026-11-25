"""Synthetic Sindhi pages with exact line boxes, for training the line detector.

Every page is drawn from corpus words with real Sindhi fonts: books (running header,
footnotes, page number), newspapers (masthead, 2-6 columns, column rules, headlines,
pictures) and magazines (pictures, shaded quote boxes). Each text line is stored as a
4-point box with a type:

    0 body   1 heading   2 header   3 footer   4 footnote

Run:  python -m linedet.synth_pages --count 5000
Out:  assets/linedet_pages/NNNNN.jpg + NNNNN.json  ({"lines": [[quad, type], ...]})
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import freetype
import numpy as np
import uharfbuzz as hb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FONT_DIR = ROOT / "Fonts"
CORPUS = ROOT / "sindhi_corpus.txt"
MAP_PATH = ROOT / "font_glyph_map.json"
OUT = ROOT / "assets" / "linedet_pages"

BODY, HEADING, HEADER, FOOTER, FOOTNOTE = range(5)
TYPE_NAMES = ["body", "heading", "header", "footer", "footnote"]
DIGIT_SETS = ["0123456789", "۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩"]

# ---------------------------------------------------------------- fonts and words

_CHAR_FONTS: dict[str, set[str]] | None = None
_FT: dict[str, freetype.Face] = {}
_HB: dict[tuple[str, int], hb.Font] = {}
_HB_FACE: dict[str, hb.Face] = {}


def char_fonts() -> dict[str, set[str]]:
    global _CHAR_FONTS
    if _CHAR_FONTS is None:
        from make_synth import font_map
        raw = font_map()
        _CHAR_FONTS = {}
        for name, codepoints in raw["fonts"].items():
            for cp in codepoints:
                _CHAR_FONTS.setdefault(chr(cp), set()).add(name)
    return _CHAR_FONTS


def font_has(font: str, text: str) -> bool:
    table = char_fonts()
    return all(ch == " " or font in table.get(ch, ()) for ch in text)


def usable_fonts() -> list[str]:
    sindhi = "ابٻڀتٿٽثپجڄڃچڇحخدڌڏڊڍذرڙزسشصضطظعغفڦقڪکگڳڱلمنڻوھءيآ"
    names = sorted(p.name for p in FONT_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"})
    from make_synth import font_map
    known = font_map()["fonts"]
    return [name for name in names if name in known and font_has(name, sindhi)]


def bold_fonts() -> list[str]:
    """Fonts whose file name says bold / black / heavy: used for headings and banners."""
    return [n for n in usable_fonts() if any(w in n.lower() for w in ("bold", "black", "heavy"))]


def _faces(font: str, size: int) -> tuple[hb.Font, freetype.Face]:
    path = str(FONT_DIR / font)
    if font not in _FT:
        _FT[font] = freetype.Face(path)
        _HB_FACE[font] = hb.Face(hb.Blob.from_file_path(path))
    key = (font, size)
    if key not in _HB:
        hb_font = hb.Font(_HB_FACE[font])
        hb_font.scale = (size * 64, size * 64)
        _HB[key] = hb_font
    face = _FT[font]
    face.set_char_size(size * 64)
    return _HB[key], face


def render_word(word: str, font: str, size: int) -> tuple[np.ndarray, int] | None:
    """Ink coverage (3*size tall, baseline at 2*size) and pen advance for one word."""
    from make_synth import bidi_runs

    hb_font, face = _faces(font, size)
    stamps = []
    pen = 0
    for piece, direction in bidi_runs(word):
        buf = hb.Buffer()
        buf.add_str(piece)
        buf.direction = direction
        buf.script = "Arab" if direction == "rtl" else "Latn"
        buf.language = "sd" if direction == "rtl" else "en"
        hb.shape(hb_font, buf)
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            if info.codepoint == 0:
                return None
            face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            bitmap = face.glyph.bitmap
            if bitmap.width and bitmap.rows:
                glyph = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.pitch)[:, : bitmap.width]
                stamps.append((glyph, face.glyph.bitmap_left + (pen + pos.x_offset) // 64, face.glyph.bitmap_top + pos.y_offset // 64))
            pen += pos.x_advance
    advance = max(1, pen // 64)
    if not stamps:
        return None
    left = min(0, min(x for _g, x, _t in stamps))
    right = max(advance, max(x + g.shape[1] for g, x, _t in stamps))
    height = size * 3
    baseline = size * 2
    canvas = np.zeros((height, right - left), dtype=np.uint8)
    for glyph, x, top in stamps:
        y = baseline - top
        y0, y1 = max(0, y), min(height, y + glyph.shape[0])
        if y1 <= y0:
            continue
        x0 = x - left
        part = glyph[y0 - y : y1 - y]
        region = canvas[y0:y1, x0 : x0 + part.shape[1]]
        np.maximum(region, part[:, : region.shape[1]], out=region)
    return canvas, advance - left


class WordSource:
    """Random runs of corpus words, read from random places in the 650 MB corpus."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.size = CORPUS.stat().st_size
        self.words: list[str] = []
        self.pos = 0

    def _refill(self) -> None:
        with CORPUS.open("rb") as handle:
            handle.seek(self.rng.randrange(0, max(1, self.size - 300_000)))
            chunk = handle.read(200_000).decode("utf-8", errors="ignore")
        words = [w for w in chunk.split() if any(ch.isalpha() for ch in w) and len(w) <= 18]
        self.words = words[1:-1] or ["سنڌ"]
        self.pos = 0

    def next(self) -> str:
        if self.pos >= len(self.words):
            self._refill()
        word = self.words[self.pos]
        self.pos += 1
        return word


# ---------------------------------------------------------------- page canvas


class Page:
    def __init__(self, width: int, height: int, rng: random.Random):
        self.w, self.h = width, height
        self.rng = rng
        self.ink = np.zeros((height, width), dtype=np.float32)  # darkness 0..1 drawn so far
        self.lines: list[tuple[list[list[float]], int]] = []
        self.cache: dict[tuple[str, str, int], tuple[np.ndarray, int] | None] = {}

    def word(self, word: str, font: str, size: int):
        key = (word, font, size)
        if key not in self.cache:
            self.cache[key] = render_word(word, font, size) if font_has(font, word) else None
        return self.cache[key]

    def paint(self, cov: np.ndarray, x: int, y: int, strength: float = 1.0, white: bool = False) -> None:
        h, w = cov.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        part = cov[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32) / 255.0 * strength
        region = self.ink[y0:y1, x0:x1]
        if white:
            region *= 1.0 - part
        else:
            np.maximum(region, part, out=region)

    def fill(self, x0: int, y0: int, x1: int, y1: int, value: float) -> None:
        self.ink[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = np.maximum(
            self.ink[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)], value
        )

    def rule(self, x0: int, y0: int, x1: int, y1: int, thick: int) -> None:
        cv2.line(self.ink, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, max(1, thick))


def make_line(page: Page, words: list[str], font: str, size: int, width: int, justify: bool, gaps: dict | None = None):
    """Words right to left into one line; returns coverage and line ink box (x0,y0,x1,y1) or None.
    `gaps` {word index: px} adds a wide blank after that word (list numbers, contents leaders)."""
    gaps = gaps or {}
    pieces = [(page.word(w, font, size), gaps.get(i, 0)) for i, w in enumerate(words)]
    pieces = [(p, g) for p, g in pieces if p is not None]
    if not pieces:
        return None
    space = max(2, int(size * page.rng.uniform(0.22, 0.38)))
    extra = sum(g for _p, g in pieces[:-1])
    total = sum(p[1] for p, _g in pieces) + space * (len(pieces) - 1) + extra
    if justify and len(pieces) > 1 and total < width:
        space += (width - total) / (len(pieces) - 1)
    canvas = np.zeros((size * 3, int(max(width, total)) + size * 4), dtype=np.uint8)
    pen = canvas.shape[1] - size * 2
    for (cov, advance), gap in pieces:
        x = max(0, int(round(pen - advance)))
        part = cov[:, : canvas.shape[1] - x]
        np.maximum(canvas[:, x : x + part.shape[1]], part, out=canvas[:, x : x + part.shape[1]])
        pen = x - space - gap
    rows = np.where(canvas.max(axis=1) > 60)[0]
    cols = np.where(canvas.max(axis=0) > 60)[0]
    if rows.size == 0:
        return None
    return canvas, (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def restyle(line, canvas: np.ndarray):
    """Same line with new coverage (outlined, shadowed...): ink box measured again."""
    rows = np.where(canvas.max(axis=1) > 60)[0]
    cols = np.where(canvas.max(axis=0) > 60)[0]
    if rows.size == 0:
        return line
    return canvas, (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def put_line(page: Page, line, right: int, y: int, kind: int, align: str, box_w: int, white=False, strength=1.0) -> int:
    """Place a made line with its ink right edge at `right` (or centred); return its ink bottom."""
    canvas, (bx0, by0, bx1, by1) = line
    ink_w = bx1 - bx0
    if align == "center":
        x_right = right - (box_w - ink_w) // 2
    elif align == "left":
        x_right = right - box_w + ink_w
    else:
        x_right = right
    ox = x_right - bx1
    oy = y - by0
    page.paint(canvas, ox, oy, strength=strength, white=white)
    quad = [[ox + bx0, oy + by0], [ox + bx1, oy + by0], [ox + bx1, oy + by1], [ox + bx0, oy + by1]]
    if 0 <= ox + bx0 and ox + bx1 <= page.w and 0 <= oy + by0 and oy + by1 <= page.h:
        page.lines.append((quad, kind))
    return oy + by1


def fill_words(page: Page, src: WordSource, font: str, size: int, width: int, count_range=(1, 999)):
    """Greedy words until the line is full; yields word lists for one paragraph."""
    words: list[str] = []
    used = 0
    space = size * 0.3
    while True:
        word = src.next()
        piece = page.word(word, font, size)
        if piece is None:
            continue
        if words and used + space + piece[1] > width:
            yield words
            words, used = [], 0
        words.append(word)
        used += piece[1] + (space if len(words) > 1 else 0)


def paragraph(page: Page, src, font, size, x0, x1, y, y_max, lead, kind=BODY, justify=True, max_lines=None, align="right"):
    """Flow a paragraph into column [x0,x1] from y; returns next y (stops at y_max)."""
    rng = page.rng
    width = x1 - x0
    n_lines = max_lines or rng.randint(2, 12)
    indent = int(size * rng.choice([0, 0, 1.5, 2])) if kind == BODY else 0
    gen = fill_words(page, src, font, size, width - indent)
    for index in range(n_lines):
        if y + lead > y_max:
            return y, False
        words = next(gen)
        last = index == n_lines - 1
        if last and len(words) > 2 and rng.random() < 0.8:
            words = words[: rng.randint(1, len(words) - 1)]
        w = width - (indent if index == 0 else 0)
        line = make_line(page, words, font, size, w, justify and (not last or rng.random() < 0.06))
        if line is None:
            continue
        right = x1 - (indent if index == 0 else 0)
        put_line(page, line, right, y + int(size * 0.35), kind, align, w)
        y += lead
    return y, True


def picture(page: Page, x0, y0, x1, y1) -> None:
    """Photo-like gray patch: blurred noise at a few scales, sometimes halftone, sometimes framed."""
    rng = page.rng
    w, h = max(4, x1 - x0), max(4, y1 - y0)
    img = np.zeros((h, w), np.float32)
    for scale in (4, 16, 48):
        small = np.random.default_rng(rng.randrange(1 << 30)).random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(0.3, 1.0)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    for _ in range(rng.randint(0, 5)):
        cx, cy = rng.randrange(w), rng.randrange(h)
        cv2.ellipse(img, (cx, cy), (rng.randint(5, w // 2 + 6), rng.randint(5, h // 2 + 6)), rng.randrange(180), 0, 360, rng.random(), -1)
    img = img * rng.uniform(0.5, 1.0) + rng.uniform(0.0, 0.3)
    if rng.random() < 0.3:
        step = rng.randint(3, 6)
        yy, xx = np.mgrid[0:h, 0:w]
        dots = (((xx % step) - step / 2) ** 2 + ((yy % step) - step / 2) ** 2) < (img * step * 0.7) ** 2
        img = dots.astype(np.float32)
    page.fill(x0, y0, x1, y1, 0)
    region = page.ink[y0 : y0 + h, x0 : x0 + w]
    np.maximum(region, np.clip(img, 0, 1)[: region.shape[0], : region.shape[1]], out=region)
    if rng.random() < 0.4:
        cv2.rectangle(page.ink, (x0, y0), (x1, y1), 1.0, rng.randint(1, 3))


def page_no(rng: random.Random) -> int:
    """Printed page numbers: often one or two digits (front matter, short books, magazines)."""
    r = rng.random()
    return rng.randint(1, 9) if r < 0.4 else rng.randint(10, 99) if r < 0.7 else rng.randint(100, 480)


def digits(n: int, rng: random.Random, font: str) -> str:
    for table in rng.sample(DIGIT_SETS, 3):
        text = "".join(table[int(d)] for d in str(n))
        if font_has(font, text):
            return text
    return str(n)


# ---------------------------------------------------------------- page kinds


def running_header(page: Page, src, font, size, x0, x1, y, rule: bool) -> int:
    rng = page.rng
    lead = int(size * 1.9)
    words = [src.next() for _ in range(rng.randint(1, 5))]
    if rng.random() < 0.15:  # header that is only the page number
        words = [digits(page_no(rng), rng, font)]
    title = make_line(page, words, font, size, x1 - x0, False)
    bottom = y
    if title is not None:
        align = rng.choice(["center", "right", "left"])
        bottom = put_line(page, title, x1, y, HEADER, align, x1 - x0)
        if align != "center" and rng.random() < 0.8:
            num = make_line(page, [digits(page_no(rng), rng, font)], font, size, size * 4, False)
            if num is not None:
                put_line(page, num, x1, y, HEADER, "left" if align == "right" else "right", x1 - x0)
    y = max(bottom, y + lead - size) + int(size * 0.4)
    if rule:
        page.rule(x0, y, x1, y, rng.randint(1, 3))
        if rng.random() < 0.3:
            page.rule(x0, y + 4, x1, y + 4, 1)
        y += int(size * 0.6)
    return y + int(size * 0.8)


def footer_lines(page: Page, src, font, size, x0, x1, y_bottom) -> int:
    """Page number and/or footer text sitting above y_bottom; returns the top used."""
    rng = page.rng
    y = y_bottom - int(size * 1.8)
    parts = []
    if rng.random() < 0.85:
        n = digits(page_no(rng), rng, font)
        parts.append([rng.choice(["", "-", "—"]) + n + rng.choice(["", "-", "—"])] if rng.random() < 0.3 else [n])
    if rng.random() < 0.35:
        parts.append([src.next() for _ in range(rng.randint(2, 6))])
    align_cycle = rng.choice([["center"], ["left", "right"], ["right", "left"]])
    for index, words in enumerate(parts):
        words = [w for w in words if w]
        line = make_line(page, words, font, size, x1 - x0, False) if words else None
        if line is not None:
            put_line(page, line, x1, y, FOOTER, align_cycle[index % len(align_cycle)], x1 - x0)
            if align_cycle == ["center"] and index < len(parts) - 1:
                y -= int(size * 1.8)  # centred parts stack instead of overlapping
    top = y - int(size * 0.5)
    if rng.random() < 0.4:
        page.rule(x0, top, x1, top, 1)
        top -= int(size * 0.5)
    return top


def footnotes(page: Page, src, font, size, x0, x1, y_bottom) -> int:
    """A few small footnote lines above y_bottom, under a short rule on the right. Returns top."""
    rng = page.rng
    small = max(9, int(size * rng.uniform(0.7, 0.85)))
    lead = int(small * rng.uniform(1.5, 1.9))
    notes = rng.randint(1, 4)
    lines = []
    for n in range(notes):
        count = rng.randint(1, 3)
        gen = fill_words(page, src, font, small, x1 - x0 - small * 2)
        for i in range(count):
            words = next(gen)
            if i == 0:
                mark = rng.choice(["(%s)", "%s.", "*", "%s-"])
                num = digits(n + 1, rng, font)
                words = [mark % num if "%s" in mark else mark] + words
            if i == count - 1 and len(words) > 2:
                words = words[: rng.randint(2, len(words))]
            lines.append(words)
    top = y_bottom - lead * len(lines)
    y = top
    for words in lines:
        line = make_line(page, words, font, small, x1 - x0, False)
        if line is not None:
            put_line(page, line, x1, y + int(small * 0.3), FOOTNOTE, "right", x1 - x0)
        y += lead
    rule_y = top - int(size * 0.4)
    if rng.random() < 0.85:
        frac = rng.choice([0.3, 0.33, 0.4, 1.0])
        page.rule(x1 - int((x1 - x0) * frac), rule_y, x1, rule_y, 1)
    return rule_y - int(size * 0.8)


def heading(page: Page, src, font, size, x0, x1, y, align=None, inverted=False) -> int:
    rng = page.rng
    hfont = rng.choice(bold_fonts() + [font])
    hsize = int(size * rng.uniform(1.3, 2.6))
    n = rng.randint(1, 7)
    words = [src.next() for _ in range(n)]
    gen = None
    line = make_line(page, words, hfont, hsize, x1 - x0, False)
    if line is None:
        return y
    if line[1][2] - line[1][0] > x1 - x0:
        gen = fill_words(page, src, hfont, hsize, x1 - x0)
        line = make_line(page, next(gen), hfont, hsize, x1 - x0, False)
        if line is None:
            return y
    align = align or rng.choice(["center", "right", "right"])
    h = line[1][3] - line[1][1]
    if inverted:
        page.fill(x0, y, x1, y + h + int(hsize * 0.6), 1.0)
        put_line(page, line, x1 - hsize // 3, y + int(hsize * 0.3), HEADING, align, x1 - x0 - hsize // 3 * 2, white=True)
        return y + h + int(hsize * 1.1)
    if rng.random() < 0.3 and size > 12:
        # bold by thickening
        canvas = cv2.dilate(line[0], np.ones((2, 2), np.uint8))
        line = (canvas, line[1])
    bottom = put_line(page, line, x1, y, HEADING, align, x1 - x0)
    return bottom + int(size * rng.uniform(0.6, 1.4))


def column_flow(page: Page, src, font, size, lead, x0, x1, y0, y1, pictures=0.15) -> None:
    rng = page.rng
    y = y0
    while y < y1 - lead:
        r = rng.random()
        if r < 0.12 and y1 - y > lead * 4:
            y = heading(page, src, font, size, x0, x1, y, inverted=rng.random() < 0.08)
        elif r < 0.12 + pictures and y1 - y > lead * 6:
            ph = int(min(y1 - y - lead * 2, (x1 - x0) * rng.uniform(0.4, 1.0)))
            picture(page, x0, y, x1, y + ph)
            y += ph + int(size * 0.5)
            if rng.random() < 0.7:  # caption
                y, _ = paragraph(page, src, font, max(9, int(size * 0.85)), x0, x1, y, y1, int(lead * 0.85), BODY, False, rng.randint(1, 2), rng.choice(["center", "right"]))
            y += int(size * 0.5)
        elif r < 0.12 + pictures + 0.05 and y1 - y > lead * 6:
            # shaded quote box
            n = rng.randint(2, 5)
            bh = n * lead + size
            page.fill(x0, y, x1, y + bh, rng.uniform(0.08, 0.3))
            paragraph(page, src, font, size, x0 + size // 2, x1 - size // 2, y + size // 2, y + bh, lead, BODY, True, n)
            y += bh + int(size * 0.6)
        elif r < 0.12 + pictures + 0.1:
            y = poetry(page, src, font, size, lead, x0, x1, y, y1)
        elif r < 0.12 + pictures + 0.15:
            y = numbered_items(page, src, font, size, lead, x0, x1, y, y1)
        elif r < 0.12 + pictures + 0.17 and y1 - y > lead * 5 and x1 - x0 > size * 12:
            y = banner(page, src, font, size, x0, x1, y)
        else:
            y, _ = paragraph(page, src, font, size, x0, x1, y, y1, lead)
            y += int(lead * rng.choice([0, 0, 0.3, 0.6]))


def poetry(page: Page, src, font, size, lead, x0, x1, y, y1) -> int:
    """Couplets: two half lines side by side, each its own line, right half first."""
    rng = page.rng
    width = x1 - x0
    half = int(width * rng.uniform(0.36, 0.46))
    for _ in range(rng.randint(1, 4)):
        if y + lead > y1:
            break
        for side in (0, 1):
            words = [src.next() for _ in range(rng.randint(2, 5))]
            line = make_line(page, words, font, size, half, False)
            if line is None or line[1][2] - line[1][0] > half:
                continue
            right = x1 if side == 0 else x0 + half
            put_line(page, line, right, y + int(size * 0.35), BODY, "center", half)
        y += lead
    return y + int(lead * 0.5)


def banner(page: Page, src, font, size, x0, x1, y) -> int:
    """Decorative title: big bold words outlined, shadowed, striped or white on a picture,
    often inside an ornamental frame. One HEADING line. Returns the next y."""
    rng = page.rng
    bold = bold_fonts()
    bfont = rng.choice(bold + [font])
    bsize = int(min(size * rng.uniform(2.4, 4.5), (x1 - x0) / 4))
    words = [src.next() for _ in range(rng.randint(1, 4))]
    line = make_line(page, words, bfont, bsize, x1 - x0, False)
    while line is not None and line[1][2] - line[1][0] > (x1 - x0) * 0.9 and len(words) > 1:
        words = words[:-1]
        line = make_line(page, words, bfont, bsize, x1 - x0, False)
    if line is None or line[1][2] - line[1][0] > (x1 - x0) * 0.9:
        return y
    cov = line[0]
    k = max(2, bsize // 14)
    thick = cv2.dilate(cov, np.ones((k, k), np.uint8))
    style = rng.choice(["outline", "shadow", "stripes", "on_picture", "framed", "outline_shadow"])
    if style in ("outline", "outline_shadow"):
        inner = cv2.erode(cov, np.ones((max(1, k - 1),) * 2, np.uint8))
        cov = np.clip(thick.astype(np.int16) - inner, 0, 255).astype(np.uint8)
    elif style == "stripes":
        step = max(2, bsize // 16)
        stripes = ((np.arange(cov.shape[0]) // step) % 2 == 0)[:, None]
        cov = np.maximum(np.where(stripes, cov, 0), np.clip(thick.astype(np.int16) - cov, 0, 255)).astype(np.uint8)
    if style in ("shadow", "outline_shadow"):
        d = max(2, bsize // 10)
        shadow = np.zeros_like(thick)
        shadow[d:, d:] = thick[:-d, :-d] // 2
        cov = np.maximum(cov, shadow)
    line = restyle(line, cov)
    h = line[1][3] - line[1][1]
    pad = int(bsize * rng.uniform(0.3, 0.7))
    top = y + pad
    align = rng.choice(["center", "center", "right"])
    white = False
    if style == "on_picture":
        picture(page, x0, y, x1, y + h + 2 * pad)
        # dark rim drawn first, then the letters in white
        rim = restyle(line, cv2.dilate(line[0], np.ones((k + 1, k + 1), np.uint8)))
        rim = (rim[0], line[1])
        put_line(page, rim, x1 - pad, top, BODY, align, x1 - x0 - 2 * pad)
        page.lines.pop()
        white = True
    put_line(page, line, x1 - pad, top, HEADING, align, x1 - x0 - 2 * pad, white=white)
    bottom = top + h + pad
    if style != "on_picture" and rng.random() < 0.6:
        t = rng.randint(1, 4)
        cv2.rectangle(page.ink, (x0, y), (x1, bottom), 1.0, t)
        if rng.random() < 0.5:
            g = t + rng.randint(2, 5)
            cv2.rectangle(page.ink, (x0 + g, y + g), (x1 - g, bottom - g), 1.0, 1)
        if rng.random() < 0.4:  # corner ornaments
            r = max(4, pad // 2)
            for cx, cy in ((x0, y), (x1, y), (x0, bottom), (x1, bottom)):
                cv2.circle(page.ink, (cx, cy), r, 1.0, -1 if rng.random() < 0.5 else 1)
    return bottom + int(size * rng.uniform(0.6, 1.4))


def numbered_items(page: Page, src, font, size, lead, x0, x1, y, y1) -> int:
    """List items: a number or bullet on the right, a wide blank, then the text, as one line."""
    rng = page.rng
    style = rng.choice(["(%s)", "%s.", "%s-", "%s)", "•", "*", "-"])
    gap = int(size * rng.uniform(1.0, 5.0))
    for n in range(rng.randint(2, 6)):
        if y + lead > y1:
            break
        mark = style % digits(n + 1, rng, font) if "%s" in style else style
        words = [mark] + [src.next() for _ in range(rng.randint(1, 8))]
        line = make_line(page, words, font, size, x1 - x0, False, {0: gap})
        while line is not None and line[1][2] - line[1][0] > x1 - x0 and len(words) > 2:
            words = words[:-1]
            line = make_line(page, words, font, size, x1 - x0, False, {0: gap})
        if line is None or line[1][2] - line[1][0] > x1 - x0:
            continue
        put_line(page, line, x1, y + int(size * 0.35), BODY, "right", x1 - x0)
        y += lead
    return y + int(lead * 0.4)


def contents_page(page: Page, src, font, size, lead, x0, x1, y, y1) -> int:
    """Table of contents: title on the right, a gap (blank or dotted) and the page number on
    the left, all one line."""
    rng = page.rng
    dot = page.word(".", font, size) if rng.random() < 0.5 else None
    number = page_no(rng)
    while y + lead <= y1:
        number = min(999, number + rng.randint(2, 30))
        words = [src.next() for _ in range(rng.randint(1, 6))]
        text_no = digits(number, rng, font)
        title = make_line(page, words, font, size, x1 - x0, False)
        num = make_line(page, [text_no], font, size, size * 4, False)
        if title is None or num is None:
            y += lead
            continue
        free = (x1 - x0) - (title[1][2] - title[1][0]) - (num[1][2] - num[1][0])
        if free < size * 2:
            y += lead
            continue
        if dot is not None:
            filler = ["." * max(1, int((free - size * 1.5) / max(1, dot[1])))]
            line = make_line(page, words + filler + [text_no], font, size, x1 - x0, False)
        else:
            line = make_line(page, words + [text_no], font, size, x1 - x0, False, {len(words) - 1: int(free - size * 1.5)})
        if line is None or line[1][2] - line[1][0] > x1 - x0:
            y += lead
            continue
        put_line(page, line, x1, y + int(size * 0.35), BODY, "right", x1 - x0)
        y += lead + (int(lead * 0.5) if rng.random() < 0.15 else 0)
    return y


def make_page(seed: int) -> tuple[np.ndarray, list]:
    rng = random.Random(seed)
    fonts = usable_fonts()
    font = rng.choice(fonts)
    src = WordSource(rng)
    kind = rng.choices(["book", "news", "magazine"], [0.45, 0.35, 0.2])[0]
    if kind == "book":
        width = rng.randint(900, 1400)
        height = int(width * rng.uniform(1.3, 1.6))
        size = rng.randint(14, 34)
        cols = rng.choices([1, 2], [0.85, 0.15])[0]
    elif kind == "news":
        width = rng.randint(1100, 1700)
        height = int(width * rng.uniform(1.15, 1.5))
        size = rng.randint(11, 24)
        cols = rng.randint(3, 6)
    else:
        width = rng.randint(1000, 1500)
        height = int(width * rng.uniform(1.25, 1.45))
        size = rng.randint(13, 26)
        cols = rng.randint(2, 3)
    page = Page(width, height, rng)
    lead = int(size * rng.uniform(1.45, 2.1))
    margin_x = int(width * rng.uniform(0.04, 0.1))
    x0, x1 = margin_x, width - margin_x
    y = int(height * rng.uniform(0.03, 0.07))
    y_end = height - int(height * rng.uniform(0.03, 0.07))

    if kind == "book" or rng.random() < 0.5:
        if rng.random() < 0.85:
            y = running_header(page, src, font, max(10, int(size * rng.uniform(0.75, 1.0))), x0, x1, y, rng.random() < 0.55)
    if rng.random() < (0.75 if kind == "book" else 0.4):
        y_end = footer_lines(page, src, font, max(10, int(size * rng.uniform(0.75, 1.0))), x0, x1, y_end)
    if kind == "book" and rng.random() < 0.5:
        y_end = footnotes(page, src, font, size, x0, x1, y_end)
    if kind == "book" and rng.random() < 0.06:
        if rng.random() < 0.5:
            y = banner(page, src, font, size, x0, x1, y) if rng.random() < 0.5 else heading(page, src, font, size, x0, x1, y, "center")
        contents_page(page, src, font, size, lead, x0, x1, y, y_end)
        return degrade(page, rng)
    if kind == "news" and rng.random() < 0.4:
        if rng.random() < 0.5:
            y = banner(page, src, font, size, x0, x1, y)
        else:
            y = heading(page, src, font, int(size * 1.8), x0, x1, y, "center", inverted=rng.random() < 0.2)
        page.rule(x0, y, x1, y, rng.randint(2, 5))
        y += size
    if kind != "book" and rng.random() < 0.3:
        y = banner(page, src, font, size, x0, x1, y)
    elif kind != "book" or rng.random() < 0.3:
        y = heading(page, src, font, size, x0, x1, y, inverted=rng.random() < 0.1)

    # columns, optionally broken by a spanning headline/picture band
    gutter = int(size * rng.uniform(0.8, 2.5))
    col_rules = rng.random() < 0.5 and cols > 1
    bands = [(y, y_end)]
    if cols > 2 and rng.random() < 0.5:
        mid = rng.randint(y + (y_end - y) // 3, y + 2 * (y_end - y) // 3)
        bands = [(y, mid), (mid, y_end)]
    for band_index, (b0, b1) in enumerate(bands):
        if band_index > 0:
            page.rule(x0, b0, x1, b0, rng.randint(1, 3))
            b0 = heading(page, src, font, size, x0, x1, b0 + size // 2) if rng.random() < 0.7 else b0 + size
        n = cols if band_index == 0 else rng.randint(2, cols)
        col_w = (x1 - x0 - gutter * (n - 1)) / n
        for c in range(n):
            cx1 = int(x1 - c * (col_w + gutter))  # first column on the right
            cx0 = int(cx1 - col_w)
            column_flow(page, src, font, size, lead, cx0, cx1, b0, b1, 0.1 if kind == "book" else 0.22)
            if col_rules and c < n - 1:
                gx = int(cx0 - gutter / 2)
                page.rule(gx, b0, gx, b1, 1)
    if rng.random() < 0.15:
        cv2.rectangle(page.ink, (x0 - size, int(height * 0.02)), (x1 + size, height - int(height * 0.02)), 1.0, rng.randint(1, 4))
    return degrade(page, rng)


# ---------------------------------------------------------------- scan look


def degrade(page: Page, rng: random.Random):
    h, w = page.h, page.w
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    ink = page.ink
    if rng.random() < 0.3:  # faint text from the other side of the sheet
        ink = np.maximum(ink, ink[:, ::-1] * rng.uniform(0.05, 0.18))
    if rng.random() < 0.3:  # worn / faded ink
        ink = ink * cv2.resize(nrng.uniform(0.55, 1.0, (8, 8)).astype(np.float32), (w, h))
    paper = rng.uniform(205, 252)
    light = cv2.resize(nrng.uniform(-1, 1, (4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(0, 30)
    ink_tone = rng.uniform(0, 70)
    img = (paper + light) * (1 - ink) + ink_tone * ink
    if rng.random() < 0.3:  # scanner shadow at one edge
        side = rng.randrange(4)
        band = int(min(h, w) * rng.uniform(0.01, 0.05))
        ramp = np.linspace(rng.uniform(40, 120), 0, band, dtype=np.float32)
        if side == 0:
            img[:, :band] -= ramp[None, :]
        elif side == 1:
            img[:, -band:] -= ramp[::-1][None, :]
        elif side == 2:
            img[:band] -= ramp[:, None]
        else:
            img[-band:] -= ramp[::-1][:, None]
    img += nrng.normal(0, rng.uniform(0, 10), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    quads = [np.array(q, np.float32) for q, _k in page.lines]
    # small rotation / perspective: the server deskews, but not perfectly
    angle = rng.uniform(-2.5, 2.5) if rng.random() < 0.6 else 0.0
    if angle or rng.random() < 0.2:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = min(w, h) * 0.02
        dst = src + np.float32([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)]) * (rng.random() < 0.3)
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        dst = (np.hstack([dst, np.ones((4, 1), np.float32)]) @ rot.T).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=int(paper))
        quads = [cv2.perspectiveTransform(q[None], matrix)[0] for q in quads]
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.3:
        f = rng.uniform(0.5, 0.85)
        img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.12:
        _t, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labels = [[q.round(1).tolist(), kind] for q, (_q, kind) in zip(quads, page.lines)]
    return img, labels


def write_one(index: int) -> int:
    img, labels = make_page(index * 7919 + 13)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.Random(index).randint(45, 92)])
    (OUT / f"{index:05d}.jpg").write_bytes(buf.tobytes())
    (OUT / f"{index:05d}.json").write_text(json.dumps({"lines": labels}, ensure_ascii=False), encoding="utf-8")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(args.start, args.start + args.count) if not (OUT / f"{i:05d}.json").exists()]
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for _n in pool.map(write_one, todo, chunksize=4):
            done += 1
            if done % 200 == 0:
                print(f"pages {done}/{len(todo)}", flush=True)
    print(f"done {done}", flush=True)


if __name__ == "__main__":
    main()
