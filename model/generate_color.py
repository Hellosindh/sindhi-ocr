"""Colourful poster / banner / invitation style Sindhi line images for the CRNN reader.

Every image is one text line cut the way the page detector would cut it, on hard
backgrounds seen in real posters: flat colours, gradients, photo-like blur, ajrak-like
patterns, banner plates, decorative ornaments; text in colour with outlines, shadows,
3D extrusion, glow, stretched kashida; then tilt, blur, low resolution and JPEG damage.

Text and font pairs come from assets/synthetic_plain/transcript.tsv (already checked to
render); lines are cut short like poster lines. The held-out test font (SINDHI_HELD_OUT_FONT) is never used.
Output: assets/synthetic_color/color/NNNN/NNNNNNN.jpg + transcript.tsv
(filename, font, kind, text) — the same columns as synthetic_plain.

Run:  python -m model.generate_color [--count 30000] [--workers 6] [--preview 40]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "assets" / "synthetic_plain" / "transcript.tsv"
OUT = ROOT / "assets" / "synthetic_color"
FONT_DIR = ROOT / "Fonts"
HELD_OUT = os.environ.get("SINDHI_HELD_OUT_FONT", "")  # file name of one font kept out of all synthetic data, to test on
TATWEEL = "ـ"
# letters that never join to the next one: no kashida after them
NON_JOINING = set("اآأإٱدڊڌڏڍذرڙزژوۄؤۅۆۇۈۉۋۏءۀ۽ﺀ")

# poster palette: Sindhi posters lean on red, maroon, yellow, green, blue, black, white
PALETTE = [
    (200, 20, 30), (150, 10, 20), (110, 15, 25), (230, 60, 40), (250, 210, 30), (255, 235, 90),
    (20, 120, 50), (10, 80, 40), (30, 60, 160), (15, 30, 90), (0, 140, 200), (120, 30, 140),
    (240, 120, 20), (0, 0, 0), (255, 255, 255), (245, 240, 225), (90, 50, 20), (200, 160, 60),
    (250, 180, 190), (40, 40, 40), (180, 0, 90), (0, 170, 160),
]
AJRAK = [(130, 15, 25), (20, 25, 70), (0, 0, 0), (245, 240, 230)]


def luminance(color) -> float:
    r, g, b = color[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------- text mask (same shaping as make_synth.draw_line, any size, soft edges) ----------

_FONTS: dict[tuple[str, int], tuple] = {}


def _faces(name: str, size: int):
    key = (name, size)
    if key not in _FONTS:
        import freetype
        import uharfbuzz as hb

        path = str(FONT_DIR / name)
        font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
        font.scale = (size * 64, size * 64)
        face = freetype.Face(path)
        face.set_char_size(size * 64)
        _FONTS[key] = (font, face)
    return _FONTS[key]


def text_mask(text: str, font_name: str, size: int, spacing: int = 0) -> np.ndarray | None:
    """Anti-aliased ink mask (float 0..1) of one shaped line, or None if a glyph is missing."""
    import freetype
    import uharfbuzz as hb
    from make_synth import bidi_runs

    font, face = _faces(font_name, size)
    placed, pen = [], 0
    for piece, direction in bidi_runs(text):
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
    baseline = size * 2
    stamps = []
    for index, (glyph_id, x_pos, y_pos) in enumerate(placed):
        face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)
        bitmap = face.glyph.bitmap
        if bitmap.width == 0 or bitmap.rows == 0:
            continue
        glyph = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.pitch)[:, : bitmap.width]
        left = face.glyph.bitmap_left + x_pos // 64 + spacing * index
        top = baseline - face.glyph.bitmap_top - y_pos // 64
        stamps.append((glyph, left, top))
    if not stamps:
        return None
    min_x = min(l for _g, l, _t in stamps)
    min_y = min(t for _g, _l, t in stamps)
    max_x = max(l + g.shape[1] for g, l, _t in stamps)
    max_y = max(t + g.shape[0] for g, _l, t in stamps)
    mask = np.zeros((max_y - min_y, max_x - min_x), np.float32)
    for glyph, left, top in stamps:
        y, x = top - min_y, left - min_x
        region = mask[y : y + glyph.shape[0], x : x + glyph.shape[1]]
        np.maximum(region, glyph / 255.0, out=region)
    return mask


def stretch(text: str, rng: random.Random) -> str:
    """Insert kashida the way poster designers stretch words (label keeps the plain text)."""
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        nxt = text[i + 1] if i + 1 < len(text) else " "
        if ch not in NON_JOINING and ch != " " and nxt != " " and "؀" <= ch <= "ۿ" and "؀" <= nxt <= "ۿ" \
                and not ("ً" <= nxt <= "ٟ") and rng.random() < 0.18:
            out.append(TATWEEL * rng.randint(1, 4))
    return "".join(out)


# ---------- backgrounds ----------

def _noise_field(h: int, w: int, rng: np.random.Generator, scale: int) -> np.ndarray:
    import cv2

    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def background(h: int, w: int, kind: str, rng: np.random.Generator, pyr: random.Random) -> np.ndarray:
    import cv2

    if kind == "flat":
        return np.full((h, w, 3), pyr.choice(PALETTE), np.float32)
    if kind == "gradient":
        a, b = np.array(pyr.choice(PALETTE), np.float32), np.array(pyr.choice(PALETTE), np.float32)
        angle = pyr.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (xx * np.cos(angle) + yy * np.sin(angle))
        t = (t - t.min()) / max(1e-3, t.max() - t.min())
        return a * (1 - t[..., None]) + b * t[..., None]
    if kind == "photo":
        # blurred colour clouds and light spots, like text over a crowd or stage photo
        base = np.zeros((h, w, 3), np.float32)
        for _ in range(3):
            colour = np.array(pyr.choice(PALETTE), np.float32)
            base += _noise_field(h, w, rng, pyr.choice([8, 16, 32]))[..., None] * colour
        base /= 1.6
        for _ in range(pyr.randint(5, 40)):
            cv2.circle(base, (pyr.randrange(w), pyr.randrange(h)), pyr.randint(2, max(3, h // 2)),
                       tuple(float(c) for c in pyr.choice(PALETTE)), -1)
        return cv2.GaussianBlur(base, (0, 0), pyr.uniform(1.5, 6))
    if kind == "pattern":
        # ajrak-like tiles: circles, diamonds and stars in maroon / indigo / black / white
        cell = pyr.randint(10, 28)
        img = np.full((h, w, 3), pyr.choice(AJRAK[:2]), np.float32)
        colours = [tuple(float(c) for c in pyr.choice(AJRAK)) for _ in range(3)]
        for y in range(0, h + cell, cell):
            for x in range(0, w + cell, cell):
                c = (x + (cell // 2 if (y // cell) % 2 else 0), y)
                shape = (x // cell + y // cell) % 3
                if shape == 0:
                    cv2.circle(img, c, cell // 3, colours[0], -1)
                    cv2.circle(img, c, cell // 6, colours[1], -1)
                elif shape == 1:
                    pts = np.array([[c[0], c[1] - cell // 2], [c[0] + cell // 2, c[1]], [c[0], c[1] + cell // 2], [c[0] - cell // 2, c[1]]])
                    cv2.polylines(img, [pts], True, colours[2], 1)
                else:
                    cv2.drawMarker(img, c, colours[1], cv2.MARKER_STAR, cell // 2, 1)
        return img
    if kind == "paper":
        tone = pyr.choice([(255, 255, 255), (250, 247, 238), (238, 234, 222), (255, 252, 240)])
        img = np.full((h, w, 3), tone, np.float32)
        img += (_noise_field(h, w, rng, 4)[..., None] - 0.5) * pyr.uniform(0, 18)
        return img
    # halftone print: dots of one colour on another
    a, b = pyr.choice(PALETTE), pyr.choice(PALETTE)
    img = np.full((h, w, 3), a, np.float32)
    step = pyr.randint(3, 7)
    for y in range(0, h, step):
        for x in range((y // step) % 2 * step // 2, w, step):
            cv2.circle(img, (x, y), max(1, step // 3), tuple(float(c) for c in b), -1)
    return img


# ---------- one image ----------

def _contrasting(bg_lum: float, pyr: random.Random, min_gap: float = 90) -> tuple:
    choices = [c for c in PALETTE if abs(luminance(c) - bg_lum) >= min_gap]
    return pyr.choice(choices or [(0, 0, 0) if bg_lum > 128 else (255, 255, 255)])


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _over(canvas: np.ndarray, alpha: np.ndarray, colour) -> None:
    a = np.clip(alpha, 0, 1)[..., None]
    if isinstance(colour, np.ndarray) and colour.ndim == 3:
        canvas[:] = canvas * (1 - a) + colour * a
    else:
        canvas[:] = canvas * (1 - a) + np.array(colour, np.float32) * a


def _ornament(canvas: np.ndarray, x: int, cy: int, size: int, colour, pyr: random.Random) -> None:
    """Small flourish beside the text: scrolls, circles, diamonds, a rule."""
    import cv2

    c = tuple(float(v) for v in colour)
    kind = pyr.randrange(4)
    if kind == 0:
        for k in (-1, 0, 1):
            cv2.ellipse(canvas, (x + k * size // 4, cy), (size // 6, size // 7), 0, pyr.randint(0, 90), 300, c, max(1, size // 14))
    elif kind == 1:
        cv2.circle(canvas, (x, cy), size // 3, c, max(1, size // 10))
        cv2.circle(canvas, (x, cy), size // 8, c, -1)
    elif kind == 2:
        pts = np.array([[x, cy - size // 3], [x + size // 3, cy], [x, cy + size // 3], [x - size // 3, cy]])
        cv2.fillPoly(canvas, [pts], c)
    else:
        cv2.line(canvas, (x - size // 3, cy), (x + size // 3, cy), c, max(1, size // 14))


def render(text: str, font: str, seed: int) -> tuple[np.ndarray, str] | None:
    import cv2

    pyr = random.Random(seed)
    rng = np.random.default_rng(seed)
    size = pyr.choice([40, 48, 56, 64, 72])
    shown = stretch(text, pyr) if pyr.random() < 0.12 else text
    mask = text_mask(shown, font, size, spacing=pyr.choice([0, 0, 0, 0, 1, 2, 4]))
    if mask is None and shown != text:
        mask = text_mask(text, font, size)
    if mask is None:
        return None
    if pyr.random() < 0.35:  # faux bold, poster fonts are heavy
        mask = _dilate(mask, pyr.randint(1, max(1, size // 22)))

    style = pyr.choices(
        ["flat", "gradient", "photo", "pattern", "paper", "halftone"], weights=[22, 18, 20, 12, 20, 8]
    )[0]
    outline = pyr.random() < (0.55 if style in ("photo", "pattern", "halftone") else 0.3)
    stroke = pyr.randint(1, max(2, size // 10)) if outline else 0
    plate = pyr.random() < 0.18  # banner box behind the text
    extra = 2 * stroke + (size // 6 if plate else 0) + size // 8 + 6
    pad_x = pyr.randint(4, size // 2) + extra
    pad_top, pad_bottom = pyr.randint(2, size // 3) + extra, pyr.randint(2, size // 3) + extra
    h, w = mask.shape[0] + pad_top + pad_bottom, mask.shape[1] + 2 * pad_x
    if pyr.random() < 0.15:  # ornaments beside the text
        w += 2 * size
        pad_x += size
    full = np.zeros((h, w), np.float32)
    full[pad_top : pad_top + mask.shape[0], pad_x : pad_x + mask.shape[1]] = mask

    canvas = background(h, w, style, rng, pyr)
    bg_lum = float(luminance(canvas.reshape(-1, 3).mean(0)))

    if plate:
        plate_colour = _contrasting(bg_lum, pyr, 60)
        ys, xs = np.nonzero(full > 0.2)
        m = size // 6
        x0, x1, y0, y1 = xs.min() - m - stroke, xs.max() + m + stroke, ys.min() - m // 2, ys.max() + m // 2
        cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), tuple(float(c) for c in plate_colour), -1)
        if pyr.random() < 0.6:
            edge = _contrasting(luminance(plate_colour), pyr, 60)
            cv2.rectangle(canvas, (int(x0) + 2, int(y0) + 2), (int(x1) - 2, int(y1) - 2), tuple(float(c) for c in edge), max(1, size // 20))
        bg_lum = luminance(plate_colour)

    fill = _contrasting(bg_lum, pyr, 70 if outline else 100)
    if pyr.random() < 0.15:  # gradient fill
        second = np.array(_contrasting(bg_lum, pyr, 70 if outline else 100), np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        fill = np.array(fill, np.float32) * (1 - t) + second * t
        fill = np.broadcast_to(fill, (h, w, 3)).copy()

    effect = pyr.random()
    if effect < 0.2:  # drop shadow
        dx, dy = pyr.randint(1, max(2, size // 12)), pyr.randint(1, max(2, size // 12))
        shadow = np.roll(np.roll(_dilate(full, stroke), dy, 0), dx, 1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), pyr.uniform(0.5, 3))
        _over(canvas, shadow * pyr.uniform(0.5, 0.9), (0, 0, 0) if bg_lum > 60 else (255, 255, 255))
    elif effect < 0.3:  # 3D extrusion
        depth = pyr.randint(2, max(3, size // 10))
        side = _contrasting(luminance(fill if not isinstance(fill, np.ndarray) else fill.mean((0, 1))), pyr, 50)
        for k in range(depth, 0, -1):
            _over(canvas, np.roll(np.roll(_dilate(full, stroke), k, 0), k, 1), side)
    elif effect < 0.37:  # glow
        glow = cv2.GaussianBlur(_dilate(full, max(2, size // 12)), (0, 0), size / 10)
        _over(canvas, np.clip(glow * 1.5, 0, 1), (255, 255, 255) if bg_lum < 128 else (255, 240, 120))

    if outline:
        fill_lum = luminance(fill if not isinstance(fill, np.ndarray) else fill.mean((0, 1)))
        outer = _contrasting(fill_lum, pyr, 90)
        if pyr.random() < 0.3:  # double outline
            _over(canvas, _dilate(full, stroke * 2), _contrasting(luminance(outer), pyr, 80))
        _over(canvas, _dilate(full, stroke), outer)
    _over(canvas, full, fill)
    gray = canvas @ np.array([0.299, 0.587, 0.114], np.float32)
    core = full > 0.6
    near = (_dilate(full, max(2, stroke + 1)) > 0.3) & ~(_dilate(full, 1) > 0.3)
    if core.sum() < 20 or near.sum() < 20:
        return None
    ink = float(np.median(gray[core]))
    around = np.abs(gray[near] - ink)
    if float(around.mean()) < 70 or float(np.mean(around < 40)) > 0.15:
        return None  # text would not stand out from what touches it

    if pad_x > size and w > 3 * size:  # ornaments sit in the extra side room
        colour = _contrasting(bg_lum, pyr, 70)
        _ornament(canvas, size // 2, h // 2, size, colour, pyr)
        _ornament(canvas, w - size // 2, h // 2, size, colour, pyr)
    if pyr.random() < 0.15:  # decorative rule under or over the line, partly touching it
        y = pyr.choice([pad_top // 2, h - pad_bottom // 2])
        cv2.line(canvas, (pyr.randint(0, w // 4), y), (w - pyr.randint(0, w // 4), y),
                 tuple(float(c) for c in _contrasting(bg_lum, pyr, 60)), max(1, size // 18))

    # camera / print / social-media damage
    if pyr.random() < 0.35:
        angle = pyr.uniform(-3.5, 3.5)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    if pyr.random() < 0.2:
        shift = pyr.uniform(-0.08, 0.08) * h
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[0, 0], [w, shift], [0, h], [w, h + shift]])
        canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    if pyr.random() < 0.35:
        canvas = cv2.GaussianBlur(canvas, (0, 0), pyr.uniform(0.4, 1.6))
    if pyr.random() < 0.35:  # low resolution, then blown up again
        f = pyr.uniform(0.3, 0.7)
        small = cv2.resize(canvas, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(small, (w, h), interpolation=pyr.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
    if pyr.random() < 0.25:
        canvas = np.clip(canvas.astype(np.int16) + rng.normal(0, pyr.uniform(3, 14), canvas.shape).astype(np.int16), 0, 255).astype(np.uint8)
    # final height like real detector crops
    target_h = pyr.randint(36, 96)
    canvas = cv2.resize(canvas, (max(16, round(w * target_h / h)), target_h), interpolation=cv2.INTER_AREA)
    return canvas, style


def work(task: tuple) -> str | None:
    import cv2

    index, font, text, out = task
    os.chdir(ROOT)
    result = None
    for attempt in range(6):
        result = render(text, font, seed=index * 7919 + 13 + attempt * 104729)
        if result is not None:
            break
    if result is None:
        return None
    image, style = result
    rel = f"color/{index // 10000:04d}/{index:07d}.jpg"
    path = Path(out) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = random.Random(index).choice([35, 50, 65, 80, 92])
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return f"{rel}\t{font}\t{style}\t{text}"


def sample_sources(count: int, rng: random.Random) -> list[tuple[str, str]]:
    """(font, text) from random places in the big synthetic_plain transcript, cut to poster length."""
    size = SOURCE.stat().st_size
    out = []
    with SOURCE.open("rb") as handle:
        while len(out) < count:
            handle.seek(rng.randrange(size))
            handle.readline()
            raw = handle.readline().decode("utf-8", "ignore").rstrip("\n").split("\t")
            if len(raw) != 4 or raw[1] == HELD_OUT:
                continue
            words = raw[3].split()
            n = min(len(words), rng.choice([1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 10]))
            start = rng.randrange(len(words) - n + 1)
            text = " ".join(words[start : start + n])
            if rng.random() < 0.08:  # dates, times, phone numbers as on posters
                text += " " + rng.choice([f"{rng.randint(1, 31)}/{rng.randint(1, 12)}/{rng.randint(1990, 2026)}",
                                          f"{rng.randint(1, 12)}:00", f"03{rng.randint(0, 49):02d}-{rng.randint(1000000, 9999999)}",
                                          str(rng.randint(1, 2026))])
            out.append((raw[1], text))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--preview", type=int, default=0, help="only write this many samples to OUT/preview")
    args = parser.parse_args()
    out = OUT / "preview" if args.preview else OUT
    rng = random.Random(2026)
    count = args.preview or args.count
    pairs = sample_sources(count, rng)
    tasks = [(i, font, text, str(out)) for i, (font, text) in enumerate(pairs)]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(work, tasks, chunksize=64):
            if row:
                rows.append(row)
            if len(rows) % 2000 == 0 and rows:
                print(f"{len(rows)} images {time.time() - started:.0f}s", flush=True)
    (out / "transcript.tsv").write_text("filename\tfont\tkind\ttext\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"done {len(rows)} images in {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
