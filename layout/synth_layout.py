"""Synthetic Sindhi pages with region boxes, for the layout model (same classes as DocLayNet plus
Contents-entry). Builds on linedet.synth_pages: its drawing functions are wrapped so every call
records the region it drew (paragraph -> Text, heading -> Section-header, banner -> Title, picture ->
Picture, caption after a picture -> Caption, list -> List-item, running header -> Page-header,
footer -> Page-footer, footnotes -> Footnote, contents line -> Contents-entry). New page types here:
tables (right-to-left, several ruling styles) and book covers.

Region boxes go through the same tilt / perspective as the page (they ride along with the line
quads in synth_pages.degrade).

Run:  python -m layout.synth_layout [--count 6000] [--test 300]
Out:  assets/layout_data/sindhi_synth/{train,test}/NNNNNNN.jpg + labels.jsonl  (same format as doclaynet)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import linedet.synth_pages as sp
from layout.model import CLASSES

OUT = ROOT / "assets" / "layout_data" / "sindhi_synth"
C = {name: i for i, name in enumerate(CLASSES)}
REGION = 100  # page.lines kinds >= REGION are region boxes (class = kind - REGION)

_orig = {name: getattr(sp, name) for name in
         ("paragraph", "heading", "banner", "picture", "poetry", "numbered_items", "running_header",
          "footer_lines", "footnotes", "contents_page")}


def _quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _add_region(page, x0, y0, x1, y1, cls: int) -> None:
    if x1 - x0 >= 3 and y1 - y0 >= 3:
        page.lines.append((_quad(x0, y0, x1, y1), REGION + cls))


def _lines_since(page, start):
    return [q for q, kind in page.lines[start:] if kind < REGION]


def _box(quads, pad=0):
    pts = np.array([p for q in quads for p in q], np.float32)
    return (float(pts[:, 0].min()) - pad, float(pts[:, 1].min()) - pad, float(pts[:, 0].max()) + pad, float(pts[:, 1].max()) + pad)


def _wrap_block(name: str, cls_fn):
    """Whole call becomes one region (class from cls_fn(page))."""
    def wrapped(page, *args, **kwargs):
        start = len(page.lines)
        result = _orig[name](page, *args, **kwargs)
        quads = _lines_since(page, start)
        if quads:
            _add_region(page, *_box(quads, 2), cls_fn(page))
        return result
    return wrapped


def _wrap_per_line(name: str, cls: int):
    """Every line drawn by the call is its own region (list items, contents entries)."""
    def wrapped(page, *args, **kwargs):
        start = len(page.lines)
        result = _orig[name](page, *args, **kwargs)
        for q in _lines_since(page, start):
            _add_region(page, *_box([q], 2), cls)
        return result
    return wrapped


def _paragraph_class(page):
    if getattr(page, "after_picture", False):
        page.after_picture = False
        return C["Caption"]
    return C["Text"]


def _picture(page, x0, y0, x1, y1):
    _orig["picture"](page, x0, y0, x1, y1)
    _add_region(page, x0, y0, x1, y1, C["Picture"])
    page.after_picture = True


def _banner(page, *args, **kwargs):
    start = len(page.lines)
    result = _orig["banner"](page, *args, **kwargs)
    page.after_picture = False
    quads = _lines_since(page, start)
    if quads:
        _add_region(page, *_box(quads, 3), C["Title"])
    return result


def install() -> None:
    sp.paragraph = _wrap_block("paragraph", _paragraph_class)
    sp.heading = _wrap_block("heading", lambda p: C["Section-header"])
    sp.poetry = _wrap_block("poetry", lambda p: C["Text"])
    sp.running_header = _wrap_block("running_header", lambda p: C["Page-header"])
    sp.footer_lines = _wrap_block("footer_lines", lambda p: C["Page-footer"])
    sp.footnotes = _wrap_block("footnotes", lambda p: C["Footnote"])
    sp.numbered_items = _wrap_per_line("numbered_items", C["List-item"])
    sp.contents_page = _wrap_per_line("contents_page", C["Contents-entry"])
    sp.picture = _picture
    sp.banner = _banner


# ---------------------------------------------------------------- new page parts


def table(page, src, font, size, x0, x1, y, y1) -> int:
    """A right-to-left table: header row, a few to many rows of words and numbers, ruled in one of
    several styles. Returns the next y."""
    rng = page.rng
    cols = rng.randint(2, 6)
    rows = rng.randint(3, 14)
    small = max(9, int(size * rng.uniform(0.75, 1.0)))
    row_h = int(small * rng.uniform(1.7, 2.3))
    if y + row_h * (rows + 1) > y1:
        rows = max(2, (y1 - y) // row_h - 1)
        if rows < 2:
            return y
    width = x1 - x0
    weights = [rng.uniform(0.6, 2.0) for _ in range(cols)]
    col_w = [int(width * w / sum(weights)) for w in weights]
    style = rng.choice(["grid", "grid", "rows", "header_only", "none", "outer"])
    numeric = [rng.random() < 0.5 for _ in range(cols)]
    top = y
    for r in range(rows + 1):
        x_right = x1
        header = r == 0
        for c in range(cols):
            w = col_w[c]
            if header:
                words = [src.next() for _ in range(rng.randint(1, 2))]
            elif numeric[c]:
                v = rng.choice([rng.randint(1, 999), rng.randint(1000, 99999)])
                words = [sp.digits(v, rng, font) if isinstance(v, int) else str(v)]
            else:
                words = [src.next() for _ in range(rng.randint(1, 3))]
            hfont = rng.choice([f for f in sp.BOLD_FONTS if (sp.FONT_DIR / f).exists()] + [font]) if header else font
            line = sp.make_line(page, words, hfont, small, w - small, False)
            while line is not None and line[1][2] - line[1][0] > w - small and len(words) > 1:
                words = words[:-1]
                line = sp.make_line(page, words, hfont, small, w - small, False)
            if line is not None and line[1][2] - line[1][0] <= w - small:
                align = "center" if header or numeric[c] else "right"
                sp.put_line(page, line, x_right - small // 2, y + int(small * 0.35), sp.BODY, align, w - small)
            x_right -= w
        y += row_h
        if style == "grid" or style == "rows" or (header and style in ("header_only", "grid")):
            page.rule(x0, y, x1, y, 2 if header else 1)
    if style in ("grid", "outer"):
        cv2.rectangle(page.ink, (x0, top), (x1, y), 1.0, rng.randint(1, 2))
    if style == "grid":
        xr = x1
        for w in col_w[:-1]:
            xr -= w
            page.rule(xr, top, xr, y, 1)
    if style in ("rows", "header_only"):
        page.rule(x0, top, x1, top, 2)
    _add_region(page, x0 - 2, top - 2, x1 + 2, y + 2, C["Table"])
    # geometry for the table-structure data (layout.synth_tables)
    page.last_table = {"top": top, "bottom": y, "row_h": row_h, "n_rows": rows + 1, "x0": x0, "x1": x1, "col_w": col_w}
    return y + int(size * 1.2)


def cover(seed: int):
    rng = random.Random(seed)
    fonts = sp.usable_fonts()
    font = rng.choice(fonts)
    src = sp.WordSource(rng)
    width = rng.randint(900, 1300)
    height = int(width * rng.uniform(1.35, 1.55))
    page = sp.Page(width, height, rng)
    size = rng.randint(18, 30)
    x0, x1 = int(width * 0.08), int(width * 0.92)
    if rng.random() < 0.5:  # background art
        sp.picture(page, 0, 0, width, height)
        page.lines.pop()  # full-page background is not a picture region
        page.ink *= rng.uniform(0.15, 0.4)
        page.after_picture = False
    y = int(height * rng.uniform(0.06, 0.2))
    y = sp.banner(page, src, font, size, x0, x1, y)
    if rng.random() < 0.6:
        y = sp.heading(page, src, font, size, x0, x1, y + size, "center")
    if rng.random() < 0.6:
        ph = int(height * rng.uniform(0.2, 0.35))
        px0 = int(width * rng.uniform(0.15, 0.3))
        sp.picture(page, px0, y + size, width - px0, y + size + ph)
        page.after_picture = False
        y += ph + size * 2
    # author name and publisher: short centred text lines
    for _ in range(rng.randint(1, 2)):
        y, _ok = sp.paragraph(page, src, font, int(size * rng.uniform(0.9, 1.3)), x0, x1, y + size, height, int(size * 2), sp.BODY, False, 1, "center")
    sp.paragraph(page, src, font, int(size * 0.8), x0, x1, height - int(height * 0.1), height, int(size * 2), sp.BODY, False, 1, "center")
    return page, rng


def table_page(seed: int):
    rng = random.Random(seed)
    fonts = sp.usable_fonts()
    font = rng.choice(fonts)
    src = sp.WordSource(rng)
    width = rng.randint(900, 1500)
    height = int(width * rng.uniform(1.3, 1.5))
    page = sp.Page(width, height, rng)
    size = rng.randint(13, 26)
    lead = int(size * rng.uniform(1.45, 2.0))
    x0, x1 = int(width * rng.uniform(0.05, 0.1)), int(width * rng.uniform(0.9, 0.95))
    y = int(height * 0.05)
    y_end = height - int(height * 0.05)
    if rng.random() < 0.7:
        y = sp.running_header(page, src, font, max(10, int(size * 0.85)), x0, x1, y, rng.random() < 0.5)
    if rng.random() < 0.7:
        y_end = sp.footer_lines(page, src, font, max(10, int(size * 0.85)), x0, x1, y_end)
    while y < y_end - lead * 3:
        r = rng.random()
        if r < 0.15:
            y = sp.heading(page, src, font, size, x0, x1, y)
        elif r < 0.5:
            y = table(page, src, font, size, x0, x1, y, y_end)
        else:
            y, _ok = sp.paragraph(page, src, font, size, x0, x1, y, y_end, lead, max_lines=rng.randint(2, 6))
            y += lead // 2
    return page, rng


def contents(seed: int):
    rng = random.Random(seed)
    fonts = sp.usable_fonts()
    font = rng.choice(fonts)
    src = sp.WordSource(rng)
    width = rng.randint(900, 1400)
    height = int(width * rng.uniform(1.3, 1.6))
    page = sp.Page(width, height, rng)
    size = rng.randint(14, 30)
    lead = int(size * rng.uniform(1.5, 2.2))
    x0, x1 = int(width * rng.uniform(0.06, 0.12)), int(width * rng.uniform(0.88, 0.94))
    y = int(height * rng.uniform(0.04, 0.1))
    y_end = height - int(height * 0.06)
    if rng.random() < 0.6:
        y_end = sp.footer_lines(page, src, font, max(10, int(size * 0.85)), x0, x1, y_end)
    y = sp.banner(page, src, font, size, x0, x1, y) if rng.random() < 0.3 else sp.heading(page, src, font, size, x0, x1, y, "center")
    r = rng.random()
    if r < 0.3:  # magazine contents: section headings between short runs of entries, thumbnails beside
        cx0 = x0
        if rng.random() < 0.7:
            split = int(x0 + (x1 - x0) * rng.uniform(0.35, 0.55))
            side_right = rng.random() < 0.5
            px0, px1 = (split + size, x1) if side_right else (x0, split - size)
            cx0, cx1 = (x0, split - size) if side_right else (split + size, x1)
            py = y + size
            while py < y_end - height * 0.12:
                ph = int(height * rng.uniform(0.08, 0.16))
                if py + ph > y_end:
                    break
                sp.picture(page, px0, py, px1, py + ph)
                page.after_picture = False
                py += ph + size
        else:
            cx1 = x1
        while y < y_end - lead * 2:
            y = sp.heading(page, src, font, size, cx0, cx1, y)
            y = contents_columns(page, src, font, size, lead, cx0, cx1, y, min(y_end, y + lead * rng.randint(2, 5)))
            y += lead // 2
    elif r < 0.55:
        contents_columns(page, src, font, size, lead, x0, x1, y, y_end)
    else:
        sp.contents_page(page, src, font, size, lead, x0, x1, y, y_end)
    return page, rng


# ---------------------------------------------------------------- imprint, dialogue, column contents


def _record(page, start, cls, pad=2):
    quads = _lines_since(page, start)
    if quads:
        _add_region(page, *_box(quads, pad), cls)


def kv_block(page, src, font, size, lead, x0, x1, y, y1) -> int:
    """Imprint lines "key : value" (book details page): keys in a right column, values to their left.
    The whole block is one Text region."""
    rng = page.rng
    key_w = int((x1 - x0) * rng.uniform(0.25, 0.38))
    colon = rng.random() < 0.8
    key_align = rng.choice(["right", "left"])
    start = len(page.lines)
    for _ in range(rng.randint(4, 12)):
        if y + lead > y1:
            break
        key = sp.make_line(page, [src.next() for _ in range(rng.choice([1, 1, 2]))] + ([":"] if colon else []),
                           font, size, key_w, False)
        if rng.random() < 0.35:
            value = [sp.digits(rng.randint(1, 9999), rng, font)] + ([src.next()] if rng.random() < 0.5 else [])
        else:
            value = [src.next() for _ in range(rng.randint(1, 6))]
        val = sp.make_line(page, value, font, size, x1 - x0 - key_w, False)
        if key is None or val is None or key[1][2] - key[1][0] > key_w or val[1][2] - val[1][0] > x1 - x0 - key_w - size:
            y += lead
            continue
        top = y + int(size * 0.35)
        sp.put_line(page, key, x1, top, sp.BODY, key_align, key_w)
        sp.put_line(page, val, x1 - key_w - size, top, sp.BODY, "right", x1 - x0 - key_w)
        y += lead
    _record(page, start, C["Text"])
    return y + lead // 2


def imprint(seed: int):
    """Book-details page: title, a key : value block or two, maybe a note and a logo."""
    rng = random.Random(seed)
    font = rng.choice(sp.usable_fonts())
    src = sp.WordSource(rng)
    width = rng.randint(900, 1400)
    height = int(width * rng.uniform(1.3, 1.6))
    page = sp.Page(width, height, rng)
    size = rng.randint(14, 26)
    lead = int(size * rng.uniform(1.4, 2.0))
    x0, x1 = int(width * rng.uniform(0.08, 0.2)), int(width * rng.uniform(0.8, 0.92))
    y = int(height * rng.uniform(0.05, 0.15))
    y_end = height - int(height * 0.06)
    if rng.random() < 0.5:
        y_end = sp.footer_lines(page, src, font, max(10, int(size * 0.85)), int(width * 0.08), int(width * 0.92), y_end)
    if rng.random() < 0.7:
        y = sp.heading(page, src, font, int(size * 1.3), x0, x1, y, "center")
    for _ in range(rng.randint(1, 2)):
        if y > y_end - lead * 4:
            break
        y = kv_block(page, src, font, size, lead, x0, x1, y, y_end)
        if rng.random() < 0.5 and y < y_end - lead * 3:
            y, _ok = sp.paragraph(page, src, font, size, x0, x1, y, y_end, lead, justify=False,
                                  max_lines=rng.randint(1, 3), align="center")
            y += lead // 2
    if rng.random() < 0.5 and y < y_end - height * 0.15:
        w = int(width * rng.uniform(0.15, 0.3))
        sp.picture(page, (width - w) // 2, y + size, (width + w) // 2, y + size + w)
        page.after_picture = False
    return page, rng


def dialogue(seed: int):
    """Play script: speaker name in a right column, ":" and the speech wrapped to its left; stage
    directions across the page. Each turn (name + speech) is one Text region."""
    rng = random.Random(seed)
    font = rng.choice(sp.usable_fonts())
    src = sp.WordSource(rng)
    width = rng.randint(900, 1400)
    height = int(width * rng.uniform(1.3, 1.6))
    page = sp.Page(width, height, rng)
    size = rng.randint(14, 24)
    lead = int(size * rng.uniform(1.4, 1.9))
    x0, x1 = int(width * rng.uniform(0.06, 0.1)), int(width * rng.uniform(0.9, 0.94))
    y = int(height * 0.05)
    y_end = height - int(height * 0.05)
    if rng.random() < 0.7:
        y_end = sp.footer_lines(page, src, font, max(10, int(size * 0.85)), x0, x1, y_end)
    if rng.random() < 0.4:
        y = sp.heading(page, src, font, size, x0, x1, y, "center")
    names = [src.next() for _ in range(rng.randint(2, 4))]
    name_w = int((x1 - x0) * rng.uniform(0.1, 0.18))
    while y < y_end - lead * 2:
        if rng.random() < 0.15:  # stage direction
            y, _ok = sp.paragraph(page, src, font, size, x0, x1, y, y_end, lead, max_lines=rng.randint(1, 3))
            y += lead // 3
            continue
        start = len(page.lines)
        name = sp.make_line(page, [rng.choice(names), ":"], font, size, name_w * 2, False)
        if name is None:
            break
        sp.put_line(page, name, x1, y + int(size * 0.35), sp.BODY, "right", name_w)
        y, _ok = sp.paragraph(page, src, font, size, x0, x1 - name_w - size // 2, y, y_end, lead,
                              justify=rng.random() < 0.5, max_lines=rng.randint(1, 4))
        _record(page, start, C["Text"])
        y += int(lead * rng.uniform(0.1, 0.5))
    return page, rng


def contents_columns(page, src, font, size, lead, x0, x1, y, y1) -> int:
    """Contents in columns: serial number | title | author, page number on the far left.
    Each row is one Contents-entry spanning the row."""
    rng = page.rng
    serial = rng.random() < 0.7
    author = rng.random() < 0.7
    number = sp.page_no(rng)
    title_w = int((x1 - x0) * rng.uniform(0.35, 0.5))
    row = 0
    while y + lead <= y1:
        row += 1
        number = min(999, number + rng.randint(2, 30))
        n = len(page.lines)
        right = x1
        top = y + int(size * 0.35)
        if serial:
            mark = sp.make_line(page, [sp.digits(row, rng, font) + rng.choice([".", "-", ")", ""])], font, size, size * 3, False)
            if mark is not None:
                sp.put_line(page, mark, right, top, sp.BODY, "right", size * 3)
            right -= size * 3
        title = sp.make_line(page, [src.next() for _ in range(rng.randint(1, 5))], font, size, title_w, False)
        if title is not None and title[1][2] - title[1][0] <= title_w:
            sp.put_line(page, title, right, top, sp.BODY, "right", title_w)
        right -= title_w + size
        if author:
            aw = int((right - x0 - size * 5) * 0.8)
            a = sp.make_line(page, [src.next() for _ in range(rng.randint(1, 3))], font, size, aw, False)
            if aw > size * 3 and a is not None and a[1][2] - a[1][0] <= aw:
                sp.put_line(page, a, right, top, sp.BODY, "right", aw)
        pg = sp.make_line(page, [sp.digits(number, rng, font)], font, size, size * 4, False)
        if pg is not None:
            sp.put_line(page, pg, x0 + size * 4, top, sp.BODY, "left", size * 4)
        quads = _lines_since(page, n)
        if quads:
            bx = _box(quads, 2)
            _add_region(page, x0 - 2, bx[1], x1 + 2, bx[3], C["Contents-entry"])
        y += lead
    return y


def make(seed: int):
    rng = random.Random(seed)
    kind = rng.choices(["mixed", "table", "contents", "cover", "imprint", "dialogue"],
                       [0.48, 0.17, 0.13, 0.08, 0.07, 0.07])[0]
    if kind == "mixed":
        _img, _labels = sp.make_page(seed)  # uses the wrapped drawing functions
        img, labels = _img, _labels
    else:
        page, prng = {"table": table_page, "contents": contents, "cover": cover,
                      "imprint": imprint, "dialogue": dialogue}[kind](seed)
        img, labels = sp.degrade(page, prng)
    boxes = []
    for quad, k in labels:
        if k >= REGION:
            q = np.array(quad, np.float32)
            boxes.append([round(float(q[:, 0].min()), 1), round(float(q[:, 1].min()), 1),
                          round(float(q[:, 0].max()), 1), round(float(q[:, 1].max()), 1), k - REGION])
    return img, boxes, kind


def write(task) -> str:
    install()
    index, split = task
    img, boxes, kind = make(index * 7919 + (17 if split == "train" else 999_331))
    h, w = img.shape[:2]
    name = f"{index:07d}.jpg"
    cv2.imwrite(str(OUT / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, random.Random(index).randint(55, 92)])
    return json.dumps({"image": name, "w": w, "h": h, "boxes": boxes, "kind": kind})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6000)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = parser.parse_args()
    for split, n in (("test", args.test), ("train", args.count)):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers) as pool, (OUT / split / "labels.jsonl").open("w", encoding="utf-8") as out:
            for done, line in enumerate(pool.map(write, [(i, split) for i in range(n)], chunksize=8)):
                out.write(line + "\n")
                if done % 500 == 0:
                    print(f"{split} {done}/{n}", flush=True)
        print(f"{split} done {n}", flush=True)


if __name__ == "__main__":
    main()
