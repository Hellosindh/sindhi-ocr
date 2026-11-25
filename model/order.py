"""Logical Sindhi order versus the left-to-right order CTC sees.

The network reads the line the way it is drawn. python-bidi's display
order is that drawing order. Turning a drawing back into typing order
is the inverse of that reorder: Arabic is reversed once, and each
English or number island is put back the way it appears on the page.
A second get_display() call is not that inverse. It treats the drawing
as a new paragraph and mirrors dates such as 2007-10-24.
"""

from __future__ import annotations

from unicodedata import bidirectional

from bidi.algorithm import get_display
from bidi.mirror import MIRRORED

from model.charset import Charset


def _has_sindhi(text: str, charset: Charset) -> bool:
    return any(charset.script(char) == "sindhi" for char in text)


def paragraph_base(text: str) -> str:
    """R when the first strong letter is Sindhi or Arabic, else L."""
    for char in text:
        kind = bidirectional(char)
        if kind in ("AL", "R"):
            return "R"
        if kind == "L":
            return "L"
    return "R"


def to_visual(text: str, charset: Charset) -> str:
    if not text or not _has_sindhi(text, charset):
        return text
    return get_display(text)


def _has_ltr(text: str) -> bool:
    return any(bidirectional(char) in ("L", "EN", "AN") for char in text)


def _reverse_digit_groups(island: str) -> str:
    """Turn 24-10-2007 back into 2007-10-24 without reversing digits inside a group."""
    out: list[str] = []
    group: list[str] = []

    def flush() -> None:
        if group:
            out.append("".join(reversed(group)))
            group.clear()

    for char in island:
        if bidirectional(char) in ("EN", "AN"):
            group.append(char)
        else:
            flush()
            out.append(char)
    flush()
    return "".join(out)


def _restore_island(island: str, arabic_before: bool) -> str:
    """The full line reverse leaves Latin and digits backwards.

    A hyphenated number after Sindhi is several runs, so only each group
    is backwards. A number before Sindhi, or an English phrase, is one
    run and the whole island is backwards.
    """
    has_letter = any(bidirectional(char) == "L" for char in island)
    if has_letter or not arabic_before:
        return island[::-1]
    return _reverse_digit_groups(island)


def _island_continues(chars: list[str], index: int) -> int | None:
    """Include spaces when another English letter or digit follows.

    Brackets stay outside the island. They are RTL punctuation here, and
    pulling them in glues a number to the next English word.
    """
    look = index
    while look < len(chars) and bidirectional(chars[look]) == "WS":
        look += 1
    if look < len(chars) and bidirectional(chars[look]) in ("L", "EN", "AN"):
        return look
    return None


def _invert_rtl(visual: str) -> str:
    """Inverse of an RTL-base display.

    Reverse the line once, which restores Sindhi. Then turn each English
    or number island forward again. Periods, colons, and quotes stay where
    that first reverse put them, so they are not pulled into the number.
    """
    chars = list(visual[::-1])
    out: list[str] = []
    index = 0
    seen_arabic = False
    while index < len(chars):
        kind = bidirectional(chars[index])
        if kind in ("L", "EN", "AN"):
            end = index + 1
            while end < len(chars):
                next_kind = bidirectional(chars[end])
                if next_kind in ("L", "EN", "AN", "NSM"):
                    end += 1
                    continue
                if next_kind in ("ES", "CS") and end + 1 < len(chars) and bidirectional(chars[end + 1]) in ("L", "EN", "AN", "ES", "CS"):
                    end += 1
                    continue
                continued = _island_continues(chars, end)
                if continued is None:
                    break
                end = continued
            out.append(_restore_island("".join(chars[index:end]), seen_arabic))
            index = end
            continue
        if kind in ("AL", "R"):
            seen_arabic = True
        out.append(MIRRORED[chars[index]] if chars[index] in MIRRORED else chars[index])
        index += 1
    return "".join(out)


def _invert_ltr(visual: str) -> str:
    """Inverse of an LTR-base display. Only the Arabic islands are reordered."""
    chars = list(visual)
    pieces: list[str] = []
    span: list[str] = []
    index = 0
    while index < len(chars):
        kind = bidirectional(chars[index])
        if kind in ("AL", "R") or (kind == "NSM" and span):
            span.append(chars[index])
            index += 1
            continue
        if span and kind in ("WS", "CS", "ON") and chars[index] not in MIRRORED:
            look = index
            while look < len(chars) and bidirectional(chars[look]) in ("WS", "CS", "ON") and chars[look] not in MIRRORED:
                look += 1
            if look < len(chars) and bidirectional(chars[look]) in ("AL", "R"):
                span.extend(chars[index:look])
                index = look
                continue
        if span:
            pieces.append(_invert_rtl("".join(span)))
            span = []
        pieces.append(chars[index])
        index += 1
    if span:
        pieces.append(_invert_rtl("".join(span)))
    return "".join(pieces)


def to_logical(text: str, charset: Charset, base_dir: str | None = None) -> str:
    """Typing order for a string the model read left to right.

    Pass base_dir from the reference line when scoring CER. Without it,
    an RTL reading is kept when it is a real preimage of this drawing,
    which is the Sindhi-base case. English-first drawings fall through
    to the LTR inverse so a leading English word is not moved.
    """
    if not text or not _has_sindhi(text, charset):
        return text
    if base_dir == "L":
        return _invert_ltr(text)
    if base_dir == "R" or not _has_ltr(text):
        return _invert_rtl(text)
    rtl = _invert_rtl(text)
    if get_display(rtl) == text:
        return rtl
    return _invert_ltr(text)
