"""Normalize Sindhi heh forms to canonical Unicode (see heh_rules.md)."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

HEH = "\u0647"  # ه ARABIC LETTER HEH
DOACHASHMEE = "\u06BE"  # ھ HEH DOACHASHMEE
GOAL = "\u06C1"  # ہ HEH GOAL
HEH_FORMS = frozenset({HEH, DOACHASHMEE, GOAL})

JEEM_GAF = frozenset("جگ")
ALIF_FAMILY = frozenset("اآأإٱ")

# Split keeping whitespace/punctuation runs so spacing is preserved.
_TOKEN = re.compile(r"\S+|\s+", re.UNICODE)


def _is_mark(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def _is_letter(char: str) -> bool:
    if char in HEH_FORMS:
        return True
    category = unicodedata.category(char)
    if category != "Lo":
        return False
    code = ord(char)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    )


def _prev_letter(chars: list[str], index: int) -> str | None:
    for pos in range(index - 1, -1, -1):
        if _is_mark(chars[pos]):
            continue
        if _is_letter(chars[pos]):
            return chars[pos]
        # punctuation / digit / other — stop; no previous letter in this cluster
        return None
    return None


def _is_word_final_heh(chars: list[str], index: int) -> bool:
    for pos in range(index + 1, len(chars)):
        if _is_mark(chars[pos]):
            continue
        if _is_letter(chars[pos]):
            return False
        # trailing punctuation after heh still counts as word-final heh
        return True
    return True


def normalize_heh_word(word: str) -> str:
    if not word or not any(char in HEH_FORMS for char in word):
        return word
    chars = list(word)
    out: list[str] = []
    for index, char in enumerate(chars):
        if char not in HEH_FORMS:
            out.append(char)
            continue
        prev = _prev_letter(chars, index)
        final = _is_word_final_heh(chars, index)
        if prev in JEEM_GAF:
            out.append(HEH)
        elif final and prev in ALIF_FAMILY:
            out.append(DOACHASHMEE)
        elif final:
            out.append(GOAL)
        else:
            out.append(DOACHASHMEE)
    return "".join(out)


def normalize_heh(text: str) -> str:
    """Apply heh rules to every non-whitespace token."""
    if not text or not any(char in HEH_FORMS for char in text):
        return text
    return "".join(normalize_heh_word(part) if part.strip() else part for part in _TOKEN.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Sindhi heh forms")
    parser.add_argument("text", nargs="?", help="text to normalize")
    parser.add_argument("--file", type=Path, help="normalize each line of a UTF-8 file to stdout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        cases = [
            ("نه", "نہ"),
            ("ته", "تہ"),
            ("به", "بہ"),
            ("نہ", "نہ"),
            ("جهنگ", "جهنگ"),
            ("گهر", "گهر"),
            ("سمجه", "سمجه"),
            ("سگه", "سگه"),
            ("واه", "واھ"),
            ("واھ", "واھ"),
            ("شاه", "شاھ"),
            ("ھٿ", "ھٿ"),
            ("سھڻو", "سھڻو"),
            ("ڪجھه", "ڪجهہ"),
            ("ڪجهہ", "ڪجهہ"),
            ("ھميشه", "ھميشہ"),
            ("ڳاله", "ڳالہ"),
        ]
        failed = 0
        for source, want in cases:
            got = normalize_heh(source)
            ok = got == want
            line = f"{'ok' if ok else 'FAIL'}\t{source} -> {got} (want {want})"
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("unicode_escape").decode("ascii"))
            failed += int(not ok)
        raise SystemExit(failed)
    if args.file:
        for line in args.file.read_text(encoding="utf-8").splitlines():
            print(normalize_heh(line))
        return
    if args.text is None:
        parser.error("pass text, --file, or --self-test")
    print(normalize_heh(args.text))


if __name__ == "__main__":
    main()


def no_heh(text: str) -> str:
    """All heh forms as one letter: for comparing a label with a read when the heh form does not matter."""
    return "".join("ه" if ch in HEH_FORMS else ch for ch in text)
