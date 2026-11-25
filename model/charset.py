"""Character set for the CRNN. Index 0 is the CTC blank."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = ROOT / "character_dictionary.txt"

ARABIC_GROUPS = {"sindhi", "urdu", "diacritic"}


class Charset:
    def __init__(self, path: Path = DICTIONARY) -> None:
        chars = [""]
        groups: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("index\t"):
                continue
            index_s, group, _shown, code, _name, *_rest = line.split("\t")
            index = int(index_s)
            char = chr(int(code[2:], 16))
            if index != len(chars):
                raise ValueError(f"dictionary index {index} is out of order")
            chars.append(char)
            groups[char] = group
        self.chars = chars
        self.groups = groups
        self.index = {char: i for i, char in enumerate(chars) if char}
        self.size = len(chars)
        if self.size < 2:
            raise ValueError("character dictionary is empty")

    def script(self, char: str) -> str | None:
        group = self.groups.get(char)
        if group in ARABIC_GROUPS:
            return "sindhi"
        if group == "english":
            return "english"
        return None

    def encode(self, text: str) -> list[int] | None:
        ids = []
        for char in text:
            index = self.index.get(char)
            if index is None:
                return None
            ids.append(index)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[index] for index in ids if index)
