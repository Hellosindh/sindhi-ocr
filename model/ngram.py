"""Character n-grams. Scores are smoothed. A zero count is not a ban."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

from model.charset import Charset


class CharNGram:
    def __init__(self, order: int = 5, k: float = 0.01) -> None:
        if order < 2 or order > 6:
            raise ValueError("n-gram order must be from 2 to 6")
        self.order = order
        self.k = k
        self.counts: dict[str, dict[str, int]] = {"sindhi": {}, "english": {}}
        self.context_total: dict[str, dict[str, int]] = {"sindhi": {}, "english": {}}
        self.vocab = {"sindhi": 1, "english": 1}

    def fit_charset(self, charset: Charset) -> None:
        counts = {"sindhi": 0, "english": 0}
        for char in charset.chars:
            script = charset.script(char)
            if script is not None:
                counts[script] += 1
        self.vocab = {name: max(1, count) for name, count in counts.items()}

    def add_text(self, text: str, charset: Charset) -> None:
        run: list[str] = []
        run_script: str | None = None
        for char in text:
            script = charset.script(char)
            if script is None:
                if run_script is not None:
                    run.append(char)
                continue
            if run_script is None:
                run_script = script
            elif script != run_script:
                self._count_run("".join(run), run_script)
                run = []
                run_script = script
            run.append(char)
        if run_script is not None and run:
            self._count_run("".join(run), run_script)

    def _count_run(self, text: str, script: str) -> None:
        padded = "^" * (self.order - 1) + text
        counts = self.counts[script]
        totals = self.context_total[script]
        for end in range(self.order - 1, len(padded)):
            gram = padded[end - self.order + 1 : end + 1]
            context = gram[:-1]
            counts[gram] = counts.get(gram, 0) + 1
            totals[context] = totals.get(context, 0) + 1

    def prune(self, minimum: int = 2) -> int:
        """Drop n-grams seen fewer than `minimum` times. Once is not kept."""
        removed = 0
        for script, counts in self.counts.items():
            totals = self.context_total[script]
            for gram, count in list(counts.items()):
                if count < minimum:
                    context = gram[:-1]
                    totals[context] = totals.get(context, 0) - count
                    if totals.get(context, 0) <= 0:
                        totals.pop(context, None)
                    del counts[gram]
                    removed += 1
        return removed

    def score(self, text: str, charset: Charset) -> float:
        """Smoothed score of logical-order text. A missing n-gram is not a ban."""
        total = 0.0
        prefix = ""
        for char in text:
            total += self.delta(prefix, char, charset)
            prefix += char
        return total

    def delta(self, prefix: str, char: str, charset: Charset) -> float:
        script = charset.script(char)
        if script is None:
            return 0.0
        same = [item for item in prefix if charset.script(item) == script]
        context = ("^" * (self.order - 1) + "".join(same))[-(self.order - 1) :]
        gram = context + char
        count = self.counts[script].get(gram, 0)
        total = self.context_total[script].get(context, 0)
        vocab = self.vocab[script]
        return math.log((count + self.k) / (total + self.k * vocab))

    def save(self, path: Path) -> None:
        self.prune(minimum=2)
        for counts in self.counts.values():
            if any(count < 2 for count in counts.values()):
                raise ValueError("refusing to save an n-gram that appears only once")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "CharNGram":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(path)
        return model
