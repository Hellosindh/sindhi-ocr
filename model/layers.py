"""Measure each language layer on a real dev set. Do not guess the weight."""

from __future__ import annotations

from pathlib import Path

from model.charset import Charset

WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
DATA = Path(__file__).resolve().parent / "data"


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        previous, rows[0] = rows[0], ref_index
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            insert_delete = min(rows[hyp_index] + 1, rows[hyp_index - 1] + 1)
            replace = previous + (ref_item != hyp_item)
            previous, rows[hyp_index] = rows[hyp_index], min(insert_delete, replace)
    return rows[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return edit_distance(ref_words, hypothesis.split()) / len(ref_words)


def keep_visual(visual_text: str, visual_score: float, lm_text: str, lm_visual_score: float, clear_margin: float) -> str:
    """A clear image beats the language model, even if the model prefers a common word."""
    if visual_text == lm_text:
        return visual_text
    if visual_score - lm_visual_score > clear_margin:
        return visual_text
    return lm_text


def tune_weight(dev_dir: Path, scores: dict[float, float]) -> float:
    """`scores` maps an LM weight to character error on the real dev set.

    The directory must exist and contain labeled real lines. Without it, the
    weight stays 0.
    """
    transcript = dev_dir / "transcript.tsv"
    if not dev_dir.is_dir() or not transcript.exists() or transcript.stat().st_size == 0:
        return 0.0
    if not scores:
        return 0.0
    return min(scores, key=scores.get)


def write_report(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["layer\tcer\tgain_vs_recognizer"]
    base = rows[0][1] if rows else 0.0
    for name, score in rows:
        lines.append(f"{name}\t{score:.6f}\t{base - score:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_weight(path: Path = DATA / "lm_weight.txt") -> float:
    if not path.exists():
        return 0.0
    return float(path.read_text(encoding="utf-8").strip())
