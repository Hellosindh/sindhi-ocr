"""CTC greedy decode, and a beam that can add a smoothed n-gram score."""

from __future__ import annotations

import math

import torch

from model.charset import Charset
from model.ngram import CharNGram

NEG = -1e30


def _logsumexp(left: float, right: float) -> float:
    if left <= NEG:
        return right
    if right <= NEG:
        return left
    base = max(left, right)
    return base + math.log(math.exp(left - base) + math.exp(right - base))


def greedy(log_probs: torch.Tensor, charset: Charset) -> str:
    chosen = log_probs.argmax(dim=-1).tolist()
    ids: list[int] = []
    previous = 0
    for label in chosen:
        if label != 0 and label != previous:
            ids.append(label)
        previous = label
    return charset.decode(ids)


def _add(store, prefix, blank_score, char_score, visual_blank, visual_char):
    old = store.get(prefix, (NEG, NEG, NEG, NEG))
    store[prefix] = (
        _logsumexp(old[0], blank_score),
        _logsumexp(old[1], char_score),
        _logsumexp(old[2], visual_blank),
        _logsumexp(old[3], visual_char),
    )


def beam(
    log_probs: torch.Tensor,
    charset: Charset,
    width: int = 8,
    ngram: CharNGram | None = None,
    ngram_weight: float = 0.0,
    clear_margin: float = 1.0,
) -> tuple[str, float]:
    """CTC beam. The n-gram only breaks ties. A clearly better image wins."""
    blank = 0
    beams: dict[tuple[int, ...], tuple[float, float, float, float]] = {(): (0.0, NEG, 0.0, NEG)}
    for frame in log_probs.detach().cpu().tolist():
        combined_top = sorted(beams.items(), key=lambda item: _logsumexp(item[1][0], item[1][1]), reverse=True)[:width]
        visual_top = sorted(beams.items(), key=lambda item: _logsumexp(item[1][2], item[1][3]), reverse=True)[:width]
        ranked = {prefix: scores for prefix, scores in combined_top + visual_top}
        nxt: dict[tuple[int, ...], tuple[float, float, float, float]] = {}
        for prefix, (blank_score, char_score, visual_blank, visual_char) in ranked.items():
            combined = _logsumexp(blank_score, char_score)
            visual = _logsumexp(visual_blank, visual_char)
            for label, score in enumerate(frame):
                if label == blank:
                    _add(nxt, prefix, score + combined, NEG, score + visual, NEG)
                    continue
                same = bool(prefix) and prefix[-1] == label
                if same:
                    _add(nxt, prefix, NEG, score + char_score, NEG, score + visual_char)
                    source = score + blank_score
                    visual_source = score + visual_blank
                else:
                    source = score + combined
                    visual_source = score + visual
                extended = prefix + (label,)
                _add(nxt, extended, NEG, source, NEG, visual_source)
        beams = nxt
    def combined_key(item: tuple) -> float:
        prefix, scores = item
        ctc = _logsumexp(scores[0], scores[1])
        if ngram is None or ngram_weight == 0.0:
            return ctc
        from model.order import to_logical

        logical = to_logical(charset.decode(list(prefix)), charset)
        return ctc + ngram_weight * ngram.score(logical, charset)

    ordered = sorted(beams.items(), key=combined_key, reverse=True)
    visual_ordered = sorted(beams.items(), key=lambda item: _logsumexp(item[1][2], item[1][3]), reverse=True)
    chosen = ordered[0]
    visual_best = visual_ordered[0]
    chosen_visual = _logsumexp(chosen[1][2], chosen[1][3])
    best_visual = _logsumexp(visual_best[1][2], visual_best[1][3])
    if best_visual - chosen_visual > clear_margin:
        chosen = visual_best
    second = _logsumexp(ordered[1][1][0], ordered[1][1][1]) if len(ordered) > 1 else NEG
    margin = _logsumexp(chosen[1][0], chosen[1][1]) - second
    return charset.decode(list(chosen[0])), margin
