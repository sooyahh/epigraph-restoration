"""Appendix C constrained Top-K search for ARI (EXP-019).

GPU-free core: classify / expand / score / order. Model forward is supplied
by the caller as a next-token logprob function.

`completed` is a running set across steps. Resetting it each step drops
single-token hanja found at step 1 while incomplete byte prefixes keep the
loop going, which would omit Fig. 12 first-class candidates from Exact@5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from src.nrich.text_utils import is_cjk_ideograph, normalize_unicode

COMPLETE_HANJA = "COMPLETE_HANJA"
INCOMPLETE = "INCOMPLETE"
INVALID = "INVALID"

REPLACEMENT_CHAR = "�"

NextLogprobsFn = Callable[[Sequence[int]], Mapping[int, float] | Sequence[float]]
DecodeFn = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class HanjaCandidate:
    char: str
    score: float
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]


@dataclass(frozen=True)
class _Hypothesis:
    prefix_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]


def json_value_prefix_d1(marker: str = "[D1]") -> str:
    """Teacher-forced JSON value prefix matching `json.dumps` SFT targets."""
    dumped = json.dumps({marker: ""}, ensure_ascii=False)
    if not dumped.endswith('"}'):
        raise ValueError(
            f"unexpected json.dumps shape for marker {marker!r}: {dumped!r}"
        )
    return dumped[:-2]


def classify_decoded_span(s: str) -> str:
    """Classify a decoded candidate span (JSON prefix excluded).

    INCOMPLETE: empty, invalid UTF-8, or replacement-char only.
    COMPLETE_HANJA: one CJK ideograph after NFC.
    INVALID: one non-hanja char, or two-or-more chars.
    """
    if s is None:
        return INCOMPLETE
    if REPLACEMENT_CHAR in s and s.replace(REPLACEMENT_CHAR, "") == "":
        return INCOMPLETE
    text = normalize_unicode(s, "NFC")
    if not text:
        return INCOMPLETE
    if len(text) == 1 and is_cjk_ideograph(text):
        return COMPLETE_HANJA
    return INVALID


def score_mean_logprob(logprobs: Sequence[float]) -> float:
    if not logprobs:
        raise ValueError("score_mean_logprob requires at least one token logprob")
    return sum(logprobs) / len(logprobs)


def _keep_better(
    current: HanjaCandidate | None, incoming: HanjaCandidate
) -> HanjaCandidate:
    if current is None:
        return incoming
    if incoming.score != current.score:
        return incoming if incoming.score > current.score else current
    if len(incoming.token_ids) != len(current.token_ids):
        return incoming if len(incoming.token_ids) < len(current.token_ids) else current
    return current


def dedupe_keep_best(
    candidates: Sequence[HanjaCandidate],
) -> dict[str, HanjaCandidate]:
    """Keep the highest-score path per character (OURS)."""
    best: dict[str, HanjaCandidate] = {}
    for cand in candidates:
        best[cand.char] = _keep_better(best.get(cand.char), cand)
    return best


def order_candidates(
    candidates: Sequence[HanjaCandidate], k: int
) -> list[HanjaCandidate]:
    unique = list(dedupe_keep_best(candidates).values())
    unique.sort(key=lambda c: (-c.score, len(c.token_ids), c.char))
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return unique[:k]


def _as_logprob_map(raw: Mapping[int, float] | Sequence[float]) -> dict[int, float]:
    if isinstance(raw, Mapping):
        return {int(tok): float(lp) for tok, lp in raw.items()}
    return {i: float(lp) for i, lp in enumerate(raw)}


def _top_tokens(
    logprobs: Mapping[int, float], take: int
) -> list[tuple[int, float]]:
    ranked = sorted(logprobs.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:take]


def expand_character_candidates(
    prefix_ids: Sequence[int],
    next_logprobs: NextLogprobsFn,
    decode_tokens: DecodeFn,
    *,
    k: int = 5,
    b1: int = 32,
    b2: int = 8,
    lmax: int = 4,
) -> list[HanjaCandidate]:
    """Constrained Top-K search: first-token branch, then complete characters.

    Merges every COMPLETE_HANJA finish into a running `completed` set. Do not
    reset that set per step; Fig. 12 ranks 1-token hanja alongside later
    byte-completed hanja.
    """
    if k <= 0:
        return []
    if lmax <= 0:
        return []

    frontier = [
        _Hypothesis(prefix_ids=tuple(prefix_ids), token_ids=(), logprobs=())
    ]
    completed: dict[str, HanjaCandidate] = {}

    for step in range(1, lmax + 1):
        take = b1 if step == 1 else b2
        new_frontier: list[_Hypothesis] = []
        for hyp in frontier:
            ranked = _top_tokens(_as_logprob_map(next_logprobs(hyp.prefix_ids)), take)
            for token_id, token_lp in ranked:
                if not math.isfinite(token_lp):
                    continue
                token_ids = hyp.token_ids + (token_id,)
                logprobs = hyp.logprobs + (token_lp,)
                decoded = decode_tokens(token_ids)
                status = classify_decoded_span(decoded)
                if status == COMPLETE_HANJA:
                    char = normalize_unicode(decoded, "NFC")
                    cand = HanjaCandidate(
                        char=char,
                        score=score_mean_logprob(logprobs),
                        token_ids=token_ids,
                        logprobs=logprobs,
                    )
                    completed[char] = _keep_better(completed.get(char), cand)
                elif status == INCOMPLETE:
                    new_frontier.append(
                        _Hypothesis(
                            prefix_ids=hyp.prefix_ids + (token_id,),
                            token_ids=token_ids,
                            logprobs=logprobs,
                        )
                    )
        frontier = new_frontier
        if len(completed) >= k and not frontier:
            break

    return order_candidates(list(completed.values()), k)
