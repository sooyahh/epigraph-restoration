"""Exact@k metrics for single-character restoration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExactMetrics:
    num_samples: int
    exact_at_1: float
    exact_at_5: float
    exact_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "num_samples": self.num_samples,
            "exact_at_1": self.exact_at_1,
            "exact_at_5": self.exact_at_5,
            "exact_at_10": self.exact_at_10,
        }


def rank_of_gold(top_chars: list[str], gold: str) -> int | None:
    for idx, ch in enumerate(top_chars, start=1):
        if ch == gold:
            return idx
    return None


def compute_exact_metrics(gold_chars: list[str], top_chars_list: list[list[str]]) -> ExactMetrics:
    if len(gold_chars) != len(top_chars_list):
        raise ValueError("gold_chars and top_chars_list length mismatch")
    n = len(gold_chars)
    if n == 0:
        return ExactMetrics(0, 0.0, 0.0, 0.0)

    hit1 = hit5 = hit10 = 0
    for gold, tops in zip(gold_chars, top_chars_list):
        rank = rank_of_gold(tops, gold)
        if rank is not None and rank <= 1:
            hit1 += 1
        if rank is not None and rank <= 5:
            hit5 += 1
        if rank is not None and rank <= 10:
            hit10 += 1

    return ExactMetrics(
        num_samples=n,
        exact_at_1=hit1 / n,
        exact_at_5=hit5 / n,
        exact_at_10=hit10 / n,
    )
