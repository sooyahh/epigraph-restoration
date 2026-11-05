"""Corpus unigram frequency baseline (EXP-002)."""

from __future__ import annotations

from collections import Counter

import pandas as pd


class FrequencyBaseline:
    """Predict top-k chars from train-split label_char frequencies."""

    def __init__(self, train_df: pd.DataFrame) -> None:
        counts = Counter(train_df["label_char"].astype(str))
        self._ranked = [ch for ch, _ in counts.most_common()]

    def predict_topk(self, _row: pd.Series, k: int = 10) -> list[str]:
        return self._ranked[:k]
