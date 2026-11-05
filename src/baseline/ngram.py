"""Character n-gram context baseline (EXP-003)."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

DEFAULT_NGRAMS = (8, 4, 2, 1)


class NgramContextBaseline:
    """Predict from left/right context keys seen in train split."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        ngram_sizes: tuple[int, ...] = DEFAULT_NGRAMS,
        context_window: int = 64,
    ) -> None:
        self.ngram_sizes = ngram_sizes
        self.context_window = context_window
        self._left_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        self._right_counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        self._unigram = Counter(train_df["label_char"].astype(str))

        for _, row in train_df.iterrows():
            label = str(row["label_char"])
            pos = int(row["mask_position"])
            text = str(row["original_text"])
            left = text[max(0, pos - context_window) : pos]
            right = text[pos + 1 : pos + 1 + context_window]
            for n in ngram_sizes:
                if len(left) >= n:
                    self._left_counts[(n, left[-n:])][label] += 1
                if len(right) >= n:
                    self._right_counts[(n, right[:n])][label] += 1

    def _score_candidates(self, row: pd.Series) -> Counter[str]:
        pos = int(row["mask_position"])
        text = str(row["original_text"])
        left = text[max(0, pos - self.context_window) : pos]
        right = text[pos + 1 : pos + 1 + self.context_window]

        scores: Counter[str] = Counter()
        for n in self.ngram_sizes:
            weight = float(n)
            if len(left) >= n:
                for ch, cnt in self._left_counts[(n, left[-n:])].items():
                    scores[ch] += cnt * weight
            if len(right) >= n:
                for ch, cnt in self._right_counts[(n, right[:n])].items():
                    scores[ch] += cnt * weight
        if not scores:
            scores.update(self._unigram)
        return scores

    def predict_topk(self, row: pd.Series, k: int = 10) -> list[str]:
        ranked = [ch for ch, _ in self._score_candidates(row).most_common()]
        if not ranked:
            ranked = [ch for ch, _ in self._unigram.most_common()]
        return ranked[:k]
