"""Unit tests for Exact@k metrics."""

from __future__ import annotations

import unittest

from src.eval.metrics import compute_exact_metrics, rank_of_gold


class ExactMetricsTests(unittest.TestCase):
    def test_rank_of_gold(self) -> None:
        self.assertEqual(rank_of_gold(["宗", "法", "師"], "宗"), 1)
        self.assertEqual(rank_of_gold(["法", "宗", "師"], "宗"), 2)
        self.assertIsNone(rank_of_gold(["法", "師"], "宗"))

    def test_compute_exact_metrics(self) -> None:
        gold = ["A", "B", "C", "D"]
        tops = [
            ["A", "X"],
            ["X", "B"],
            ["Y", "Z", "C"],
            ["Q"],
        ]
        metrics = compute_exact_metrics(gold, tops)
        self.assertEqual(metrics.num_samples, 4)
        self.assertAlmostEqual(metrics.exact_at_1, 0.25)
        self.assertAlmostEqual(metrics.exact_at_5, 0.75)
        self.assertAlmostEqual(metrics.exact_at_10, 0.75)

    def test_empty_inputs(self) -> None:
        metrics = compute_exact_metrics([], [])
        self.assertEqual(metrics.num_samples, 0)
        self.assertEqual(metrics.exact_at_1, 0.0)


if __name__ == "__main__":
    unittest.main()
