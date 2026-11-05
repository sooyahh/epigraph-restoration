"""Unit tests for expert panel selection constraints."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.select_expert_panel import select_artificial, select_real_defect


class ExpertPanelSelectionTests(unittest.TestCase):
    def test_artificial_respects_inscription_and_gap(self) -> None:
        preds = pd.DataFrame(
            {
                "sample_id": [f"m{i}" for i in range(12)],
                "label_char": ["甲"] * 12,
                "top1_char": ["甲"] * 6 + ["乙"] * 6,
                "top1_score": [0.9, 0.8, 0.1, 0.15, 0.5, 0.55, 0.92, 0.12, 0.4, 0.45, 0.7, 0.2],
                "top5_chars": ["甲;乙"] * 12,
                "top5_scores": ["0.5;0.1"] * 12,
                "top10_chars": ["甲;乙"] * 12,
                "top10_scores": ["0.5;0.1"] * 12,
                "rank_of_gold": [1] * 6 + [5] * 6,
                "is_exact_at_1": [True] * 6 + [False] * 6,
                "is_exact_at_5": [True] * 12,
                "is_exact_at_10": [True] * 12,
            }
        )
        meta = pd.DataFrame(
            {
                "sample_id": [f"m{i}" for i in range(12)],
                "inscription_id": ["A"] * 6 + ["B"] * 3 + ["C"] * 3,
                "artifact_id": ["a"] * 12,
                "title": ["t"] * 12,
                "era": ["조선"] * 8 + ["고려"] * 4,
                "mask_position": [10, 20, 30, 200, 210, 400, 10, 200, 400, 10, 200, 400],
                "original_text": ["x"] * 12,
                "masked_input": ["x[MASK]y"] * 12,
            }
        )
        out = select_artificial(
            preds, meta, n=6, max_per_inscription=2, min_position_gap=80, seed=0
        )
        self.assertLessEqual(len(out), 6)
        counts = out["inscription_id"].value_counts()
        self.assertTrue((counts <= 2).all())
        for _, group in out.groupby("inscription_id"):
            positions = sorted(int(p) for p in group["mask_position"])
            for i in range(1, len(positions)):
                self.assertGreaterEqual(positions[i] - positions[i - 1], 80)

    def test_real_defect_score_buckets(self) -> None:
        preds = pd.DataFrame(
            {
                "case_id": [f"r{i}" for i in range(9)],
                "top1_char": ["甲"] * 9,
                "top1_score": [0.05, 0.1, 0.15, 0.4, 0.45, 0.5, 0.9, 0.92, 0.95],
                "top5_chars": ["甲"] * 9,
                "top5_scores": ["0.1"] * 9,
                "top10_chars": ["甲"] * 9,
                "top10_scores": ["0.1"] * 9,
                "input_text": ["a[MASK]b"] * 9,
            }
        )
        cases = pd.DataFrame(
            {
                "case_id": [f"r{i}" for i in range(9)],
                "inscription_id": [f"I{i}" for i in range(9)],
                "artifact_id": ["a"] * 9,
                "title": ["t"] * 9,
                "era": ["조선"] * 5 + ["고려"] * 4,
                "defect_position": [10 * i for i in range(9)],
                "model_input": ["a[MASK]b"] * 9,
                "text_with_defect": ["a□b"] * 9,
            }
        )
        out = select_real_defect(
            preds, cases, n=6, max_per_inscription=1, min_position_gap=5, seed=1
        )
        self.assertEqual(len(out), 6)
        self.assertTrue((out.groupby("inscription_id").size() <= 1).all())
        self.assertGreaterEqual(out["panel_bucket"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
