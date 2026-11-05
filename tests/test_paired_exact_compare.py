"""Unit tests for paired Exact@k McNemar comparison."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.paired_exact_compare import (
    chi2_sf_1df,
    mcnemar_chi2_continuity,
    paired_exact_compare,
)


class McNemarUnitTests(unittest.TestCase):
    def test_mcnemar_zero_discordant(self) -> None:
        self.assertEqual(mcnemar_chi2_continuity(0, 0), 0.0)
        self.assertAlmostEqual(chi2_sf_1df(0.0), 1.0)

    def test_mcnemar_known_values(self) -> None:
        # b=10, c=2 → (|10-2|-1)^2 / 12 = 49/12
        self.assertAlmostEqual(mcnemar_chi2_continuity(10, 2), 49 / 12)

    def test_paired_exact_compare_toy(self) -> None:
        # 8 samples: both+ 3, both- 2, only_a 1, only_b 2
        rows = [
            (1, True, True),
            (2, True, True),
            (3, True, True),
            (4, False, False),
            (5, False, False),
            (6, True, False),
            (7, False, True),
            (8, False, True),
        ]
        pred_a = pd.DataFrame(
            {"sample_id": [r[0] for r in rows], "is_exact_at_5": [r[1] for r in rows]}
        )
        pred_b = pd.DataFrame(
            {"sample_id": [r[0] for r in rows], "is_exact_at_5": [r[2] for r in rows]}
        )
        out = paired_exact_compare(
            pred_a, pred_b, k=5, label_a="base", label_b="cand", n_boot=500, seed=0
        )
        self.assertEqual(out["n_paired"], 8)
        self.assertEqual(out["both_correct"], 3)
        self.assertEqual(out["both_incorrect"], 2)
        self.assertEqual(out["only_a_correct"], 1)
        self.assertEqual(out["only_b_correct"], 2)
        self.assertAlmostEqual(out["exact_at_5_a"], 4 / 8)
        self.assertAlmostEqual(out["exact_at_5_b"], 5 / 8)
        self.assertAlmostEqual(out["diff_b_minus_a"], 1 / 8)
        self.assertAlmostEqual(out["mcnemar"]["chi2"], mcnemar_chi2_continuity(1, 2))
        self.assertTrue(out["mcnemar"]["continuity_correction"])

    def test_requires_overlap(self) -> None:
        a = pd.DataFrame({"sample_id": [1], "is_exact_at_5": [True]})
        b = pd.DataFrame({"sample_id": [2], "is_exact_at_5": [True]})
        with self.assertRaises(ValueError):
            paired_exact_compare(a, b, k=5, n_boot=10)


if __name__ == "__main__":
    unittest.main()
