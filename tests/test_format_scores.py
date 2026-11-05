"""Tests for prediction score formatting."""

from __future__ import annotations

import unittest

from src.eval.run_fill_mask_eval import _format_scores


class FormatScoresTests(unittest.TestCase):
    def test_empty(self) -> None:
        top1, s1, top5, s5, top10, s10 = _format_scores([])
        self.assertEqual(top1, "")
        self.assertEqual(s1, "")
        self.assertEqual(top5, "")
        self.assertEqual(s5, "")

    def test_with_scores(self) -> None:
        chars = ["甲", "乙", "丙"]
        scores = [0.5, 0.25, 0.1]
        top1, s1, top5, s5, top10, s10 = _format_scores(chars, scores)
        self.assertEqual(top1, "甲")
        self.assertEqual(s1, "0.500000")
        self.assertEqual(top5, "甲;乙;丙")
        self.assertEqual(s5, "0.500000;0.250000;0.100000")
        self.assertEqual(top10, "甲;乙;丙")
        self.assertIn("0.500000", s10)


if __name__ == "__main__":
    unittest.main()
