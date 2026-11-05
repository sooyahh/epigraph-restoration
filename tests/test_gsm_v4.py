"""Tests for gsm_preprocess_v4 defect/hangul symbol separation."""

from __future__ import annotations

import unittest

from src.preprocess.gsm_v4 import (
    NORMALIZED_DEFECT_SYMBOL,
    NORMALIZED_HANGUL_SYMBOL,
    preprocess_reading_text,
)


class GsmV4Tests(unittest.TestCase):
    def test_defect_and_hangul_use_distinct_symbols(self) -> None:
        result = preprocess_reading_text("大華嚴○宗師班씨詩")
        self.assertIn(NORMALIZED_DEFECT_SYMBOL, result.text_normalized)
        self.assertIn(NORMALIZED_HANGUL_SYMBOL, result.text_normalized)
        self.assertNotIn("○", result.text_normalized)
        self.assertEqual(result.defect_count, 1)
        self.assertEqual(result.hangul_replaced, 1)
        self.assertTrue(result.has_real_defect)
        self.assertFalse(result.is_training_text)
        self.assertEqual(result.quality_flag, "review")

    def test_hangul_only_is_not_real_defect(self) -> None:
        result = preprocess_reading_text("班씨詩")
        self.assertEqual(result.text_normalized, f"班{NORMALIZED_HANGUL_SYMBOL}詩")
        self.assertEqual(result.defect_count, 0)
        self.assertFalse(result.has_real_defect)
        self.assertEqual(result.quality_flag, "review")
        self.assertIn("[hangul->◇]", result.defect_markers_raw)

    def test_clean_text_stays_training_eligible(self) -> None:
        result = preprocess_reading_text("嘉慶二十五年庚辰四月日")
        self.assertTrue(result.is_training_text)
        self.assertEqual(result.defect_count, 0)
        self.assertEqual(result.hangul_replaced, 0)


if __name__ == "__main__":
    unittest.main()
