"""Tests for real defect inference dataset generation."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_real_defect_inference import (
    _case_row,
    _map_defect_position,
    _strip_hangul_markers,
)
from src.preprocess.gsm_v4 import NORMALIZED_DEFECT_SYMBOL, NORMALIZED_HANGUL_SYMBOL


class BuildRealDefectInferenceTests(unittest.TestCase):
    def test_strip_hangul_markers(self) -> None:
        text = f"大華嚴{NORMALIZED_HANGUL_SYMBOL}宗師"
        self.assertEqual(_strip_hangul_markers(text), "大華嚴宗師")

    def test_map_defect_position_skips_hangul_before_defect(self) -> None:
        text = f"AB{NORMALIZED_HANGUL_SYMBOL}C{NORMALIZED_DEFECT_SYMBOL}D"
        pos = text.index(NORMALIZED_DEFECT_SYMBOL)
        self.assertEqual(_map_defect_position(text, pos), 3)

    def test_case_row_strips_hangul_from_model_fields(self) -> None:
        text = f"[{NORMALIZED_HANGUL_SYMBOL}{NORMALIZED_HANGUL_SYMBOL}]郡守鄭候□碑"
        pos = text.index(NORMALIZED_DEFECT_SYMBOL)
        reading = pd.Series(
            {
                "reading_id": "nrich_test",
                "source": "nrich",
                "source_record_id": "1",
                "inscription_id": "insc_test",
                "artifact_id": "",
                "title": "test",
                "era": "조선",
                "text_normalized": text,
                "defect_markers_raw": "□",
            }
        )
        case = _case_row(reading, pos, 1)
        self.assertTrue(case["has_hangul_context"])
        self.assertNotIn(NORMALIZED_HANGUL_SYMBOL, case["model_input"])
        self.assertNotIn(NORMALIZED_HANGUL_SYMBOL, case["left_context"])
        self.assertNotIn(NORMALIZED_HANGUL_SYMBOL, case["right_context"])
        self.assertIn(NORMALIZED_HANGUL_SYMBOL, case["text_with_defect"])
        self.assertIn("[MASK]", case["model_input"])

    def test_case_row_without_hangul_context(self) -> None:
        text = "大華嚴□宗師"
        pos = text.index(NORMALIZED_DEFECT_SYMBOL)
        reading = pd.Series(
            {
                "reading_id": "nrich_test",
                "source": "nrich",
                "source_record_id": "1",
                "inscription_id": "insc_test",
                "artifact_id": "",
                "title": "test",
                "era": "조선",
                "text_normalized": text,
                "defect_markers_raw": "□",
            }
        )
        case = _case_row(reading, pos, 2)
        self.assertFalse(case["has_hangul_context"])
        self.assertEqual(case["model_input"], "大華嚴[MASK]宗師")


if __name__ == "__main__":
    unittest.main()
