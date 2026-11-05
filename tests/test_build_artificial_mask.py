"""Tests for artificial mask dataset generation."""

from __future__ import annotations

import random
import unittest

import pandas as pd

from src.data.build_artificial_mask import _masks_for_reading, validate_no_label_leak


class ArtificialMaskTests(unittest.TestCase):
    def test_label_not_leaked_at_mask_position(self) -> None:
        row = pd.Series(
            {
                "reading_id": "nrich_ksm_000005",
                "source": "nrich",
                "source_record_id": "5",
                "inscription_id": "insc_nrich_000005",
                "artifact_id": "",
                "title": "이소한신도비",
                "era": "조선",
                "text_normalized": "延安李公昭漢神道碑有明朝鮮國",
            }
        )
        rng = random.Random(0)
        samples = _masks_for_reading(row, "train", rng, max_masks_per_reading=3)
        self.assertGreater(len(samples), 0)
        df = pd.DataFrame(samples)
        validate_no_label_leak(df)
        for sample in samples:
            self.assertEqual(sample["mask_type"], "single_char")
            self.assertEqual(len(sample["label_char"]), 1)


if __name__ == "__main__":
    unittest.main()
