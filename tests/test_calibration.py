"""Unit tests for calibration metrics and calibrators."""

from __future__ import annotations

import unittest

import numpy as np

from src.eval.calibration import (
    HistogramCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    summarize,
)


class CalibrationMetricTests(unittest.TestCase):
    def test_perfectly_calibrated_bins_have_zero_ece(self) -> None:
        prob = [0.25] * 4 + [0.75] * 4
        hit = [1, 0, 0, 0, 1, 1, 1, 0]
        self.assertAlmostEqual(expected_calibration_error(prob, hit, n_bins=4), 0.0)

    def test_overconfident_scores_have_positive_ece(self) -> None:
        prob = [0.9] * 10
        hit = [1] * 5 + [0] * 5
        self.assertAlmostEqual(expected_calibration_error(prob, hit, n_bins=10), 0.4)

    def test_bins_skip_empty_and_count_all(self) -> None:
        prob = [0.05, 0.06, 0.95]
        hit = [0, 0, 1]
        bins = reliability_bins(prob, hit, n_bins=10)
        self.assertEqual(sum(b["n"] for b in bins), 3)
        self.assertEqual(len(bins), 2)

    def test_quantile_bins_count_all(self) -> None:
        rng = np.random.default_rng(0)
        prob = rng.uniform(size=200)
        hit = rng.uniform(size=200) < prob
        bins = reliability_bins(prob, hit, n_bins=5, strategy="quantile")
        self.assertEqual(sum(b["n"] for b in bins), 200)

    def test_brier(self) -> None:
        self.assertAlmostEqual(brier_score([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(brier_score([0.5, 0.5], [1, 0]), 0.25)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            brier_score([0.1, 0.2], [1])

    def test_summarize_keys(self) -> None:
        out = summarize([0.2, 0.8], [0, 1], n_bins=2)
        self.assertEqual(out["n"], 2)
        self.assertIn("ece_uniform", out)


class CalibratorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        true_p = rng.uniform(0.05, 0.95, size=4000)
        # Scores are systematically too low: the model under-reports confidence.
        self.score = true_p ** 2
        self.hit = (rng.uniform(size=true_p.size) < true_p).astype(int)

    def test_platt_reduces_ece(self) -> None:
        before = expected_calibration_error(self.score, self.hit)
        cal = PlattCalibrator().fit(self.score, self.hit)
        after = expected_calibration_error(cal.predict(self.score), self.hit)
        self.assertLess(after, before)

    def test_isotonic_is_monotone_and_reduces_ece(self) -> None:
        cal = IsotonicCalibrator().fit(self.score, self.hit)
        grid = np.linspace(0, 1, 101)
        out = cal.predict(grid)
        self.assertTrue(np.all(np.diff(out) >= -1e-12))
        before = expected_calibration_error(self.score, self.hit)
        after = expected_calibration_error(cal.predict(self.score), self.hit)
        self.assertLess(after, before)

    def test_histogram_values_in_unit_interval(self) -> None:
        cal = HistogramCalibrator(n_bins=10).fit(self.score, self.hit)
        out = cal.predict(np.linspace(0, 1, 50))
        self.assertTrue(np.all((out >= 0) & (out <= 1)))

    def test_predict_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            IsotonicCalibrator().predict([0.5])


if __name__ == "__main__":
    unittest.main()
