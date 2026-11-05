"""Unit tests for eval --run-mode safety."""

from __future__ import annotations

import unittest

from src.eval.run_fill_mask_eval import EvalConfigError, resolve_eval_run_config


class EvalRunModeTests(unittest.TestCase):
    def test_full_rejects_positive_max_samples(self) -> None:
        with self.assertRaises(EvalConfigError):
            resolve_eval_run_config("full", max_samples=2000)

    def test_full_accepts_none_or_zero(self) -> None:
        for value in (None, 0):
            cfg = resolve_eval_run_config("full", max_samples=value)
            self.assertEqual(cfg["run_mode"], "full")
            self.assertIsNone(cfg["max_samples"])
            self.assertFalse(cfg["subsample"])

    def test_smoke_requires_positive_max_samples(self) -> None:
        with self.assertRaises(EvalConfigError):
            resolve_eval_run_config("smoke", max_samples=None)
        with self.assertRaises(EvalConfigError):
            resolve_eval_run_config("smoke", max_samples=0)
        cfg = resolve_eval_run_config("smoke", max_samples=32)
        self.assertEqual(cfg["max_samples"], 32)
        self.assertTrue(cfg["subsample"])

    def test_require_model_path(self) -> None:
        with self.assertRaises(EvalConfigError):
            resolve_eval_run_config(
                "full",
                max_samples=None,
                model_path="",
                require_model_path=True,
            )
        cfg = resolve_eval_run_config(
            "full",
            max_samples=None,
            model_path="outputs/finetune/sillokbert_exp012_gpu/best",
            require_model_path=True,
        )
        self.assertEqual(cfg["run_mode"], "full")


if __name__ == "__main__":
    unittest.main()
