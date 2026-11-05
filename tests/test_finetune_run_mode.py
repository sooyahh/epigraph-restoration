"""Unit tests for finetune --run-mode safety and checkpoint selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.finetune_sillokbert import (
    RunConfigError,
    assert_safe_output_dir,
    get_git_commit,
    parse_resume_epoch,
    resolve_resume_start,
    resolve_run_config,
    select_best_checkpoint,
)


class FinetuneRunModeTests(unittest.TestCase):
    def test_full_rejects_positive_max_train(self) -> None:
        with self.assertRaises(RunConfigError):
            resolve_run_config("full", max_train=128, max_eval=None)

    def test_full_rejects_positive_max_eval(self) -> None:
        with self.assertRaises(RunConfigError):
            resolve_run_config("full", max_train=None, max_eval=64)

    def test_full_accepts_omit_or_zero(self) -> None:
        for max_train, max_eval in ((None, None), (0, 0), (None, 0), (0, None)):
            cfg = resolve_run_config("full", max_train=max_train, max_eval=max_eval)
            self.assertEqual(cfg["run_mode"], "full")
            self.assertIsNone(cfg["max_train"])
            self.assertIsNone(cfg["max_eval"])
            self.assertFalse(cfg["allow_subsample"])
            self.assertEqual(cfg["selection_metric"], "valid_loss")

    def test_smoke_defaults_and_rejects_zero(self) -> None:
        cfg = resolve_run_config("smoke", max_train=None, max_eval=None)
        self.assertEqual(cfg["max_train"], 32)
        self.assertEqual(cfg["max_eval"], 16)
        self.assertTrue(cfg["allow_subsample"])
        with self.assertRaises(RunConfigError):
            resolve_run_config("smoke", max_train=0, max_eval=16)
        with self.assertRaises(RunConfigError):
            resolve_run_config("smoke", max_train=32, max_eval=0)

    def test_invalid_mode(self) -> None:
        with self.assertRaises(RunConfigError):
            resolve_run_config("turbo", max_train=None, max_eval=None)

    def test_select_best_by_min_valid_loss(self) -> None:
        records = [
            {"epoch": 1, "valid_loss": 3.5, "checkpoint_dir": "e1"},
            {"epoch": 2, "valid_loss": 2.1, "checkpoint_dir": "e2"},
            {"epoch": 3, "valid_loss": 2.4, "checkpoint_dir": "e3"},
        ]
        best = select_best_checkpoint(records)
        self.assertEqual(best["epoch"], 2)
        self.assertEqual(best["checkpoint_dir"], "e2")

    def test_select_best_tie_breaks_lower_epoch(self) -> None:
        records = [
            {"epoch": 1, "valid_loss": 2.0, "checkpoint_dir": "e1"},
            {"epoch": 2, "valid_loss": 2.0, "checkpoint_dir": "e2"},
        ]
        best = select_best_checkpoint(records)
        self.assertEqual(best["epoch"], 1)

    def test_select_best_rejects_missing_valid_loss(self) -> None:
        with self.assertRaises(RunConfigError):
            select_best_checkpoint(
                [{"epoch": 1, "valid_loss": None, "checkpoint_dir": "e1"}]
            )

    def test_parse_resume_epoch(self) -> None:
        self.assertEqual(parse_resume_epoch(Path("checkpoint-epoch-2")), 2)
        with self.assertRaises(RunConfigError):
            parse_resume_epoch(Path("best"))
        with self.assertRaises(RunConfigError):
            parse_resume_epoch(Path("checkpoint-epoch-0"))

    def test_resolve_resume_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "checkpoint-epoch-1"
            ckpt.mkdir()
            info = resolve_resume_start(ckpt, num_epochs=2)
            self.assertEqual(info["completed_epoch"], 1)
            self.assertEqual(info["start_epoch_idx"], 1)
            with self.assertRaises(RunConfigError):
                resolve_resume_start(ckpt, num_epochs=1)
            missing = Path(tmp) / "checkpoint-epoch-9"
            with self.assertRaises(RunConfigError):
                resolve_resume_start(missing, num_epochs=2)
        none_info = resolve_resume_start(None, num_epochs=2)
        self.assertEqual(none_info["start_epoch_idx"], 0)

    def test_path_guard_refuses_notebooks_best(self) -> None:
        with self.assertRaises(RunConfigError):
            assert_safe_output_dir(Path("notebooks/best"), planned_experiment_id="EXP-018")
        with self.assertRaises(RunConfigError):
            assert_safe_output_dir(
                Path("notebooks/best/subdir"), planned_experiment_id="EXP-018"
            )

    def test_path_guard_refuses_exp012_and_exp011(self) -> None:
        with self.assertRaises(RunConfigError):
            assert_safe_output_dir(
                Path("outputs/finetune/sillokbert_exp012_gpu"),
                planned_experiment_id="EXP-018",
            )
        with self.assertRaises(RunConfigError):
            assert_safe_output_dir(
                Path("outputs/finetune/sillokbert_gsm_baseline"),
                planned_experiment_id="EXP-018",
            )

    def test_path_guard_allows_exp018(self) -> None:
        resolved = assert_safe_output_dir(
            Path("outputs/finetune/exp018/018a"),
            planned_experiment_id="EXP-018A",
        )
        self.assertTrue(str(resolved).endswith("outputs/finetune/exp018/018a") or
                        "exp018/018a" in str(resolved))

    def test_get_git_commit_returns_string(self) -> None:
        commit = get_git_commit()
        self.assertIsInstance(commit, str)
        self.assertTrue(len(commit) >= 7)


if __name__ == "__main__":
    unittest.main()
