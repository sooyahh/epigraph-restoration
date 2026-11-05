"""Unit tests for post-hoc Exact@5 checkpoint selection and eval-chain gates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.run_ft_eval_chain import evaluate_gate
from src.eval.select_checkpoint_by_exact import (
    SelectConfigError,
    list_epoch_checkpoints,
    load_valid_loss_map,
    select_best_by_exact5,
)


class SelectExact5Tests(unittest.TestCase):
    def test_max_exact5_wins(self) -> None:
        records = [
            {"epoch": 1, "exact_at_5": 0.50, "valid_loss": 1.0},
            {"epoch": 2, "exact_at_5": 0.55, "valid_loss": 1.2},
            {"epoch": 3, "exact_at_5": 0.52, "valid_loss": 0.9},
        ]
        best = select_best_by_exact5(records)
        self.assertEqual(best["epoch"], 2)

    def test_tie_breaks_min_valid_loss(self) -> None:
        records = [
            {"epoch": 1, "exact_at_5": 0.55, "valid_loss": 2.0},
            {"epoch": 2, "exact_at_5": 0.55, "valid_loss": 1.5},
            {"epoch": 3, "exact_at_5": 0.55, "valid_loss": 1.8},
        ]
        best = select_best_by_exact5(records)
        self.assertEqual(best["epoch"], 2)

    def test_tie_breaks_lower_epoch_when_loss_equal(self) -> None:
        records = [
            {"epoch": 1, "exact_at_5": 0.55, "valid_loss": 1.5},
            {"epoch": 2, "exact_at_5": 0.55, "valid_loss": 1.5},
        ]
        best = select_best_by_exact5(records)
        self.assertEqual(best["epoch"], 1)

    def test_missing_exact5_raises(self) -> None:
        with self.assertRaises(SelectConfigError):
            select_best_by_exact5([{"epoch": 1, "exact_at_5": None, "valid_loss": 1.0}])

    def test_empty_raises(self) -> None:
        with self.assertRaises(SelectConfigError):
            select_best_by_exact5([])

    def test_list_epoch_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoint-epoch-2").mkdir()
            (root / "checkpoint-epoch-1").mkdir()
            (root / "best").mkdir()
            (root / "notes.txt").write_text("x", encoding="utf-8")
            found = list_epoch_checkpoints(root)
            self.assertEqual([e for e, _ in found], [1, 2])
            with self.assertRaises(SelectConfigError):
                list_epoch_checkpoints(root / "missing")

    def test_load_valid_loss_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "epoch_metrics.json").write_text(
                json.dumps(
                    [
                        {"epoch": 1, "valid_loss": 3.5},
                        {"epoch": 2, "valid_loss": 2.1},
                    ]
                ),
                encoding="utf-8",
            )
            mapping = load_valid_loss_map(root)
            self.assertEqual(mapping[1], 3.5)
            self.assertEqual(mapping[2], 2.1)
            self.assertEqual(load_valid_loss_map(root / "nope"), {})


class EvalChainGateTests(unittest.TestCase):
    def test_subsample_gate_requires_delta_and_p(self) -> None:
        # EXP-015 = 0.5324; +0.7pp => need >= 0.5394
        ok = evaluate_gate(
            candidate_exact5=0.5400,
            mcnemar_p=0.01,
            stage="subsample",
        )
        self.assertTrue(ok["allow_full_valid"])
        self.assertTrue(ok["delta_ok"])
        self.assertTrue(ok["p_ok"])

        no_p = evaluate_gate(
            candidate_exact5=0.5400,
            mcnemar_p=None,
            stage="subsample",
        )
        self.assertFalse(no_p["allow_full_valid"])

        small = evaluate_gate(
            candidate_exact5=0.5350,
            mcnemar_p=0.01,
            stage="subsample",
        )
        self.assertFalse(small["allow_full_valid"])

    def test_full_valid_adoption_gate(self) -> None:
        # +1.0pp => need >= 0.5424
        adopt = evaluate_gate(
            candidate_exact5=0.5450,
            mcnemar_p=0.001,
            stage="full_valid",
        )
        self.assertTrue(adopt["adoption_candidate"])
        reject = evaluate_gate(
            candidate_exact5=0.5380,
            mcnemar_p=0.001,
            stage="full_valid",
        )
        self.assertFalse(reject["adoption_candidate"])


class SelectCliSplitGuardTests(unittest.TestCase):
    def test_cli_refuses_test_split(self) -> None:
        from src.eval.select_checkpoint_by_exact import main

        with self.assertRaises(SystemExit):
            main(
                [
                    "--run-dir",
                    "outputs/finetune/exp018/018a",
                    "--run-mode",
                    "smoke",
                    "--max-samples",
                    "10",
                    "--split",
                    "test",
                ]
            )


if __name__ == "__main__":
    unittest.main()
