"""Tests for evaluation summary aggregation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_eval_summaries import aggregate_summaries


class AggregateEvalSummariesTests(unittest.TestCase):
    def test_aggregate_overall_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for exp_id, model in [("EXP003", "frequency"), ("EXP004", "ngram")]:
                path = root / f"{exp_id.lower()}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "experiment_id",
                            "experiment_name",
                            "model_name",
                            "dataset_name",
                            "split",
                            "num_samples",
                            "exact_at_1",
                            "exact_at_5",
                            "exact_at_10",
                            "group_at_1",
                            "group_at_5",
                            "group_at_10",
                            "created_at",
                            "notes",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "experiment_id": exp_id,
                            "experiment_name": f"{model} baseline",
                            "model_name": model,
                            "dataset_name": "artificial_mask_dataset_v1",
                            "split": "test",
                            "num_samples": "10",
                            "exact_at_1": "0.1",
                            "exact_at_5": "0.2",
                            "exact_at_10": "0.3",
                            "group_at_1": "",
                            "group_at_5": "",
                            "group_at_10": "",
                            "created_at": "2026-07-06",
                            "notes": "",
                        }
                    )
                    writer.writerow(
                        {
                            "experiment_id": exp_id,
                            "experiment_name": f"{model} baseline [조선]",
                            "model_name": model,
                            "dataset_name": "artificial_mask_dataset_v1",
                            "split": "test",
                            "num_samples": "5",
                            "exact_at_1": "0.0",
                            "exact_at_5": "0.0",
                            "exact_at_10": "0.0",
                            "group_at_1": "",
                            "group_at_5": "",
                            "group_at_10": "",
                            "created_at": "2026-07-06",
                            "notes": "era=조선",
                        }
                    )
            out = root / "merged.csv"
            count = aggregate_summaries(
                [root / "exp003.csv", root / "exp004.csv"],
                out,
            )
            self.assertEqual(count, 2)
            with out.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("[" not in row["experiment_name"] for row in rows))


if __name__ == "__main__":
    unittest.main()
