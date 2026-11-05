"""Re-aggregate Exact@k from EXP-012 predictions and compare to checkpoint metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED = {
    "exact_at_1": 0.3374832801728573,
    "exact_at_5": 0.5304043625887437,
    "exact_at_10": 0.6046918407243543,
    "num_samples": 9719,
}


def _mean_bool(series: pd.Series) -> float:
    return float(series.astype(bool).mean())


def reverify(
    predictions_csv: Path,
    evaluation_summary_csv: Path | None,
    finetune_metrics_json: Path | None,
    best_checkpoint_json: Path | None,
    *,
    atol: float = 1e-9,
) -> dict:
    df = pd.read_csv(predictions_csv)
    recomputed = {
        "num_samples": int(len(df)),
        "exact_at_1": _mean_bool(df["is_exact_at_1"]),
        "exact_at_5": _mean_bool(df["is_exact_at_5"]),
        "exact_at_10": _mean_bool(df["is_exact_at_10"]),
        "model_version_unique": sorted(
            {str(v) for v in df["model_version"].fillna("").unique()}
        ),
        "prediction_type_unique": sorted(
            {str(v) for v in df["prediction_type"].fillna("").unique()}
        ),
    }

    matches_official = (
        recomputed["num_samples"] == EXPECTED["num_samples"]
        and abs(recomputed["exact_at_1"] - EXPECTED["exact_at_1"]) <= atol
        and abs(recomputed["exact_at_5"] - EXPECTED["exact_at_5"]) <= atol
        and abs(recomputed["exact_at_10"] - EXPECTED["exact_at_10"]) <= atol
    )

    summary_overall = None
    if evaluation_summary_csv and evaluation_summary_csv.exists():
        summary = pd.read_csv(evaluation_summary_csv)
        overall = summary[
            ~summary["experiment_name"].astype(str).str.contains(r"\[", regex=True)
        ]
        if not overall.empty:
            row = overall.iloc[0]
            summary_overall = {
                "exact_at_1": float(row["exact_at_1"]),
                "exact_at_5": float(row["exact_at_5"]),
                "exact_at_10": float(row["exact_at_10"]),
                "num_samples": int(row["num_samples"]),
            }

    finetune = None
    if finetune_metrics_json and finetune_metrics_json.exists():
        finetune = json.loads(finetune_metrics_json.read_text(encoding="utf-8"))

    best_ckpt = None
    if best_checkpoint_json and best_checkpoint_json.exists():
        best_ckpt = json.loads(best_checkpoint_json.read_text(encoding="utf-8"))

    return {
        "recomputed_from_predictions": recomputed,
        "expected_official": EXPECTED,
        "matches_official_exact_k": matches_official,
        "evaluation_summary_overall": summary_overall,
        "finetune_metrics": finetune,
        "best_checkpoint": best_ckpt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("notebooks/best/exp012_sillok_ft_full_predictions.csv"),
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("notebooks/best/evaluation_summary.csv"),
    )
    parser.add_argument(
        "--finetune-metrics",
        type=Path,
        default=Path("notebooks/best/finetune_metrics.json"),
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=Path("notebooks/best/best_checkpoint.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/exp012_metrics_reverify.json"),
    )
    args = parser.parse_args()

    payload = reverify(
        args.predictions_csv,
        args.evaluation_summary,
        args.finetune_metrics,
        args.best_checkpoint,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["matches_official_exact_k"]:
        raise SystemExit("Exact@k re-aggregation does not match official EXP-012 figures")


if __name__ == "__main__":
    main()
