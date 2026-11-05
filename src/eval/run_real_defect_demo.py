"""Run demo inference on real defect cases (no gold labels)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.eval.mask_window import CONTEXT_WINDOW, MAX_SEQ_LEN
from src.eval.predictors import FillMaskPredictor

OUTPUT_COLUMNS = [
    "prediction_id",
    "experiment_id",
    "model_name",
    "model_version",
    "sample_id",
    "case_id",
    "input_text",
    "label_char",
    "top1_char",
    "top1_score",
    "top5_chars",
    "top5_scores",
    "top10_chars",
    "top10_scores",
    "rank_of_gold",
    "is_exact_at_1",
    "is_exact_at_5",
    "is_exact_at_10",
    "prediction_type",
    "created_at",
    "notes",
]


def _window_model_input(model_input: str, mask_token: str) -> str | None:
    pos = model_input.find(mask_token)
    if pos < 0:
        return None
    left = max(0, pos - CONTEXT_WINDOW)
    right = min(len(model_input), pos + len(mask_token) + CONTEXT_WINDOW)
    window = model_input[left:right]
    if len(window) > MAX_SEQ_LEN:
        return None
    return window


def run_demo(
    cases_csv: Path,
    model_path: str,
    output_csv: Path,
    *,
    experiment_id: str = "EXP-DEMO",
    max_cases: int = 15,
    seed: int = 42,
    batch_size: int = 4,
    notes: str = "",
    demo_clean_only: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(cases_csv)
    if demo_clean_only and "has_hangul_context" in df.columns:
        df = df[df["has_hangul_context"] != True].copy()  # noqa: E712
    if len(df) > max_cases:
        df = df.sample(n=max_cases, random_state=seed).sort_values("case_id")
    df = df.reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = FillMaskPredictor(model_path, batch_size=batch_size)
    created_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []

    case_rows = [row for _, row in df.iterrows()]
    for start in range(0, len(case_rows), batch_size):
        batch = case_rows[start : start + batch_size]
        texts = [
            _window_model_input(str(row["model_input"]), predictor.mask_token) or ""
            for row in batch
        ]
        valid_texts = [text for text in texts if text]
        batch_preds = (
            predictor._predict_batch_model(valid_texts, k=10)  # noqa: SLF001
            if valid_texts
            else []
        )
        pred_iter = iter(batch_preds)
        for row, text in zip(batch, texts):
            tops = next(pred_iter) if text else []
            rows.append(
                {
                    "prediction_id": f"pred_demo_{row['case_id']}",
                    "experiment_id": experiment_id,
                    "model_name": model_path,
                    "model_version": "",
                    "sample_id": "",
                    "case_id": row["case_id"],
                    "input_text": text or str(row["model_input"]),
                    "label_char": "",
                    "top1_char": tops[0] if tops else "",
                    "top1_score": "",
                    "top5_chars": ";".join(tops[:5]),
                    "top5_scores": "",
                    "top10_chars": ";".join(tops[:10]),
                    "top10_scores": "",
                    "rank_of_gold": "",
                    "is_exact_at_1": "",
                    "is_exact_at_5": "",
                    "is_exact_at_10": "",
                    "prediction_type": "real_defect",
                    "created_at": created_at,
                    "notes": (
                        f"device={device}; demo_only; clean_only={demo_clean_only}; {notes}"
                    ).strip("; "),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-csv",
        type=Path,
        default=Path("data/model_inputs/real_defect_inference.csv"),
    )
    parser.add_argument(
        "--model-path",
        default="outputs/finetune/sillokbert_gsm_baseline/final",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/predictions/real_defect_demo.csv"),
    )
    parser.add_argument(
        "--experiment-id",
        default="DEMO001",
        help="Experiment id recorded in demo output rows.",
    )
    parser.add_argument("--max-cases", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--include-hangul-context",
        action="store_true",
        help="Include cases where original text had hangul ◇ markers (default: clean only).",
    )
    parser.add_argument("--notes", default="CPU demo subset")
    args = parser.parse_args()

    out = run_demo(
        args.cases_csv,
        args.model_path,
        args.output_csv,
        experiment_id=args.experiment_id,
        max_cases=args.max_cases,
        seed=args.seed,
        batch_size=args.batch_size,
        notes=args.notes,
        demo_clean_only=not args.include_hangul_context,
    )
    print(f"Wrote {len(out)} demo predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
