"""Run fill-mask evaluation on artificial mask dataset."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.baseline.frequency import FrequencyBaseline
from src.baseline.ngram import NgramContextBaseline
from src.eval.mask_window import build_masked_window
from src.eval.metrics import compute_exact_metrics, rank_of_gold
from src.eval.predictors import BaselinePredictor, FillMaskPredictor, Predictor

COMMITTED_SUMMARY_DIR = Path("reports/evaluation")

_MODEL_SLUG = {
    "frequency": "freq",
    "ngram": "ngram",
    "bert-base-chinese": "bert",
    "ddokbaro/SillokBert": "sillokbert",
}


def committed_summary_path(
    experiment_id: str,
    model_kind: str,
    split: str = "test",
) -> Path:
    """Return the version-controlled summary path for an experiment."""
    exp_slug = experiment_id.lower().replace("-", "")
    model_slug = _MODEL_SLUG.get(
        model_kind,
        model_kind.replace("/", "_").replace("-", ""),
    )
    return COMMITTED_SUMMARY_DIR / f"{exp_slug}_{model_slug}_{split}_summary.csv"


PREDICTION_COLUMNS = [
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

SUMMARY_COLUMNS = [
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
]


class EvalConfigError(ValueError):
    """Invalid evaluation --run-mode / --max-samples combination."""


def resolve_eval_run_config(
    run_mode: str,
    max_samples: int | None,
    *,
    split: str = "test",
    model_path: str | None = None,
    require_model_path: bool = False,
) -> dict:
    """Validate smoke/full eval caps. Raises EvalConfigError on bad combos."""
    mode = (run_mode or "").strip().lower()
    if mode not in {"smoke", "full"}:
        raise EvalConfigError("--run-mode must be 'smoke' or 'full'")

    if require_model_path and not (model_path and str(model_path).strip()):
        raise EvalConfigError(
            "--require-model-path was set but --model-path is empty. "
            "For planned EXP-012 full test, pass the best checkpoint path explicitly."
        )

    if mode == "full":
        if max_samples is not None and max_samples > 0:
            raise EvalConfigError(
                "full eval mode forbids --max-samples > 0 "
                f"(got {max_samples}). Omit --max-samples for the full split."
            )
        return {
            "run_mode": "full",
            "max_samples": None,
            "split": split,
            "subsample": False,
        }

    if max_samples is None or max_samples <= 0:
        raise EvalConfigError(
            "smoke eval mode requires --max-samples > 0 "
            "(got {!r}). Use --run-mode full for the entire split.".format(max_samples)
        )
    return {
        "run_mode": "smoke",
        "max_samples": max_samples,
        "split": split,
        "subsample": True,
    }


def _select_rows(
    df: pd.DataFrame,
    split: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    subset = df[df["split"] == split].copy()
    if max_samples is not None and len(subset) > max_samples:
        subset = subset.sample(n=max_samples, random_state=seed)
    return subset.reset_index(drop=True)


def _build_predictor(
    model_kind: str,
    train_df: pd.DataFrame,
    model_path: str | None,
    batch_size: int,
) -> Predictor:
    if model_kind == "frequency":
        return BaselinePredictor(FrequencyBaseline(train_df))
    if model_kind == "ngram":
        return BaselinePredictor(NgramContextBaseline(train_df))
    model_name = model_path or model_kind
    return FillMaskPredictor(model_name, batch_size=batch_size)


def _format_scores(
    chars: list[str],
    scores: list[float] | None = None,
) -> tuple[str, str, str, str, str, str]:
    top1 = chars[0] if chars else ""
    top5 = chars[:5]
    top10 = chars[:10]
    score_list = scores or []
    top1_score = f"{score_list[0]:.6f}" if score_list else ""
    top5_scores = ";".join(f"{s:.6f}" for s in score_list[:5]) if score_list else ""
    top10_scores = ";".join(f"{s:.6f}" for s in score_list[:10]) if score_list else ""
    return (
        top1,
        top1_score,
        ";".join(top5),
        top5_scores,
        ";".join(top10),
        top10_scores,
    )


def _input_text_for_row(row: pd.Series, mask_token: str = "[MASK]") -> str:
    built = build_masked_window(
        str(row["original_text"]),
        int(row["mask_position"]),
        str(row["label_char"]),
        mask_token,
    )
    if built is None:
        return str(row.get("masked_input", ""))
    return built[0]


def run_evaluation(
    mask_csv: Path,
    experiment_id: str,
    experiment_name: str,
    model_kind: str,
    *,
    split: str = "test",
    model_path: str | None = None,
    model_version: str = "",
    max_samples: int | None = None,
    seed: int = 42,
    batch_size: int = 8,
    predictions_csv: Path = Path("outputs/predictions/model_predictions.csv"),
    summary_csv: Path = Path("outputs/evaluation/evaluation_summary.csv"),
    append_outputs: bool = True,
    notes: str = "",
) -> dict:
    df = pd.read_csv(mask_csv)
    train_df = df[df["split"] == "train"]
    eval_df = _select_rows(df, split, max_samples, seed)
    if eval_df.empty:
        raise RuntimeError(f"No rows for split={split}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = _build_predictor(model_kind, train_df, model_path, batch_size)
    created_at = datetime.now(timezone.utc).isoformat()

    prediction_rows: list[dict] = []
    gold_chars: list[str] = []
    top_chars_list: list[list[str]] = []

    use_batch = isinstance(predictor, FillMaskPredictor)
    rows = [row for _, row in eval_df.iterrows()]
    total = len(rows)

    if use_batch:
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            batch_preds = predictor.predict_batch_scored(batch, k=10)
            if start == 0 or (start // batch_size) % 25 == 0:
                print(f"  eval progress: {min(start + batch_size, total)}/{total}", flush=True)
            for row, (tops, scores) in zip(batch, batch_preds):
                gold = str(row["label_char"])
                gold_chars.append(gold)
                top_chars_list.append(tops)
                rank = rank_of_gold(tops, gold)
                top1, top1_score, top5, top5_scores, top10, top10_scores = _format_scores(
                    tops, scores
                )
                prediction_rows.append(
                    {
                        "prediction_id": f"pred_{experiment_id.lower()}_{row['sample_id']}",
                        "experiment_id": experiment_id,
                        "model_name": model_path or model_kind,
                        "model_version": model_version,
                        "sample_id": row["sample_id"],
                        "case_id": "",
                        "input_text": _input_text_for_row(row),
                        "label_char": gold,
                        "top1_char": top1,
                        "top1_score": top1_score,
                        "top5_chars": top5,
                        "top5_scores": top5_scores,
                        "top10_chars": top10,
                        "top10_scores": top10_scores,
                        "rank_of_gold": rank if rank is not None else "",
                        "is_exact_at_1": rank is not None and rank <= 1,
                        "is_exact_at_5": rank is not None and rank <= 5,
                        "is_exact_at_10": rank is not None and rank <= 10,
                        "prediction_type": "artificial_mask",
                        "created_at": created_at,
                        "notes": notes,
                    }
                )
    else:
        for row in rows:
            tops = predictor.predict_topk(row, k=10)
            gold = str(row["label_char"])
            gold_chars.append(gold)
            top_chars_list.append(tops)
            rank = rank_of_gold(tops, gold)
            top1, top1_score, top5, top5_scores, top10, top10_scores = _format_scores(tops)
            prediction_rows.append(
                {
                    "prediction_id": f"pred_{experiment_id.lower()}_{row['sample_id']}",
                    "experiment_id": experiment_id,
                    "model_name": model_path or model_kind,
                    "model_version": model_version,
                    "sample_id": row["sample_id"],
                    "case_id": "",
                    "input_text": _input_text_for_row(row),
                    "label_char": gold,
                    "top1_char": top1,
                    "top1_score": top1_score,
                    "top5_chars": top5,
                    "top5_scores": top5_scores,
                    "top10_chars": top10,
                    "top10_scores": top10_scores,
                    "rank_of_gold": rank if rank is not None else "",
                    "is_exact_at_1": rank is not None and rank <= 1,
                    "is_exact_at_5": rank is not None and rank <= 5,
                    "is_exact_at_10": rank is not None and rank <= 10,
                    "prediction_type": "artificial_mask",
                    "created_at": created_at,
                    "notes": notes,
                }
            )

    overall = compute_exact_metrics(gold_chars, top_chars_list)
    summary_rows: list[dict] = [
        {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "model_name": model_path or model_kind,
            "dataset_name": "artificial_mask_dataset_v1",
            "split": split,
            "num_samples": overall.num_samples,
            "exact_at_1": overall.exact_at_1,
            "exact_at_5": overall.exact_at_5,
            "exact_at_10": overall.exact_at_10,
            "group_at_1": "",
            "group_at_5": "",
            "group_at_10": "",
            "created_at": created_at,
            "notes": f"device={device}; {notes}".strip("; "),
        }
    ]

    era_metrics: dict[str, dict] = {}
    for era, era_df in eval_df.groupby(eval_df["era"].fillna("unknown")):
        era_samples = era_df["sample_id"].tolist()
        era_gold: list[str] = []
        era_tops: list[list[str]] = []
        sample_to_idx = {row["sample_id"]: i for i, row in enumerate(prediction_rows)}
        for sample_id in era_samples:
            idx = sample_to_idx[sample_id]
            era_gold.append(gold_chars[idx])
            era_tops.append(top_chars_list[idx])
        metrics = compute_exact_metrics(era_gold, era_tops)
        era_metrics[str(era)] = metrics.as_dict()
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "experiment_name": f"{experiment_name} [{era}]",
                "model_name": model_path or model_kind,
                "dataset_name": "artificial_mask_dataset_v1",
                "split": split,
                "num_samples": metrics.num_samples,
                "exact_at_1": metrics.exact_at_1,
                "exact_at_5": metrics.exact_at_5,
                "exact_at_10": metrics.exact_at_10,
                "group_at_1": "",
                "group_at_5": "",
                "group_at_10": "",
                "created_at": created_at,
                "notes": f"era={era}; device={device}; {notes}".strip("; "),
            }
        )

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    _write_csv_rows(predictions_csv, PREDICTION_COLUMNS, prediction_rows, append_outputs)
    _write_csv_rows(summary_csv, SUMMARY_COLUMNS, summary_rows, append_outputs)

    payload = {
        "experiment_id": experiment_id,
        "device": device,
        "num_samples": overall.num_samples,
        "max_samples_requested": max_samples,
        "seed": seed,
        "overall": overall.as_dict(),
        "era_breakdown": era_metrics,
        "predictions_csv": str(predictions_csv),
        "summary_csv": str(summary_csv),
    }
    return payload


def _write_csv_rows(
    path: Path,
    columns: list[str],
    rows: list[dict],
    append: bool,
) -> None:
    write_header = not path.exists() or not append
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-mode",
        choices=["smoke", "full"],
        required=True,
        help="Required. smoke needs --max-samples > 0; full forbids sample caps.",
    )
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--model-kind",
        choices=["frequency", "ngram", "bert-base-chinese", "ddokbaro/SillokBert"],
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Explicit checkpoint/model dir. For planned EXP-012 full test, "
        "pass outputs/finetune/sillokbert_exp012_gpu/best (or winning checkpoint-epoch-*).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/predictions/model_predictions.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/evaluation/evaluation_summary.csv"),
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--committed-summary",
        action="store_true",
        help="Write summary CSV to reports/evaluation/ (version-controlled).",
    )
    parser.add_argument(
        "--require-model-path",
        action="store_true",
        help="Fail if --model-path is empty (use for planned EXP-012 best-checkpoint test).",
    )
    parser.add_argument(
        "--model-version",
        default="",
        help=(
            "Checkpoint/version tag written to predictions CSV. "
            "For EXP-012 official best-by-valid-loss use "
            "exp012_sillokbert_gpu_best_valid_loss."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        eval_config = resolve_eval_run_config(
            args.run_mode,
            args.max_samples,
            split=args.split,
            model_path=args.model_path or None,
            require_model_path=args.require_model_path,
        )
    except EvalConfigError as exc:
        parser.error(str(exc))

    summary_csv = args.summary_csv
    if args.committed_summary:
        summary_csv = committed_summary_path(
            args.experiment_id,
            args.model_kind,
            split=args.split,
        )

    notes = args.notes
    mode_note = (
        f"run_mode={eval_config['run_mode']}; "
        f"max_samples={eval_config['max_samples']}; "
        f"split={args.split}"
    )
    notes = f"{mode_note}; {notes}".strip("; ")

    print(
        f"[eval run_mode={eval_config['run_mode']}] "
        f"split={args.split} max_samples={eval_config['max_samples']} "
        f"model_path={args.model_path or '(hub/default)'}",
        flush=True,
    )

    payload = run_evaluation(
        args.mask_dataset,
        args.experiment_id,
        args.experiment_name,
        args.model_kind,
        split=args.split,
        model_path=args.model_path or None,
        model_version=args.model_version,
        max_samples=eval_config["max_samples"],
        seed=args.seed,
        batch_size=args.batch_size,
        predictions_csv=args.predictions_csv,
        summary_csv=summary_csv,
        append_outputs=not args.overwrite,
        notes=notes,
    )
    payload["run_mode"] = eval_config["run_mode"]
    payload["official_performance"] = (
        eval_config["run_mode"] == "full" and args.split == "test"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
