"""Compute Exact@k on full / overlap / no-overlap test subsets."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.audit_ngram_leakage import build_per_sample_overlap_flags
from src.eval.metrics import compute_exact_metrics

SUBSET_NAMES = ("all", "overlap", "no_overlap")


def _parse_top_chars(value: str) -> list[str]:
    if not value or pd.isna(value):
        return []
    return [ch for ch in str(value).split(";") if ch]


def _metrics_row(label: str, subset: str, metrics) -> dict:
    return {
        "model": label,
        "subset": subset,
        "num_samples": metrics.num_samples,
        "exact_at_1": metrics.exact_at_1,
        "exact_at_5": metrics.exact_at_5,
        "exact_at_10": metrics.exact_at_10,
    }


def evaluate_predictions_with_overlap(
    predictions_csv: Path,
    *,
    model_label: str,
    flags_df: pd.DataFrame,
    overlap_column: str = "has_train_2gram_match",
) -> list[dict]:
    preds = pd.read_csv(predictions_csv)
    if "sample_id" not in preds.columns:
        raise ValueError(f"predictions CSV missing sample_id: {predictions_csv}")
    merged = preds.merge(flags_df, on="sample_id", how="inner")
    if merged.empty:
        raise RuntimeError(f"No rows after joining {predictions_csv} with overlap flags")

    gold_key = "label_char" if "label_char" in merged.columns else "label_char_x"
    if gold_key not in merged.columns:
        raise ValueError("Merged frame missing label_char")

    results: list[dict] = []
    subsets = {
        "all": merged,
        "overlap": merged[merged[overlap_column] == True],  # noqa: E712
        "no_overlap": merged[merged[overlap_column] != True],  # noqa: E712
    }
    for subset_name, frame in subsets.items():
        gold_chars = [str(g) for g in frame[gold_key].tolist()]
        top_chars_list = [_parse_top_chars(row.get("top10_chars", "")) for _, row in frame.iterrows()]
        metrics = compute_exact_metrics(gold_chars, top_chars_list)
        results.append(_metrics_row(model_label, subset_name, metrics))
    return results


def render_markdown(
    rows: list[dict],
    *,
    overlap_column: str,
    flag_counts: dict[str, int],
) -> str:
    lines = [
        "# Overlap-adjusted evaluation (EXP-011)",
        "",
        f"- **생성일:** {date.today().isoformat()}",
        f"- **overlap 정의:** `{overlap_column}` (train과 좌/우 2-gram+ context key 공유)",
        f"- **test overlap:** {flag_counts['overlap']:,} / {flag_counts['total']:,} "
        f"({flag_counts['overlap'] / flag_counts['total'] * 100:.1f}%)",
        f"- **test no-overlap:** {flag_counts['no_overlap']:,}",
        "",
        "## Exact@k by subset",
        "",
        "| model | subset | n | Exact@1 | Exact@5 | Exact@10 |",
        "|-------|--------|--:|--------:|--------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['subset']} | {row['num_samples']:,} | "
            f"{row['exact_at_1'] * 100:.1f}% | {row['exact_at_5'] * 100:.1f}% | "
            f"{row['exact_at_10'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- **all:** 기존 headline Exact@k (EXP-003~007과 동일 split).",
            "- **overlap:** train과 2-gram+ 문맥을 공유하는 쉬운 케이스.",
            "- **no_overlap:** 공유 없음 — memorization·정형구 영향이 적은 subset.",
            "- 발표 시 **all과 no_overlap을 병기**한다 (감사 권고).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        metavar="LABEL=PATH",
        required=True,
        help="Model label and predictions CSV, e.g. ngram=outputs/predictions/exp004.csv",
    )
    parser.add_argument(
        "--flags-csv",
        type=Path,
        default=None,
        help="Precomputed flags CSV; if omitted, built from mask dataset.",
    )
    parser.add_argument(
        "--overlap-column",
        default="has_train_2gram_match",
        choices=[
            "has_train_2gram_match",
            "has_train_4gram_match",
            "has_train_ngram_match",
        ],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    if args.flags_csv and args.flags_csv.exists():
        flags_df = pd.read_csv(args.flags_csv)
    else:
        flags_df = build_per_sample_overlap_flags(args.mask_dataset, split="test")

    overlap_col = args.overlap_column
    flag_counts = {
        "total": len(flags_df),
        "overlap": int(flags_df[overlap_col].sum()),
        "no_overlap": int((~flags_df[overlap_col].astype(bool)).sum()),
    }

    all_rows: list[dict] = []
    for spec in args.predictions:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, got: {spec}")
        label, path_str = spec.split("=", 1)
        path = Path(path_str)
        all_rows.extend(
            evaluate_predictions_with_overlap(
                path,
                model_label=label,
                flags_df=flags_df,
                overlap_column=overlap_col,
            )
        )

    payload = {
        "overlap_column": overlap_col,
        "flag_counts": flag_counts,
        "results": all_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_markdown(all_rows, overlap_column=overlap_col, flag_counts=flag_counts),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
