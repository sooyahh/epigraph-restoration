"""Stratified Exact@k analysis for baseline predictions (Phase 2 style)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from src.eval.metrics import compute_exact_metrics, rank_of_gold

FREQ_BUCKETS: list[tuple[str, callable]] = [
    ("very_common (100+)", lambda c: c >= 100),
    ("common (21-100)", lambda c: 21 <= c <= 100),
    ("mid (6-20)", lambda c: 6 <= c <= 20),
    ("rare (2-5)", lambda c: 2 <= c <= 5),
    ("hapax (count=1)", lambda c: c == 1),
    ("unseen_in_train", lambda c: c == 0),
]

LEN_BUCKETS: list[tuple[str, callable]] = [
    ("short (<50)", lambda n: n < 50),
    ("medium (50-199)", lambda n: 50 <= n <= 199),
    ("long (200-999)", lambda n: 200 <= n <= 999),
    ("very_long (1000-4999)", lambda n: 1000 <= n <= 4999),
    ("ultra_long (5000+)", lambda n: n >= 5000),
]


def _parse_top_chars(row: pd.Series, k: int = 10) -> list[str]:
    raw = str(row.get("top10_chars") or row.get(f"top{k}_chars") or "")
    if not raw.strip():
        top1 = str(row.get("top1_char") or "").strip()
        return [top1] if top1 else []
    chars = [part.strip() for part in raw.split(";") if part.strip()]
    return chars[:k]


def _train_label_counts(mask_df: pd.DataFrame) -> Counter[str]:
    train = mask_df[mask_df["split"] == "train"]
    return Counter(train["label_char"].astype(str))


def _freq_bucket(count: int) -> str:
    for name, pred in FREQ_BUCKETS:
        if pred(count):
            return name
    return "unseen_in_train"


def _len_bucket(length: int) -> str:
    for name, pred in LEN_BUCKETS:
        if pred(length):
            return name
    return "ultra_long (5000+)"


def _metrics_for_rows(rows: pd.DataFrame) -> dict[str, float | int]:
    if rows.empty:
        return {"num_samples": 0, "exact_at_1": 0.0, "exact_at_5": 0.0, "exact_at_10": 0.0}
    gold = rows["label_char"].astype(str).tolist()
    tops = [_parse_top_chars(row) for _, row in rows.iterrows()]
    return compute_exact_metrics(gold, tops).as_dict()


def build_stratified_report(
    predictions_csv: Path,
    mask_csv: Path,
    *,
    model_key: str,
    split: str = "test",
) -> dict:
    preds = pd.read_csv(predictions_csv)
    mask_df = pd.read_csv(mask_csv)
    test_df = mask_df[mask_df["split"] == split].copy()
    merged = preds.merge(
        test_df[
            [
                "sample_id",
                "label_char",
                "era",
                "original_text",
                "mask_position",
            ]
        ],
        on="sample_id",
        how="inner",
        suffixes=("_pred", "_mask"),
    )
    if "label_char_pred" in merged.columns:
        merged["label_char"] = merged["label_char_mask"]
    merged["text_len"] = merged["original_text"].astype(str).str.len()
    label_counts = _train_label_counts(mask_df)
    merged["train_label_count"] = merged["label_char"].astype(str).map(
        lambda ch: label_counts.get(ch, 0)
    )
    merged["freq_bucket"] = merged["train_label_count"].map(_freq_bucket)
    merged["len_bucket"] = merged["text_len"].map(_len_bucket)

    overall = _metrics_for_rows(merged)
    by_freq = {
        name: _metrics_for_rows(merged[merged["freq_bucket"] == name])
        for name, _ in FREQ_BUCKETS
    }
    by_len = {
        name: _metrics_for_rows(merged[merged["len_bucket"] == name])
        for name, _ in LEN_BUCKETS
    }
    by_era = {}
    for era in merged["era"].fillna("unknown").unique():
        by_era[str(era)] = _metrics_for_rows(
            merged[merged["era"].fillna("unknown") == era]
        )

    return {
        model_key: {
            "overall": overall,
            "by_freq_bucket": by_freq,
            "by_len_bucket": by_len,
            "by_era": by_era,
        }
    }


def merge_reports(reports: list[dict]) -> dict:
    merged: dict = {}
    for report in reports:
        merged.update(report)
    return merged


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(
    report: dict,
    *,
    title: str,
    models: list[tuple[str, str]],
    split: str = "test",
) -> str:
    """Render markdown. `models` is (report_key, display_label) pairs."""
    lines = [
        f"# {title}",
        "",
        f"- **생성일:** {date.today().isoformat()}",
        f"- **평가 split:** {split}",
        f"- **데이터:** gsm_preprocess_v4, artificial_mask_dataset.csv",
        "",
    ]
    for report_key, label in models:
        lines.append(f"- **{label}** (`{report_key}`)")
    lines.append("")
    lines.append("## 1. 전체 요약")
    lines.append("")
    lines.append("| 모델 | Exact@1 | Exact@5 | Exact@10 |")
    lines.append("|------|--------:|--------:|---------:|")
    for report_key, label in models:
        overall = report[report_key]["overall"]
        lines.append(
            f"| {label} | {_pct(overall['exact_at_1'])} | "
            f"{_pct(overall['exact_at_5'])} | {_pct(overall['exact_at_10'])} |"
        )
    lines.append("")

    for report_key, label in models:
        lines.append(f"## 정답 글자 train 빈도별 ({label})")
        lines.append("")
        lines.append("| 빈도 구간 | n | Exact@1 | Exact@5 | Exact@10 |")
        lines.append("|-----------|--:|--------:|--------:|---------:|")
        for bucket_name, _ in FREQ_BUCKETS:
            row = report[report_key]["by_freq_bucket"][bucket_name]
            lines.append(
                f"| {bucket_name} | {row['num_samples']} | "
                f"{_pct(row['exact_at_1'])} | {_pct(row['exact_at_5'])} | "
                f"{_pct(row['exact_at_10'])} |"
            )
        lines.append("")

    lines.append("## 희귀자 구간 Exact@5 비교 (파인튜닝 게이트)")
    lines.append("")
    lines.append("| 모델 | very_common | common | mid | rare | hapax | unseen |")
    lines.append("|------|------------:|-------:|----:|-----:|------:|-------:|")
    for report_key, label in models:
        buckets = report[report_key]["by_freq_bucket"]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                _pct(buckets[name]["exact_at_5"])
                for name, _ in FREQ_BUCKETS
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("MODEL_KEY", "CSV_PATH"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="Stratified analysis report")
    parser.add_argument(
        "--experiment-label",
        action="append",
        nargs=2,
        metavar=("EXP_ID", "LABEL"),
        default=[],
    )
    args = parser.parse_args()

    model_specs = [(model_key, Path(csv_path)) for model_key, csv_path in args.prediction]
    reports = [
        build_stratified_report(
            csv_path,
            args.mask_dataset,
            model_key=model_key,
            split=args.split,
        )
        for model_key, csv_path in model_specs
    ]
    merged = merge_reports(reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.experiment_label:
        models = [
            (model_key, f"{exp_id} ({label})")
            for (model_key, _), (exp_id, label) in zip(model_specs, args.experiment_label, strict=True)
        ]
    else:
        models = [(key, key) for key, _ in model_specs]
    md = render_markdown(
        merged,
        title=args.title,
        models=models,
        split=args.split,
    )
    args.output_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
