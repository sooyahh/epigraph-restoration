"""Generate baseline results report from evaluation_summary.csv."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


def _overall_rows(summary: pd.DataFrame) -> pd.DataFrame:
    mask = ~summary["experiment_name"].astype(str).str.contains(r"\[", regex=True)
    return summary[mask].copy()


def _latest_overall_per_id(summary: pd.DataFrame) -> pd.DataFrame:
    overall = _overall_rows(summary)
    return (
        overall.sort_values("created_at")
        .groupby("experiment_id", as_index=False)
        .tail(1)
        .sort_values("experiment_id")
    )


def _failure_cases(predictions_csv: Path, limit: int = 20) -> list[dict]:
    if not predictions_csv.exists():
        return []
    df = pd.read_csv(predictions_csv)
    misses = df[df["is_exact_at_10"] == False].head(limit)  # noqa: E712
    rows: list[dict] = []
    for _, row in misses.iterrows():
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "label_char": row.get("label_char", ""),
                "top5_chars": row.get("top5_chars", ""),
                "input_text": str(row.get("input_text", ""))[:80],
            }
        )
    return rows


def generate_report(
    summary_csv: Path = Path("outputs/evaluation/evaluation_summary.csv"),
    output_md: Path = Path("reports/baseline_results_2026-07-03.md"),
    failure_predictions: Path = Path("outputs/predictions/exp006_sillok_ft_predictions.csv"),
    verification_json: Path = Path("work/step1_verification.json"),
) -> Path:
    summary = pd.read_csv(summary_csv)
    canonical_ids = ["EXP-002b", "EXP-003b", "EXP-004", "EXP-005", "EXP-006"]
    compare = _latest_overall_per_id(summary)
    compare = compare[compare["experiment_id"].isin(canonical_ids)]

    verify: dict = {}
    if verification_json.exists():
        verify = json.loads(verification_json.read_text(encoding="utf-8"))

    lines = [
        "# Baseline Results Report",
        "",
        f"- Generated: {date.today().isoformat()}",
        "- Device: **CPU**",
        "- Eval split: `test`",
        "- Primary comparison: **n=2000 subsample, seed=42** (EXP-002b through EXP-006)",
        "",
        "## Data integrity (pre-FT gate)",
        "",
    ]
    if verify:
        si = verify.get("split_integrity", {})
        unk = verify.get("unk_on_test", {})
        lines.extend(
            [
                f"- Mask dataset rows: {verify.get('mask_rows', 'N/A')}",
                f"- Inscription split integrity: train∩test={si.get('train_intersect_test')}, "
                f"train∩valid={si.get('train_intersect_valid')}, valid∩test={si.get('valid_intersect_test')} "
                f"({'PASS' if si.get('passed') else 'FAIL'})",
                f"- SillokBERT UNK labels on full test: {unk.get('unk_labels')} / {unk.get('test_samples')} "
                f"({unk.get('unk_ratio_pct')}%)",
                "",
            ]
        )
    else:
        lines.append("_Verification JSON not found (`work/step1_verification.json`)._")
        lines.append("")

    lines.extend(
        [
            "## Overall Exact@k (test subsample n=2000)",
            "",
            "| EXP | Model | n | Exact@1 | Exact@5 | Exact@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in compare.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['model_name']} | {int(row['num_samples'])} | "
            f"{row['exact_at_1']:.4f} | {row['exact_at_5']:.4f} | {row['exact_at_10']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full test reference (n=9719, non-DL only)",
            "",
            "| EXP | Model | n | Exact@1 | Exact@5 | Exact@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    full_ref = _latest_overall_per_id(summary)
    full_ref = full_ref[full_ref["experiment_id"].isin(["EXP-002", "EXP-003", "EXP-004b", "EXP-005b"])]
    for _, row in full_ref.iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['model_name']} | {int(row['num_samples'])} | "
            f"{row['exact_at_1']:.4f} | {row['exact_at_5']:.4f} | {row['exact_at_10']:.4f} |"
        )

    lines.extend(["", "## Era breakdown (EXP-006 FT)", ""])
    era_rows = summary[
        (summary["experiment_id"] == "EXP-006")
        & summary["experiment_name"].astype(str).str.contains(r"\[", regex=True)
    ]
    if era_rows.empty:
        lines.append("_EXP-006 era breakdown not available yet._")
    else:
        lines.append("| Era | n | Exact@1 | Exact@5 | Exact@10 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, row in era_rows.iterrows():
            era = str(row["experiment_name"]).split("[")[-1].rstrip("]")
            lines.append(
                f"| {era} | {int(row['num_samples'])} | "
                f"{row['exact_at_1']:.4f} | {row['exact_at_5']:.4f} | {row['exact_at_10']:.4f} |"
            )

    failures = _failure_cases(failure_predictions)
    lines.extend(["", "## Failure cases (FT model, Exact@10 miss, up to 20)", ""])
    if failures:
        lines.append("| sample_id | label | top5 | input (truncated) |")
        lines.append("| --- | --- | --- | --- |")
        for item in failures:
            lines.append(
                f"| {item['sample_id']} | {item['label_char']} | {item['top5_chars']} | {item['input_text']} |"
            )
    else:
        lines.append("_No failure-case predictions file yet._")

    unk_pct = verify.get("unk_on_test", {}).get("unk_ratio_pct", "N/A")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- **Split:** inscription-level split; cross-split inscription overlap verified as 0.",
            f"- **Tokenizer:** SillokBERT UNK label ratio on test ≈ {unk_pct}%. "
            "Training skips UNK labels; eval includes all subsampled rows.",
            "- **Context:** MLM eval uses ±64 char window; sequences >512 tokens skipped in FT.",
            "- **FT scale:** CPU run used train subsample max_train=8000 (not full 46,139).",
            "- **Eval scale:** Primary table uses n=2000 subsample (seed=42) for fair DL comparison.",
            "- **SillokBERT advantage:** may partly reflect Joseon annals pretraining overlap with "
            "formulaic epigraphic phrases—not fully separated from generalization in this baseline.",
            "- **Mask type:** single-character artificial masks only; consecutive defects (□□□) not covered.",
            "- **Real defects:** qualitative demo only (`outputs/predictions/real_defect_demo.csv`).",
            "",
            "## Output paths",
            "",
            "- `outputs/evaluation/evaluation_summary.csv`",
            "- `outputs/evaluation/evaluation_summary_canonical.csv`",
            "- `outputs/predictions/exp002b_frequency_sub2k_predictions.csv`",
            "- `outputs/predictions/exp003b_ngram_sub2k_predictions.csv`",
            "- `outputs/predictions/exp004_bert_zs_predictions.csv`",
            "- `outputs/predictions/exp005_sillok_zs_predictions.csv`",
            "- `outputs/predictions/exp006_sillok_ft_predictions.csv`",
            "- `outputs/finetune/sillokbert_gsm_baseline/final/`",
            "- `data/model_inputs/real_defect_inference.csv`",
            "- `outputs/predictions/real_defect_demo.csv`",
            "- `work/step1_verification.json`",
        ]
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def export_canonical_summary(
    summary_csv: Path = Path("outputs/evaluation/evaluation_summary.csv"),
    output_csv: Path = Path("outputs/evaluation/evaluation_summary_canonical.csv"),
) -> Path:
    summary = pd.read_csv(summary_csv)
    overall = _overall_rows(summary)
    canonical = overall.sort_values("created_at").groupby("experiment_id", as_index=False).tail(1)
    canonical = canonical.sort_values(["experiment_id"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv


def main() -> None:
    report_path = generate_report()
    canonical_path = export_canonical_summary()
    print(f"Wrote {report_path}")
    print(f"Wrote {canonical_path}")


if __name__ == "__main__":
    main()
