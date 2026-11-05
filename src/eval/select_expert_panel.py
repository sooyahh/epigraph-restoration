"""Select diversified expert-review cases from artificial / real_defect predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_MODEL_VERSION = "exp012_sillokbert_gpu_best_valid_loss"


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes"}


def select_artificial(
    predictions: pd.DataFrame,
    mask_meta: pd.DataFrame,
    *,
    n: int = 25,
    max_per_inscription: int = 2,
    min_position_gap: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    meta_cols = [
        "sample_id",
        "inscription_id",
        "artifact_id",
        "title",
        "era",
        "mask_position",
        "original_text",
        "masked_input",
    ]
    meta = mask_meta[meta_cols].drop_duplicates("sample_id")
    df = predictions.merge(meta, on="sample_id", how="inner")
    df["is_hit"] = df["is_exact_at_1"].map(_parse_bool)
    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df = df.dropna(subset=["top1_score", "inscription_id"]).copy()

    score_q33 = float(df["top1_score"].quantile(0.33))
    score_q66 = float(df["top1_score"].quantile(0.66))

    def bucket(row: pd.Series) -> str:
        hit = "hit" if row["is_hit"] else "miss"
        if row["top1_score"] <= score_q33:
            score_b = "low"
        elif row["top1_score"] >= score_q66:
            score_b = "high"
        else:
            score_b = "mid"
        return f"{hit}_{score_b}"

    df["panel_bucket"] = df.apply(bucket, axis=1)
    # stable shuffle within buckets
    shuffled = (
        df.sample(frac=1.0, random_state=seed)
        .sort_values(["panel_bucket", "era", "sample_id"])
        .reset_index(drop=True)
    )

    selected: list[pd.Series] = []
    per_ins: dict[str, list[int]] = {}
    era_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    target_per_bucket = max(1, n // max(shuffled["panel_bucket"].nunique(), 1))

    def can_take(row: pd.Series) -> bool:
        ins = str(row["inscription_id"])
        pos = int(row["mask_position"])
        taken_pos = per_ins.get(ins, [])
        if len(taken_pos) >= max_per_inscription:
            return False
        if any(abs(pos - p) < min_position_gap for p in taken_pos):
            return False
        return True

    # Pass 1: fill buckets while respecting inscription/position constraints
    for _, row in shuffled.iterrows():
        if len(selected) >= n:
            break
        b = row["panel_bucket"]
        if bucket_counts.get(b, 0) >= target_per_bucket + 1:
            continue
        if not can_take(row):
            continue
        selected.append(row)
        ins = str(row["inscription_id"])
        per_ins.setdefault(ins, []).append(int(row["mask_position"]))
        era_counts[str(row["era"])] = era_counts.get(str(row["era"]), 0) + 1
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # Pass 2: fill remaining slots
    if len(selected) < n:
        chosen_ids = {str(r["sample_id"]) for r in selected}
        for _, row in shuffled.iterrows():
            if len(selected) >= n:
                break
            if str(row["sample_id"]) in chosen_ids:
                continue
            if not can_take(row):
                continue
            selected.append(row)
            ins = str(row["inscription_id"])
            per_ins.setdefault(ins, []).append(int(row["mask_position"]))
            chosen_ids.add(str(row["sample_id"]))

    out = pd.DataFrame(selected)
    out["prediction_type"] = "artificial_mask"
    out["selection_reason"] = out["panel_bucket"].map(
        lambda b: f"diversified; bucket={b}; max_per_ins={max_per_inscription}"
    )
    return out


def select_real_defect(
    predictions: pd.DataFrame,
    cases: pd.DataFrame,
    *,
    n: int = 25,
    max_per_inscription: int = 2,
    min_position_gap: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    meta_cols = [
        "case_id",
        "inscription_id",
        "artifact_id",
        "title",
        "era",
        "defect_position",
        "model_input",
        "text_with_defect",
    ]
    meta = cases[[c for c in meta_cols if c in cases.columns]].drop_duplicates("case_id")
    df = predictions.merge(meta, on="case_id", how="inner")
    df["top1_score"] = pd.to_numeric(df["top1_score"], errors="coerce")
    df = df.dropna(subset=["top1_score", "inscription_id"]).copy()
    if "defect_position" not in df.columns:
        df["defect_position"] = 0

    score_q33 = float(df["top1_score"].quantile(0.33))
    score_q66 = float(df["top1_score"].quantile(0.66))

    def bucket(score: float) -> str:
        if score <= score_q33:
            return "low"
        if score >= score_q66:
            return "high"
        return "mid"

    df["panel_bucket"] = df["top1_score"].map(bucket)
    shuffled = df.sample(frac=1.0, random_state=seed).sort_values(
        ["panel_bucket", "era", "case_id"]
    )

    selected: list[pd.Series] = []
    per_ins: dict[str, list[int]] = {}
    target_per_bucket = max(1, n // 3)

    def can_take(row: pd.Series) -> bool:
        ins = str(row["inscription_id"])
        pos = int(row["defect_position"])
        taken = per_ins.get(ins, [])
        if len(taken) >= max_per_inscription:
            return False
        if any(abs(pos - p) < min_position_gap for p in taken):
            return False
        return True

    bucket_counts: dict[str, int] = {}
    for _, row in shuffled.iterrows():
        if len(selected) >= n:
            break
        b = str(row["panel_bucket"])
        if bucket_counts.get(b, 0) >= target_per_bucket + 1:
            continue
        if not can_take(row):
            continue
        selected.append(row)
        ins = str(row["inscription_id"])
        per_ins.setdefault(ins, []).append(int(row["defect_position"]))
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    if len(selected) < n:
        chosen = {str(r["case_id"]) for r in selected}
        for _, row in shuffled.iterrows():
            if len(selected) >= n:
                break
            if str(row["case_id"]) in chosen:
                continue
            if not can_take(row):
                continue
            selected.append(row)
            ins = str(row["inscription_id"])
            per_ins.setdefault(ins, []).append(int(row["defect_position"]))
            chosen.add(str(row["case_id"]))

    out = pd.DataFrame(selected)
    out["prediction_type"] = "real_defect"
    out["is_hit"] = ""
    out["selection_reason"] = out["panel_bucket"].map(
        lambda b: f"diversified_real; score_bucket={b}; no_gold"
    )
    return out


def _candidates_from_row(row: pd.Series, k: int = 10) -> list[dict]:
    chars = str(row.get("top10_chars") or "").split(";")
    scores_raw = str(row.get("top10_scores") or "").split(";")
    out: list[dict] = []
    for i, ch in enumerate(chars[:k]):
        if not ch:
            continue
        score = None
        if i < len(scores_raw) and scores_raw[i]:
            try:
                score = float(scores_raw[i])
            except ValueError:
                score = None
        item = {"rank": i + 1, "char": ch}
        if score is not None:
            item["score"] = round(score, 6)
        out.append(item)
    return out


def to_handoff_json(
    panel: pd.DataFrame,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict:
    samples = []
    for _, row in panel.iterrows():
        text = str(row.get("input_text") or row.get("masked_input") or row.get("model_input") or "")
        mask_index = text.find("[MASK]")
        sample = {
            "sample_id": str(row.get("sample_id") or ""),
            "case_id": str(row.get("case_id") or ""),
            "prediction_type": str(row.get("prediction_type")),
            "inscription_id": str(row.get("inscription_id") or ""),
            "era": str(row.get("era") or ""),
            "selection_reason": str(row.get("selection_reason") or ""),
            "request": {"text": text, "top_k": 10},
            "response": {
                "input_text": text,
                "mask_index": mask_index if mask_index >= 0 else None,
                "candidates": _candidates_from_row(row),
                "model_version": model_version,
                "meta": {"max_seq_len": 512, "context_window": 64},
            },
        }
        if str(row.get("prediction_type")) == "artificial_mask":
            sample["gold"] = {
                "label_char": str(row.get("label_char") or ""),
                "rank_of_gold": row.get("rank_of_gold"),
                "is_exact_at_1": _parse_bool(row.get("is_exact_at_1")),
                "is_exact_at_5": _parse_bool(row.get("is_exact_at_5")),
                "is_exact_at_10": _parse_bool(row.get("is_exact_at_10")),
            }
        else:
            sample["gold"] = None
        samples.append(sample)
    return {
        "description": (
            "EXP-012 expert usability panel — diversified inscriptions; "
            "scores are relative preference, not calibrated confidence."
        ),
        "model_version": model_version,
        "count": len(samples),
        "samples": samples,
    }


PANEL_COLUMNS = [
    "prediction_type",
    "sample_id",
    "case_id",
    "inscription_id",
    "artifact_id",
    "title",
    "era",
    "mask_position",
    "defect_position",
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
    "is_hit",
    "panel_bucket",
    "selection_reason",
    "model_version",
]


def normalize_panel(df: pd.DataFrame, model_version: str) -> pd.DataFrame:
    out = df.copy()
    if "input_text" not in out.columns or out["input_text"].isna().all():
        if "masked_input" in out.columns:
            out["input_text"] = out["masked_input"]
        elif "model_input" in out.columns:
            out["input_text"] = out["model_input"]
    if "mask_position" not in out.columns:
        out["mask_position"] = ""
    if "defect_position" not in out.columns:
        out["defect_position"] = ""
    out["model_version"] = model_version
    for col in PANEL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[PANEL_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artificial-predictions",
        type=Path,
        default=Path("notebooks/best/exp012_sillok_ft_full_predictions.csv"),
    )
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument(
        "--real-predictions",
        type=Path,
        default=Path("outputs/predictions/exp012_real_defect_predictions.csv"),
    )
    parser.add_argument(
        "--real-cases",
        type=Path,
        default=Path("data/model_inputs/real_defect_inference.csv"),
    )
    parser.add_argument("--n-artificial", type=int, default=25)
    parser.add_argument("--n-real", type=int, default=25)
    parser.add_argument("--max-per-inscription", type=int, default=2)
    parser.add_argument("--min-position-gap", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/expert_panel/expert_review_panel.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("handoff/fixtures/expert_review_panel.json"),
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Build artificial-only panel if real_defect predictions are missing.",
    )
    args = parser.parse_args()

    art_pred = pd.read_csv(args.artificial_predictions)
    mask_meta = pd.read_csv(
        args.mask_dataset,
        usecols=[
            "sample_id",
            "inscription_id",
            "artifact_id",
            "title",
            "era",
            "mask_position",
            "original_text",
            "masked_input",
        ],
    )
    art = select_artificial(
        art_pred,
        mask_meta,
        n=args.n_artificial,
        max_per_inscription=args.max_per_inscription,
        min_position_gap=args.min_position_gap,
        seed=args.seed,
    )

    frames = [normalize_panel(art, args.model_version)]
    if not args.skip_real:
        if not args.real_predictions.exists():
            raise SystemExit(
                f"Missing real predictions: {args.real_predictions}. "
                "Run real_defect inference first or pass --skip-real."
            )
        real_pred = pd.read_csv(args.real_predictions)
        real_cases = pd.read_csv(args.real_cases)
        real = select_real_defect(
            real_pred,
            real_cases,
            n=args.n_real,
            max_per_inscription=args.max_per_inscription,
            min_position_gap=args.min_position_gap,
            seed=args.seed,
        )
        frames.append(normalize_panel(real, args.model_version))

    panel = pd.concat(frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out_csv, index=False)
    payload = to_handoff_json(panel, model_version=args.model_version)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # sanity print
    by_type = panel["prediction_type"].value_counts().to_dict()
    max_ins = (
        panel.groupby(["prediction_type", "inscription_id"]).size().max()
        if len(panel)
        else 0
    )
    print(
        json.dumps(
            {
                "wrote_csv": str(args.out_csv),
                "wrote_json": str(args.out_json),
                "counts": by_type,
                "n": len(panel),
                "max_rows_per_inscription": int(max_ins),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
