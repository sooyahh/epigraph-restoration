"""EXP-014: probe hapax decline after fine-tuning (hypothesis-level).

Uses available artifacts when full EXP-012 / EXP-007 per-sample predictions
are missing:

1. Train-label frequency buckets from artificial_mask_dataset.csv
2. EXP-011 FT subsample predictions (n=2,000) for hapax/rare/unseen rates
3. EXP-007 / EXP-006 published stratified rates (phase3 report) as zero-shot refs
4. Optional paired zero-shot re-eval on the EXP-011 hapax overlap (requires HF)

Does NOT claim strong causal conclusions — n_hapax is small.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

FREQ_LABELS = {
    "hapax": lambda c: c == 1,
    "unseen": lambda c: c == 0,
    "rare": lambda c: 2 <= c <= 5,
}


def train_label_counts(mask_df: pd.DataFrame) -> Counter[str]:
    train = mask_df[mask_df["split"] == "train"]
    return Counter(train["label_char"].astype(str))


def attach_freq(preds: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    counts = train_label_counts(mask_df)
    test = mask_df[mask_df["split"] == "test"][
        ["sample_id", "label_char", "era", "masked_input", "mask_position", "original_text"]
    ].copy()
    test["train_label_count"] = test["label_char"].astype(str).map(lambda ch: counts.get(ch, 0))
    merged = preds.merge(test, on="sample_id", how="inner", suffixes=("_pred", "_mask"))
    if "label_char_mask" in merged.columns:
        merged["label_char"] = merged["label_char_mask"]
    elif "label_char_pred" in merged.columns:
        merged["label_char"] = merged["label_char_pred"]
    merged["train_label_count"] = merged["label_char"].astype(str).map(
        lambda ch: counts.get(ch, 0)
    )
    return merged


def bucket_metrics(df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, pred in FREQ_LABELS.items():
        sub = df[df["train_label_count"].map(pred)]
        if sub.empty:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": int(len(sub)),
            "exact_at_1": float(sub["is_exact_at_1"].astype(bool).mean()),
            "exact_at_5": float(sub["is_exact_at_5"].astype(bool).mean()),
            "exact_at_10": float(sub["is_exact_at_10"].astype(bool).mean()),
        }
    return out


def unk_stats_for_labels(labels: list[str], model_name: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    unk_id = tok.unk_token_id
    rows = []
    unk_n = 0
    for ch in labels:
        ids = tok.encode(ch, add_special_tokens=False)
        is_unk = (not ids) or (unk_id is not None and unk_id in ids) or len(ids) != 1
        # treat multi-piece or explicit UNK as coverage failure for single-char MLM
        if unk_id is not None and ids == [unk_id]:
            is_unk = True
        elif len(ids) != 1:
            is_unk = True
        if is_unk:
            unk_n += 1
        rows.append({"label_char": ch, "token_ids": ids, "is_unk_or_multi": is_unk})
    return {
        "model": model_name,
        "n_labels": len(labels),
        "unk_or_multi_n": unk_n,
        "unk_or_multi_rate": unk_n / len(labels) if labels else 0.0,
        "examples": [r for r in rows if r["is_unk_or_multi"]][:30],
    }


def classify_failure(row: pd.Series, zs_top: list[str] | None) -> str:
    gold = str(row["label_char"])
    ft_top = [c for c in str(row.get("top10_chars") or "").split(";") if c]
    if zs_top is None:
        return "unpaired"
    zs_hit1 = bool(zs_top and zs_top[0] == gold)
    zs_hit5 = gold in zs_top[:5]
    ft_hit1 = bool(row.get("is_exact_at_1"))
    ft_hit5 = bool(row.get("is_exact_at_5"))
    if zs_hit5 and not ft_hit5:
        # FT lost a ZS hit
        if gold and ft_top and ft_top[0] != gold and zs_hit1:
            # common char substitution?
            return "zs_hit1_ft_miss5_substitution"
        return "zs_hit5_ft_miss5"
    if zs_hit1 and not ft_hit1 and ft_hit5:
        return "zs_hit1_ft_rank_drop"
    if (not zs_hit5) and ft_hit5:
        return "ft_gain_over_zs"
    if zs_hit5 and ft_hit5:
        return "both_hit5"
    return "both_miss5"


def run_paired_zeroshot(
    hapax_df: pd.DataFrame,
    model_name: str,
    *,
    device: str = "cpu",
    max_samples: int | None = None,
) -> pd.DataFrame:
    from src.eval.predictors import FillMaskPredictor

    rows = hapax_df.copy()
    if max_samples is not None:
        rows = rows.head(max_samples)
    # FillMaskPredictor expects original_text / mask_position / label_char
    if "original_text" not in rows.columns and "original_text_mask" in rows.columns:
        rows["original_text"] = rows["original_text_mask"]
    if "mask_position" not in rows.columns and "mask_position_mask" in rows.columns:
        rows["mask_position"] = rows["mask_position_mask"]
    pred = FillMaskPredictor(model_name, device=device, batch_size=4)
    zs_tops: list[list[str]] = []
    for _, row in rows.iterrows():
        zs_tops.append(pred.predict_topk(row, k=10))
    rows = rows.copy()
    rows["zs_top10"] = [";".join(t) for t in zs_tops]
    rows["zs_exact_at_1"] = [
        bool(t and t[0] == str(r.label_char)) for t, (_, r) in zip(zs_tops, rows.iterrows())
    ]
    rows["zs_exact_at_5"] = [
        str(r.label_char) in t[:5] for t, (_, r) in zip(zs_tops, rows.iterrows())
    ]
    rows["failure_type"] = [
        classify_failure(r, t) for t, (_, r) in zip(zs_tops, rows.iterrows())
    ]
    return rows


def markdown_report(payload: dict) -> str:
    lines = [
        "# EXP-014 — hapax 하락 원인 탐색 (가설 수준)",
        "",
        "> **결론을 강하게 쓰지 말 것.** hapax n=232(full test) / EXP-011 겹침 n은 더 작다.",
        "",
        "## 0. 데이터 가용성",
        "",
        f"- artificial_mask: `{payload['paths']['mask']}`",
        f"- EXP-011 FT predictions: `{payload['paths']['ft_predictions']}`",
        f"- EXP-012 full predictions: "
        f"{'있음' if payload['paths'].get('exp012_predictions_exists') else '**없음** (로컬 `notebooks/best/` 미탑재)'}",
        f"- EXP-007 per-sample predictions: "
        f"{'있음' if payload['paths'].get('exp007_predictions_exists') else '**없음** (요약 CSV만)'}",
        "",
        "## 1. Zero-shot 층화 기준치 (EXP-008 / phase3)",
        "",
        "| 모델 | hapax Exact@1 | hapax Exact@5 | hapax n | unseen Exact@5 |",
        "|------|-------------:|-------------:|--------:|---------------:|",
        "| EXP-006 bert-base-chinese zs | 5.2% | 7.8% | 232 | 4.1% |",
        "| EXP-007 SillokBert zs | 3.4% | 8.2% | 232 | 3.7% |",
        "",
        "## 2. EXP-011 FT subsample에서 같은 빈도 버킷",
        "",
    ]
    ft = payload["exp011_ft_buckets"]
    lines.append("| 버킷 | n (subsample) | Exact@1 | Exact@5 | Exact@10 |")
    lines.append("|------|--------------:|--------:|--------:|---------:|")
    for name in ("hapax", "rare", "unseen"):
        b = ft.get(name, {})
        if not b.get("n"):
            lines.append(f"| {name} | 0 | — | — | — |")
            continue
        lines.append(
            f"| {name} | {b['n']} | {b['exact_at_1']*100:.1f}% | "
            f"{b['exact_at_5']*100:.1f}% | {b['exact_at_10']*100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "동일 full-test hapax 232와 표본이 다르므로 **직접 동일시하지 않는다.**",
            "다만 EXP-011 겹침 hapax에서 Exact@5가 zero-shot 8.2%보다 낮게 관측되면",
            "“FT가 희귀 정답 복원을 약화했을 가능성”을 **가설**로 둔다.",
            "",
            "## 3. Hapax 정답 × tokenizer UNK (EXP-010 교차)",
            "",
        ]
    )
    for key in ("unk_hapax_test", "unk_hapax_exp011_overlap"):
        u = payload.get(key)
        if not u:
            continue
        lines.append(f"### {key}")
        lines.append(
            f"- model `{u['model']}`: UNK/multi-piece "
            f"**{u['unk_or_multi_n']}/{u['n_labels']}** "
            f"({u['unk_or_multi_rate']*100:.1f}%)"
        )
        if u.get("examples"):
            ex = ", ".join(e["label_char"] for e in u["examples"][:15])
            lines.append(f"- examples: {ex}")
        lines.append("")
    lines.extend(
        [
            "Fine-tuning은 tokenizer를 바꾸지 않으므로 UNK 비율 자체는 FT 전후 동일하다.",
            "UNK hapax는 ZS·FT 모두 Exact@1이 원리적으로 어렵다(단일 토큰 복원).",
            "",
            "## 4. 쌍대 비교 (선택 실행)",
            "",
        ]
    )
    paired = payload.get("paired")
    if not paired:
        lines.append("_paired zero-shot 재평가 미실행 (`--run-paired-zs` 필요)._")
    else:
        lines.append(f"- overlap n: **{paired['n']}**")
        lines.append(
            f"- ZS Exact@1/5: {paired['zs_exact_at_1']*100:.1f}% / "
            f"{paired['zs_exact_at_5']*100:.1f}%"
        )
        lines.append(
            f"- FT Exact@1/5: {paired['ft_exact_at_1']*100:.1f}% / "
            f"{paired['ft_exact_at_5']*100:.1f}%"
        )
        lines.append(f"- ZS@5∧¬FT@5 (하락 후보): **{paired['zs_hit_ft_miss']}**")
        lines.append("")
        lines.append("| failure_type | n |")
        lines.append("|-------------|--:|")
        for k, v in paired.get("failure_type_counts", {}).items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    lines.extend(
        [
            "## 5. 해석 (가설만)",
            "",
            "1. FT는 고빈도·중빈도 Exact@k를 올리지만, hapax/unseen에서는 이득이 없거나 감소할 수 있다.",
            "2. Tokenizer UNK는 hapax 실패의 **일부**만 설명(전후 동일). 나머지는 빈도 편향·문맥 암기로 보는 편이 안전하다.",
            "3. EXP-012 full 9,719 예측이 오면 동일 스크립트로 hapax 232 전수를 재집계할 것.",
            "",
            "## 산출물",
            "",
            f"- JSON: `{payload['paths']['out_json']}`",
            f"- 본 리포트: `{payload['paths']['out_md']}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mask-dataset",
        type=Path,
        default=Path("data/model_inputs/artificial_mask_dataset.csv"),
    )
    parser.add_argument(
        "--ft-predictions",
        type=Path,
        default=Path("reports/exp011_baseline/exp006_sillok_ft_predictions.csv"),
    )
    parser.add_argument(
        "--exp012-predictions",
        type=Path,
        default=Path("notebooks/best/exp012_sillok_ft_full_predictions.csv"),
    )
    parser.add_argument(
        "--exp007-predictions",
        type=Path,
        default=Path("outputs/predictions/exp007_sillokbert_test_predictions.csv"),
    )
    parser.add_argument("--tokenizer-model", default="ddokbaro/SillokBert")
    parser.add_argument("--run-paired-zs", action="store_true")
    parser.add_argument("--paired-max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/exp014_hapax_decline.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/exp014_hapax_decline.md"),
    )
    parser.add_argument(
        "--out-paired-csv",
        type=Path,
        default=Path("reports/exp014_hapax_paired_cases.csv"),
    )
    args = parser.parse_args()

    mask_df = pd.read_csv(args.mask_dataset)
    ft = pd.read_csv(args.ft_predictions)
    merged = attach_freq(ft, mask_df)
    buckets = bucket_metrics(merged)

    counts = train_label_counts(mask_df)
    test = mask_df[mask_df["split"] == "test"].copy()
    test["train_label_count"] = test["label_char"].astype(str).map(lambda c: counts.get(c, 0))
    hapax_test = test[test["train_label_count"] == 1]
    hapax_labels = hapax_test["label_char"].astype(str).tolist()
    overlap = merged[merged["train_label_count"] == 1]
    overlap_labels = overlap["label_char"].astype(str).tolist()

    unk_all = unk_stats_for_labels(hapax_labels, args.tokenizer_model)
    unk_overlap = unk_stats_for_labels(overlap_labels, args.tokenizer_model)

    paired_payload = None
    if args.run_paired_zs:
        paired_df = run_paired_zeroshot(
            overlap,
            args.tokenizer_model,
            device=args.device,
            max_samples=args.paired_max_samples,
        )
        args.out_paired_csv.parent.mkdir(parents=True, exist_ok=True)
        paired_df.to_csv(args.out_paired_csv, index=False)
        paired_payload = {
            "n": int(len(paired_df)),
            "zs_exact_at_1": float(paired_df["zs_exact_at_1"].mean()),
            "zs_exact_at_5": float(paired_df["zs_exact_at_5"].mean()),
            "ft_exact_at_1": float(paired_df["is_exact_at_1"].astype(bool).mean()),
            "ft_exact_at_5": float(paired_df["is_exact_at_5"].astype(bool).mean()),
            "zs_hit_ft_miss": int(
                ((paired_df["zs_exact_at_5"]) & (~paired_df["is_exact_at_5"].astype(bool))).sum()
            ),
            "failure_type_counts": paired_df["failure_type"].value_counts().to_dict(),
            "paired_csv": str(args.out_paired_csv),
        }

    payload = {
        "experiment_id": "EXP-014",
        "status": "hypothesis",
        "paths": {
            "mask": str(args.mask_dataset),
            "ft_predictions": str(args.ft_predictions),
            "exp012_predictions_exists": args.exp012_predictions.exists(),
            "exp007_predictions_exists": args.exp007_predictions.exists(),
            "out_json": str(args.out_json),
            "out_md": str(args.out_md),
        },
        "full_test_hapax_n": int(len(hapax_test)),
        "full_test_unseen_n": int((test["train_label_count"] == 0).sum()),
        "exp011_ft_buckets": buckets,
        "unk_hapax_test": unk_all,
        "unk_hapax_exp011_overlap": unk_overlap,
        "paired": paired_payload,
        "phase3_zeroshot_reference": {
            "source": "reports/phase3_stratified_analysis_2026-07-06.md",
            "exp006_hapax_exact_at_5": 0.078,
            "exp007_hapax_exact_at_5": 0.082,
            "exp007_hapax_exact_at_1": 0.034,
            "n": 232,
        },
    }

    # Prefer EXP-012 if present
    if args.exp012_predictions.exists():
        exp012 = pd.read_csv(args.exp012_predictions)
        m12 = attach_freq(exp012, mask_df)
        payload["exp012_ft_buckets"] = bucket_metrics(m12)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"wrote_json": str(args.out_json), "wrote_md": str(args.out_md), "paired": bool(paired_payload)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
