"""Step 1 verification for baseline resume plan."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    mask_path = ROOT / "data/model_inputs/artificial_mask_dataset.csv"
    rd_path = ROOT / "data/model_inputs/real_defect_inference.csv"
    summary_path = ROOT / "outputs/evaluation/evaluation_summary.csv"

    mask = pd.read_csv(mask_path)
    g = mask.groupby("split")["inscription_id"].apply(set)

    tok = AutoTokenizer.from_pretrained("ddokbaro/SillokBert")
    test = mask[mask["split"] == "test"]
    unk = sum(
        1
        for ch in test["label_char"]
        if tok.convert_tokens_to_ids(ch) in (None, tok.unk_token_id)
    )

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    exp45 = summary[
        summary["experiment_id"].isin(["EXP-004", "EXP-005"])
        & ~summary["experiment_name"].astype(str).str.contains(r"\[", regex=True)
    ]
    canonical = {}
    for exp_id, grp in exp45.groupby("experiment_id"):
        canonical[exp_id] = grp.sort_values("created_at").iloc[-1].to_dict()

    result = {
        "mask_rows": int(len(mask)),
        "mask_splits": mask["split"].value_counts().to_dict(),
        "real_defect_rows": int(len(pd.read_csv(rd_path))) if rd_path.exists() else None,
        "split_integrity": {
            "train_inscriptions": len(g["train"]),
            "valid_inscriptions": len(g["valid"]),
            "test_inscriptions": len(g["test"]),
            "train_intersect_test": len(g["train"] & g["test"]),
            "train_intersect_valid": len(g["train"] & g["valid"]),
            "valid_intersect_test": len(g["valid"] & g["test"]),
            "passed": len(g["train"] & g["test"]) == 0
            and len(g["train"] & g["valid"]) == 0
            and len(g["valid"] & g["test"]) == 0,
        },
        "unk_on_test": {
            "test_samples": int(len(test)),
            "unk_labels": int(unk),
            "unk_ratio_pct": round(unk / len(test) * 100, 2) if len(test) else 0.0,
        },
        "exp004_005_canonical": canonical,
    }

    out = ROOT / "work/step1_verification.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["split_integrity"]["passed"]:
        raise SystemExit("Split integrity check FAILED")


if __name__ == "__main__":
    main()
