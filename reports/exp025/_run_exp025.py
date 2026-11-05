import json
from pathlib import Path
from src.eval.paired_exact_compare import paired_exact_compare, load_predictions

inp = Path("reports/exp025/inputs")
out = Path("reports/exp025")
out.mkdir(parents=True, exist_ok=True)

pred024 = load_predictions(inp / "exp024_qlora_beam5_n200_predictions.csv")
pred021 = load_predictions(inp / "exp021_g3_beam5_n500_predictions.csv")
pred015 = load_predictions(inp / "exp015_sillok_ft_valid_predictions.csv")

ids024 = set(pred024["sample_id"])
print("n024", len(ids024))
print("024n021", len(ids024 & set(pred021["sample_id"])))
print("024n015", len(ids024 & set(pred015["sample_id"])))
print(
    "024 Exact rates",
    float(pred024["is_exact_at_1"].astype(float).mean()),
    float(pred024["is_exact_at_5"].astype(float).mean()),
    float(pred024["is_exact_at_10"].astype(float).mean()),
)

pred021_200 = pred021[pred021["sample_id"].isin(ids024)].copy()
pred015_200 = pred015[pred015["sample_id"].isin(ids024)].copy()
pred021_200.to_csv(out / "exp021_g3_beam5_same200_as_exp024.csv", index=False)
pred015_200.to_csv(out / "exp015_sillok_same200_as_exp024.csv", index=False)
pred024.to_csv(out / "exp024_qlora_beam5_n200_predictions.csv", index=False)

pairs = [
    ("EXP015_SillokBERT", pred015, "ARI8B_QLoRA_FT", pred024, "paired_exp015_vs_exp024"),
    ("ARI8B_ZS_EXP021", pred021, "ARI8B_QLoRA_FT", pred024, "paired_exp021_vs_exp024"),
]

summary = {"n_exp024": int(len(ids024)), "comparisons": {}}
for label_a, a, label_b, b, stem in pairs:
    for k in (1, 5, 10):
        try:
            res = paired_exact_compare(a, b, k=k, label_a=label_a, label_b=label_b)
        except Exception as e:
            res = {"error": str(e), "k": k, "label_a": label_a, "label_b": label_b}
        path = out / f"{stem}_k{k}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["comparisons"][f"{stem}_k{k}"] = res
        print(stem, f"k={k}")
        print(json.dumps(res, ensure_ascii=False, indent=2))

(out / "exp025_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("WROTE", out / "exp025_summary.json")
