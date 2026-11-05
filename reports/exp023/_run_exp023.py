from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"")
OUT = ROOT / "reports" / "exp023"
OUT.mkdir(parents=True, exist_ok=True)

FREQ_BUCKETS = [
    ("very_common (100+)", lambda c: c >= 100),
    ("common (21-100)", lambda c: 21 <= c <= 100),
    ("mid (6-20)", lambda c: 6 <= c <= 20),
    ("rare (2-5)", lambda c: 2 <= c <= 5),
    ("hapax (count=1)", lambda c: c == 1),
    ("unseen_in_train", lambda c: c == 0),
]

def freq_bucket(count: int) -> str:
    for name, pred in FREQ_BUCKETS:
        if pred(count):
            return name
    return "unseen_in_train"

def cov_label(v):
    if v is None:
        return "달성 불가 (스윕 내)"
    return f"{100 * v:.0f}%"

conf = pd.read_csv(ROOT / "reports/exp020/exp020_sample_confidence.csv")
mask = pd.read_csv(ROOT / "data/model_inputs/artificial_mask_dataset.csv")
train_counts = Counter(mask.loc[mask["split"] == "train", "label_char"].astype(str))
valid = mask[mask["split"] == "valid"][["sample_id", "label_char"]].copy()
merged = conf.merge(valid, on="sample_id", how="inner")
merged["train_label_count"] = merged["label_char"].astype(str).map(lambda ch: train_counts.get(ch, 0))
merged["freq_bucket"] = merged["train_label_count"].map(freq_bucket)
assert len(merged) == len(conf), (len(merged), len(conf))

n = len(merged)
k25 = max(1, int(round(n * 0.25)))
top25 = merged.nlargest(k25, "top1_score")
comp_all = merged["freq_bucket"].value_counts(normalize=True)
comp_top = top25["freq_bucket"].value_counts(normalize=True)
comp_rows = []
for name, _ in FREQ_BUCKETS:
    a = float(comp_all.get(name, 0.0))
    t = float(comp_top.get(name, 0.0))
    comp_rows.append({
        "freq_bucket": name,
        "share_all": round(a, 4),
        "share_top25_conf": round(t, 4),
        "lift_top25_vs_all": round(t / a, 3) if a > 0 else None,
        "n_all": int((merged["freq_bucket"] == name).sum()),
        "n_top25": int((top25["freq_bucket"] == name).sum()),
    })
pd.DataFrame(comp_rows).to_csv(OUT / "exp023_top25_composition.csv", index=False)

coverages = [i / 100 for i in range(5, 101, 5)]
curve_rows = []
threshold_90 = []

for name, _ in FREQ_BUCKETS:
    sub = merged[merged["freq_bucket"] == name].sort_values("top1_score", ascending=False).reset_index(drop=True)
    n_b = len(sub)
    hit5 = sub["hit_at_5"].astype(bool).to_numpy()
    hit1 = sub["hit_at_1"].astype(bool).to_numpy()
    scores = sub["top1_score"].to_numpy()
    first_cov_90 = None
    first_thr_90 = None
    for cov in coverages:
        k = max(1, int(round(n_b * cov))) if n_b else 0
        if n_b == 0 or k == 0:
            e5 = None
            e1 = None
            thr = None
            sel_n = 0
        else:
            e5 = float(hit5[:k].mean())
            e1 = float(hit1[:k].mean())
            thr = float(scores[k - 1])
            sel_n = k
            if e5 >= 0.90 and first_cov_90 is None:
                first_cov_90 = cov
                first_thr_90 = thr
        curve_rows.append({
            "freq_bucket": name,
            "n_bucket": n_b,
            "coverage": cov,
            "selected_n": sel_n,
            "threshold_top1_score": thr,
            "exact_at_5": None if e5 is None else round(e5, 4),
            "exact_at_1": None if e1 is None else round(e1, 4),
        })
    e5_full = float(hit5.mean()) if n_b else None
    threshold_90.append({
        "freq_bucket": name,
        "n_bucket": n_b,
        "exact_at_5_full": None if e5_full is None else round(e5_full, 4),
        "min_coverage_for_exact5_ge_90": first_cov_90,
        "threshold_at_that_coverage": first_thr_90,
        "reaches_exact5_90": first_cov_90 is not None,
    })

pd.DataFrame(curve_rows).to_csv(OUT / "exp023_coverage_curves_by_freq.csv", index=False)
pd.DataFrame(threshold_90).to_csv(OUT / "exp023_exact5_90_by_freq.csv", index=False)
curve_df = pd.DataFrame(curve_rows)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for name, _ in FREQ_BUCKETS:
    d = curve_df[curve_df["freq_bucket"] == name]
    if len(d) == 0 or int(d["n_bucket"].iloc[0]) == 0:
        continue
    ax.plot(
        d["coverage"] * 100,
        d["exact_at_5"] * 100,
        marker="o",
        markersize=3,
        label=f"{name} (n={int(d['n_bucket'].iloc[0])})",
    )
ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="Exact@5 = 90%")
ax.set_xlabel("Coverage within frequency bucket (%)")
ax.set_ylabel("Exact@5 among selected (%)")
ax.set_title("EXP-023 Stratified coverage-Exact@5 (valid / EXP-020)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "exp023_stratified_coverage_exact5.png", dpi=150)
fig.savefig(OUT / "exp023_stratified_coverage_exact5.pdf")
plt.close()

lines = []
lines.append("# EXP-023 — 층화 커버리지 (빈도 구간별)")
lines.append("")
lines.append("- 생성: valid split + `reports/exp020/exp020_sample_confidence.csv`")
lines.append("- join: train `label_char` 카운트 → `FREQ_BUCKETS` (`src/eval/stratified_analysis.py`와 동일)")
lines.append("- 점수: `top1_score` 내림차순, **구간 내부** 커버리지 스윕 (5% 단위)")
lines.append("")
lines.append("## 1. 확신 상위 25% 구성")
lines.append("")
lines.append("| 구간 | 전체 비중 | top25% 비중 | lift | n_all | n_top25 |")
lines.append("|---|--:|--:|--:|--:|--:|")
for r in comp_rows:
    lift = "—" if r["lift_top25_vs_all"] is None else f"{r['lift_top25_vs_all']:.2f}"
    lines.append(
        f"| {r['freq_bucket']} | {100*r['share_all']:.1f}% | {100*r['share_top25_conf']:.1f}% | {lift} | {r['n_all']} | {r['n_top25']} |"
    )
vc_all = next(r for r in comp_rows if r["freq_bucket"].startswith("very_common"))
rareish = sum(
    r["share_top25_conf"]
    for r in comp_rows
    if r["freq_bucket"].startswith(("rare", "hapax", "unseen"))
)
lines.append("")
lines.append(
    f"요약: top25% confidence에서 very_common {100*vc_all['share_top25_conf']:.1f}% "
    f"(전체 {100*vc_all['share_all']:.1f}%, lift {vc_all['lift_top25_vs_all']}). "
    f"rare+hapax+unseen 합 {100*rareish:.1f}%."
)
lines.append("")
lines.append("## 2. 구간별 Exact@5 ≥ 90% 최소 커버리지")
lines.append("")
lines.append("| 구간 | n | Exact@5@100% | Exact@5≥90% 최소 커버리지 | threshold |")
lines.append("|---|--:|--:|--:|--:|")
for r in threshold_90:
    thr = "—" if r["threshold_at_that_coverage"] is None else f"{r['threshold_at_that_coverage']:.4f}"
    efull = "—" if r["exact_at_5_full"] is None else f"{100*r['exact_at_5_full']:.1f}%"
    lines.append(
        f"| {r['freq_bucket']} | {r['n_bucket']} | {efull} | {cov_label(r['min_coverage_for_exact5_ge_90'])} | {thr} |"
    )
lines.append("")
lines.append("## 3. 해석 (발표·QA)")
lines.append("")
by = {r["freq_bucket"]: r for r in threshold_90}
for key in ["very_common (100+)", "common (21-100)", "mid (6-20)", "rare (2-5)", "hapax (count=1)", "unseen_in_train"]:
    r = by[key]
    lines.append(
        f"- {key}: full Exact@5 {100*r['exact_at_5_full']:.1f}%; "
        f"Exact@5≥90% → {cov_label(r['min_coverage_for_exact5_ge_90'])}."
    )
lines.append(
    "- 시사점: 전역 임계값(EXP-020/022)은 고빈도 쪽에 선택이 쏠린다. "
    "희귀자 QA에는 구간별 임계값·비예측/검색 보조를 향후 개선으로 제시하는 근거가 된다."
)
lines.append("")
lines.append("## 4. 산출물")
lines.append("")
lines.append("- `exp023_top25_composition.csv`")
lines.append("- `exp023_coverage_curves_by_freq.csv`")
lines.append("- `exp023_exact5_90_by_freq.csv`")
lines.append("- `exp023_stratified_coverage_exact5.png` / `.pdf`")
report = "\n".join(lines)
(OUT / "exp023_results.md").write_text(report, encoding="utf-8")
(OUT / "exp023_summary.json").write_text(
    json.dumps({"n_valid": int(n), "top25_composition": comp_rows, "exact5_90_by_freq": threshold_90}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(report)
print("WROTE", OUT)
