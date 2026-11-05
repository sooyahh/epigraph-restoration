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
        return "달성 불가 (5% 단위 스윕)"
    return f"{100 * v:.0f}%"

def pct(x):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"

conf = pd.read_csv(ROOT / "reports/exp020/exp020_sample_confidence.csv")
mask = pd.read_csv(ROOT / "data/model_inputs/artificial_mask_dataset.csv")
train_counts = Counter(mask.loc[mask["split"] == "train", "label_char"].astype(str))
valid = mask[mask["split"] == "valid"][["sample_id", "label_char"]].copy()
merged = conf.merge(valid, on="sample_id", how="inner")
merged["train_label_count"] = merged["label_char"].astype(str).map(lambda ch: train_counts.get(ch, 0))
merged["freq_bucket"] = merged["train_label_count"].map(freq_bucket)
assert len(merged) == len(conf)

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
summary = []
for name, _ in FREQ_BUCKETS:
    sub = merged[merged["freq_bucket"] == name].sort_values("top1_score", ascending=False).reset_index(drop=True)
    n_b = len(sub)
    hit5 = sub["hit_at_5"].astype(bool).to_numpy()
    hit1 = sub["hit_at_1"].astype(bool).to_numpy()
    scores = sub["top1_score"].to_numpy()
    max_cov = None
    thr_at = None
    for cov in coverages:
        k = max(1, int(round(n_b * cov))) if n_b else 0
        if n_b == 0 or k == 0:
            e5 = e1 = thr = None
            sel_n = 0
        else:
            e5 = float(hit5[:k].mean())
            e1 = float(hit1[:k].mean())
            thr = float(scores[k - 1])
            sel_n = k
            if e5 >= 0.90:
                max_cov = cov
                thr_at = thr
        curve_rows.append({
            "freq_bucket": name,
            "n_bucket": n_b,
            "coverage": cov,
            "selected_n": sel_n,
            "threshold_top1_score": thr,
            "exact_at_5": None if e5 is None else round(e5, 4),
            "exact_at_1": None if e1 is None else round(e1, 4),
        })

    def at(c):
        if not n_b:
            return None
        kk = max(1, int(round(n_b * c)))
        return round(float(hit5[:kk].mean()), 4)

    e5_full = float(hit5.mean()) if n_b else None
    summary.append({
        "freq_bucket": name,
        "n_bucket": n_b,
        "exact_at_5_full": None if e5_full is None else round(e5_full, 4),
        "exact_at_5_cov25": at(0.25),
        "exact_at_5_cov50": at(0.50),
        "max_coverage_for_exact5_ge_90": max_cov,
        "threshold_at_that_coverage": thr_at,
        "reaches_exact5_90": max_cov is not None,
    })

pd.DataFrame(curve_rows).to_csv(OUT / "exp023_coverage_curves_by_freq.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "exp023_exact5_90_by_freq.csv", index=False)
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
lines.append("- 입력: valid + `exp020_sample_confidence.csv` + train label 빈도 버킷")
lines.append("- 지표: 구간 내부 `top1_score` 스윕; **Exact@5≥90%를 유지하는 최대 커버리지**")
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
vc = next(x for x in comp_rows if x["freq_bucket"].startswith("very_common"))
rareish = sum(
    x["share_top25_conf"]
    for x in comp_rows
    if x["freq_bucket"].startswith(("rare", "hapax", "unseen"))
)
lines.append("")
lines.append(
    f"요약: top25% confidence에서 very_common {100*vc['share_top25_conf']:.1f}% "
    f"(전체 {100*vc['share_all']:.1f}%, lift {vc['lift_top25_vs_all']}). "
    f"rare+hapax+unseen 합 {100*rareish:.1f}%."
)
lines.append("")
lines.append("## 2. Exact@5≥90% 최대 커버리지 (구간 내부)")
lines.append("")
lines.append("| 구간 | n | Exact@5@25% | @50% | @100% | Exact@5≥90% 최대 커버리지 | threshold |")
lines.append("|---|--:|--:|--:|--:|--:|--:|")
for r in summary:
    thr = "—" if r["threshold_at_that_coverage"] is None else f"{r['threshold_at_that_coverage']:.4f}"
    lines.append(
        f"| {r['freq_bucket']} | {r['n_bucket']} | {pct(r['exact_at_5_cov25'])} | {pct(r['exact_at_5_cov50'])} | {pct(r['exact_at_5_full'])} | {cov_label(r['max_coverage_for_exact5_ge_90'])} | {thr} |"
    )
by = {r["freq_bucket"]: r for r in summary}
lines.append("")
lines.append("## 3. 해석 (발표 9 / QA)")
lines.append("")
lines.append(
    f"- very_common: Exact@5≥90%를 **{cov_label(by['very_common (100+)']['max_coverage_for_exact5_ge_90'])}**까지 유지."
)
lines.append(
    f"- common: 최대 커버리지 {cov_label(by['common (21-100)']['max_coverage_for_exact5_ge_90'])}."
)
lines.append(
    f"- mid: 최대 커버리지 {cov_label(by['mid (6-20)']['max_coverage_for_exact5_ge_90'])}."
)
lines.append(
    f"- rare/hapax/unseen: Exact@5≥90% **달성 불가** "
    f"(full Exact@5 각각 {100*by['rare (2-5)']['exact_at_5_full']:.1f}% / "
    f"{100*by['hapax (count=1)']['exact_at_5_full']:.1f}% / {100*by['unseen_in_train']['exact_at_5_full']:.1f}%)."
)
lines.append(
    "- 시사점: 전역 임계값은 고빈도 선택에 유리. 희귀자는 임계값만으로 90%대 Exact@5를 못 만들므로, "
    "구간별 임계값 또는 검색/비예측 보조를 향후 개선·QA 답으로 쓰면 됨."
)
lines.append("")
lines.append("## 4. 산출물")
lines.append("")
lines.append("- `exp023_results.md` / `exp023_summary.json`")
lines.append("- `exp023_top25_composition.csv`")
lines.append("- `exp023_coverage_curves_by_freq.csv`")
lines.append("- `exp023_exact5_90_by_freq.csv`")
lines.append("- `exp023_stratified_coverage_exact5.png` / `.pdf`")
report = "\n".join(lines)
(OUT / "exp023_results.md").write_text(report, encoding="utf-8")
(OUT / "exp023_summary.json").write_text(
    json.dumps({"n_valid": n, "top25_composition": comp_rows, "exact5_90_by_freq": summary}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(report)
