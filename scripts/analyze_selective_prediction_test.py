"""EXP-022: apply the EXP-020 valid-fixed selective-prediction threshold to EXP-012 test predictions.

Post-processing only. No model loading, no re-inference, no threshold search on test.
The threshold is read from reports/exp020/exp020_results.json and never re-tuned here.

Usage:
    python scripts/analyze_selective_prediction_test.py --predictions <exp012_sillok_ft_full_predictions.csv>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_selective_prediction import (  # noqa: E402
    COVERAGES,
    FONT_PATH,
    KS,
    _split,
    add_confidence,
    confidence_order,
    load_predictions,
    wilson,
)
from src.eval.stratified_analysis import FREQ_BUCKETS, _freq_bucket  # noqa: E402

DEFAULT_PRED = ROOT / "notebooks/best/exp012_sillok_ft_full_predictions.csv"
DEFAULT_MASK = ROOT / "data/model_inputs/artificial_mask_dataset.csv"
EXP020_JSON = ROOT / "reports/exp020/exp020_results.json"
EXP020_CONF = ROOT / "reports/exp020/exp020_sample_confidence.csv"
DEFAULT_OUT = ROOT / "reports/exp022"

EXPECTED_N = 9719
OFFICIAL_EXP012 = {
    "exact_at_1": 0.3374832801728573,
    "exact_at_5": 0.5304043625887437,
    "exact_at_10": 0.6046918407243543,
}
ROUNDED_THRESHOLD = 0.617


def fixed_threshold_eval(conf: pd.DataFrame, thr: float) -> dict:
    sel = (conf["top1_score"] >= thr).to_numpy()
    m, n = int(sel.sum()), len(conf)
    res = {"threshold": thr, "n_total": n, "n_selected": m, "coverage": m / n}
    for k in KS:
        hits = int(conf[f"hit_at_{k}"].to_numpy()[sel].sum())
        lo, hi = wilson(hits, m)
        res[f"exact_at_{k}"] = hits / m if m else None
        res[f"exact_at_{k}_hits"] = hits
        res[f"exact_at_{k}_wilson95"] = [lo, hi]
    cov_lo, cov_hi = wilson(m, n)
    res["coverage_wilson95"] = [cov_lo, cov_hi]
    return res


def two_proportion_z(h1: int, n1: int, h2: int, n2: int) -> dict:
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    pval = math.erfc(abs(z) / math.sqrt(2))
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"diff_test_minus_valid": p2 - p1, "diff_ci95": [p2 - p1 - 1.96 * se_diff, p2 - p1 + 1.96 * se_diff],
            "z": z, "p_two_sided": pval}


def curve(conf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    order = confidence_order(conf, "top1_score")
    n = len(conf)
    cum = {k: np.cumsum(conf[f"hit_at_{k}"].to_numpy()[order].astype(int)) for k in KS}
    conf_sorted = conf["top1_score"].to_numpy()[order]
    rows = []
    for c in COVERAGES:
        m = max(1, int(round(c * n)))
        row = {"coverage": c, "n_selected": m, "threshold_at_coverage": float(conf_sorted[m - 1])}
        for k in KS:
            row[f"exact_at_{k}"] = cum[k][m - 1] / m
        rows.append(row)
    return pd.DataFrame(rows), {**{f"cum{k}": cum[k] for k in KS}, "n": n}


def bucket_crosscheck(conf: pd.DataFrame, mask_csv: Path, sel: np.ndarray, split: str) -> dict:
    mask = pd.read_csv(mask_csv, usecols=["sample_id", "split", "label_char"], dtype=str)
    counts = Counter(mask.loc[mask["split"] == "train", "label_char"].astype(str))
    split_ids = set(mask.loc[mask["split"] == split, "sample_id"])
    buckets = conf["label_char"].astype(str).map(lambda ch: _freq_bucket(counts.get(ch, 0)))
    m, n = int(sel.sum()), len(conf)
    out = {"n_total": n, "n_selected": m,
           f"rows_not_in_{split}_split": int((~conf["sample_id"].isin(split_ids)).sum()), "buckets": {}}
    for name, _ in FREQ_BUCKETS:
        in_b = (buckets == name).to_numpy()
        sel_b = in_b & sel
        n_b, n_sel = int(in_b.sum()), int(sel_b.sum())
        e = {"n_all": n_b, "share_all": n_b / n, "n_selected": n_sel,
             "share_selected": n_sel / m if m else 0.0, "bucket_coverage": n_sel / n_b if n_b else 0.0}
        for k in KS:
            hits = conf[f"hit_at_{k}"].to_numpy()
            e[f"exact_at_{k}_all"] = float(hits[in_b].mean()) if n_b else None
            e[f"exact_at_{k}_selected"] = float(hits[sel_b].mean()) if n_sel else None
        h5 = int(conf["hit_at_5"].to_numpy()[sel_b].sum())
        e["exact_at_5_selected_wilson95"] = list(wilson(h5, n_sel)) if n_sel else None
        out["buckets"][name] = e
    return out


def plot_overlay(vcum: dict, tcum: dict, vfix: dict, tfix: dict, out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if not FONT_PATH.exists():
        raise SystemExit(f"[STOP] Korean font not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 15,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    series = [("valid", vcum, vfix, "#2c3e50", "-"), ("test", tcum, tfix, "#c0392b", "-")]
    for name, cum, fix, color, ls in series:
        n = cum["n"]
        sizes = np.arange(1, n + 1)
        x = sizes / n * 100
        start = max(1, int(0.01 * n)) - 1
        ax.plot(x[start:], (cum["cum5"] / sizes * 100)[start:], color=color, lw=3.0, ls=ls,
                label=f"Exact@5 · {name} (n={n:,})")
        ax.plot(x[start:], (cum["cum1"] / sizes * 100)[start:], color=color, lw=1.4, ls="--", alpha=0.6,
                label=f"Exact@1 · {name}")

    ax.axhline(90, color="#888", ls=":", lw=1.2)
    ax.text(99, 90.8, "Exact@5 = 90%", color="#666", ha="right", va="bottom", fontsize=12)

    for name, fix, color, tx, ty in [("valid", vfix, "#2c3e50", 34, 31), ("test", tfix, "#c0392b", 34, 17)]:
        cx, cy = fix["coverage"] * 100, fix["exact_at_5"] * 100
        lo, hi = (v * 100 for v in fix["exact_at_5_wilson95"])
        ax.errorbar([cx], [cy], yerr=[[cy - lo], [hi - cy]], fmt="o", color=color, ms=9, capsize=5, zorder=6)
        ax.annotate(
            f"{name}: 커버리지 {cx:.1f}% (n={fix['n_selected']:,})\nExact@5 {cy:.1f}% (95% CI {lo:.1f}–{hi:.1f})",
            xy=(cx, cy), xytext=(tx, ty), fontsize=13, color=color, zorder=7,
            bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": color, "lw": 1.2},
            arrowprops={"arrowstyle": "->", "color": color, "relpos": (0, 1)},
        )

    ax.set_xlim(0, 101)
    ax.set_ylim(0, 102)
    ax.set_xlabel("커버리지 (모델이 후보를 제시하는 결손 비율, %)")
    ax.set_ylabel("정확도 (%)")
    ax.set_title(f"선택적 예측 재현성: valid에서 고정한 임계값(top-1 확률 ≥ {vfix['threshold']:.3f})을 test에 그대로 적용",
                 fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.text(0.99, 0.01, "EXP-022 · 곡선은 참고용, 주력 수치는 고정 임계값 지점 · EXP-015 valid / EXP-012 test 예측 후처리 (재추론 없음)",
             ha="right", fontsize=10, color="#666")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def ci(pair: list[float] | None) -> str:
    return "-" if not pair else f"{pair[0] * 100:.1f}–{pair[1] * 100:.1f}"


def render_md(s: dict, tcurve: pd.DataFrame, vcurve: pd.DataFrame) -> str:
    v, t = s["fixed_threshold"]["valid"], s["fixed_threshold"]["test"]
    L = ["# EXP-022 — 선택적 예측 임계값 test 재현성 확인 (EXP-020 valid 임계값 고정)\n",
         f"- 생성: {s['generated_at']}",
         f"- test 입력: `{s['input_csv']}` (n={s['n_test']:,}; EXP-012 기존 예측 재집계, 재추론 없음)",
         f"- valid 참조: `{s['exp020_json']}`, `{s['exp020_sample_confidence']}`",
         f"- 고정 임계값: top-1 확률 ≥ **{s['threshold']}** (EXP-020 `max_coverage.top1_score[\"exact_at_5>=90%\"].fine.threshold`). "
         "test에서 임계값 재탐색 없음.\n",
         "## 1. sanity check\n"]
    for line in s["sanity"]["notes"]:
        L.append(f"- {line}")
    L.append(f"- 판정: **{'통과' if s['sanity']['passed'] else '불일치'}**\n")

    L.append("## 2. 고정 임계값 적용 결과 (주력)\n")
    L.append("| split | 임계값 | 커버리지 | 선택 n / 전체 | Exact@1 (95% CI) | Exact@5 (95% CI) | Exact@10 (95% CI) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for name, r in [("valid (EXP-020)", v), ("**test (EXP-022)**", t)]:
        L.append(f"| {name} | {r['threshold']} | {pct(r['coverage'])} | {r['n_selected']:,} / {r['n_total']:,} | "
                 f"{pct(r['exact_at_1'])} ({ci(r['exact_at_1_wilson95'])}) | "
                 f"{pct(r['exact_at_5'])} ({ci(r['exact_at_5_wilson95'])}) | "
                 f"{pct(r['exact_at_10'])} ({ci(r['exact_at_10_wilson95'])}) |")
    L.append("")
    L.append("valid − test 차이 (독립 두 비율 z 검정, 참고):\n")
    L.append("| 항목 | test − valid | 95% CI | p |")
    L.append("|---|--:|--:|--:|")
    for key, d in s["valid_vs_test"].items():
        L.append(f"| {key} | {d['diff_test_minus_valid'] * 100:+.1f}pp | "
                 f"{d['diff_ci95'][0] * 100:+.1f} ~ {d['diff_ci95'][1] * 100:+.1f}pp | {d['p_two_sided']:.3g} |")
    L.append("")
    r2 = s["fixed_threshold"]["test_rounded_0617"]
    L.append(f"참고 — 반올림 임계값 {ROUNDED_THRESHOLD} 적용 시 test: 커버리지 {pct(r2['coverage'])} (n={r2['n_selected']:,}), "
             f"Exact@1/5/10 {pct(r2['exact_at_1'])} / {pct(r2['exact_at_5'])} / {pct(r2['exact_at_10'])}. "
             "주력 수치는 EXP-020 저장값 0.617386 기준.\n")

    L.append("## 3. coverage-accuracy 곡선 (참고용, 5% 단위)\n")
    L.append("> 보조 자료. test 곡선에서 'Exact@5 90%를 만족하는 최대 커버리지'를 찾는 것은 test 튜닝이므로 산출·보고하지 않는다.\n")
    L.append("| 커버리지 | valid Exact@1 | valid Exact@5 | valid Exact@10 | test n | test 임계값(해당 커버리지) | test Exact@1 | test Exact@5 | test Exact@10 |")
    L.append("|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for (_, vr), (_, tr) in zip(vcurve.iterrows(), tcurve.iterrows()):
        L.append(f"| {tr['coverage'] * 100:.0f}% | {pct(vr['exact_at_1'])} | {pct(vr['exact_at_5'])} | {pct(vr['exact_at_10'])} | "
                 f"{int(tr['n_selected']):,} | {tr['threshold_at_coverage']:.4f} | {pct(tr['exact_at_1'])} | "
                 f"{pct(tr['exact_at_5'])} | {pct(tr['exact_at_10'])} |")
    L.append("")

    L.append("## 4. 빈도 구간 교차 확인 (고정 임계값 선택 구간)\n")
    vb, tb = s["bucket_crosscheck"]["valid"], s["bucket_crosscheck"]["test"]
    L.append("| 구간 | valid 선택 비중 (전체) | test 선택 비중 (전체) | valid 구간 내 선택률 | test 구간 내 선택률 | "
             "valid 선택 Exact@5 (n) | test 선택 Exact@5 (n, 95% CI) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for name in tb["buckets"]:
        a, b = vb["buckets"][name], tb["buckets"][name]
        L.append(f"| {name} | {pct(a['share_selected'])} ({pct(a['share_all'])}) | {pct(b['share_selected'])} ({pct(b['share_all'])}) | "
                 f"{pct(a['bucket_coverage'])} | {pct(b['bucket_coverage'])} | "
                 f"{pct(a['exact_at_5_selected'])} ({a['n_selected']:,}) | "
                 f"{pct(b['exact_at_5_selected'])} ({b['n_selected']:,}, {ci(b['exact_at_5_selected_wilson95'])}) |")
    L.append("")
    L.append(f"그래프: `{s['outputs']['png']}` (PDF: `{s['outputs']['pdf']}`)\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    exp020 = json.loads(EXP020_JSON.read_text(encoding="utf-8"))
    thr = float(exp020["max_coverage"]["top1_score"]["exact_at_5>=90%"]["fine"]["threshold"])

    raw = load_predictions(args.predictions)
    missing = [c for c in ["top10_chars", "top10_scores", "label_char", "rank_of_gold"] if c not in raw.columns]
    if missing:
        raise SystemExit(f"[STOP] missing columns: {missing}")
    n = len(raw)
    conf = add_confidence(raw)

    stored = pd.to_numeric(raw["rank_of_gold"], errors="coerce").fillna(0).astype(int).to_numpy()
    stored = np.where(stored > 10, 0, stored)
    rank_mismatch = int((stored != conf["rank_recomputed"].to_numpy()).sum())
    overall = {f"exact_at_{k}": float(conf[f"hit_at_{k}"].mean()) for k in KS}
    stored_flags = {f"exact_at_{k}": float(raw[f"is_exact_at_{k}"].astype(bool).mean()) for k in KS
                    if f"is_exact_at_{k}" in raw.columns}
    matches = all(abs(overall[k] - v) < 1e-9 for k, v in OFFICIAL_EXP012.items())
    scores_all = np.concatenate([np.array([float(x) for x in _split(sc)]) for sc in raw["top10_scores"]])
    splits = sorted({part.split("=", 1)[1] for note in raw["notes"].dropna() for part in str(note).split("; ")
                     if part.startswith("split=")}) if "notes" in raw.columns else []
    sanity_notes = [
        f"행 수 {n:,} (기대 {EXPECTED_N:,}), notes의 split 값: {splits}",
        f"재계산 Exact@1/5/10: {overall['exact_at_1']:.6f} / {overall['exact_at_5']:.6f} / {overall['exact_at_10']:.6f} "
        f"vs EXP-012 공식 {OFFICIAL_EXP012['exact_at_1']:.6f} / {OFFICIAL_EXP012['exact_at_5']:.6f} / "
        f"{OFFICIAL_EXP012['exact_at_10']:.6f} → 일치 {matches}",
        f"저장된 is_exact_at_k 평균: {', '.join(f'{k}={v:.6f}' for k, v in stored_flags.items())}",
        f"저장된 rank_of_gold(≤10)와 재계산 rank 불일치 행: {rank_mismatch}",
        f"score 범위 [{scores_all.min():.6f}, {scores_all.max():.6f}], 샘플별 top-10 합 최대 "
        f"{conf['topk_mass'].max():.4f} / 중앙값 {conf['topk_mass'].median():.4f} → 확률(logit 아님), EXP-020과 동일 형식",
        f"후보 수 10 미만 행: {int((conf['n_candidates'] < 10).sum())}, 후보 0개 행: {int((conf['n_candidates'] == 0).sum())}",
    ]
    passed = (n == EXPECTED_N and matches and rank_mismatch == 0
              and scores_all.min() >= 0 and conf["topk_mass"].max() <= 1.0 + 1e-6)
    if not passed:
        print("\n".join(sanity_notes))
        raise SystemExit("[STOP] sanity check failed")

    vconf = pd.read_csv(EXP020_CONF, dtype={"sample_id": str})
    vfix = fixed_threshold_eval(vconf, thr)
    exp020_fine = exp020["max_coverage"]["top1_score"]["exact_at_5>=90%"]["fine"]
    if vfix["n_selected"] != exp020_fine["n_selected"]:
        raise SystemExit(f"[STOP] valid re-selection mismatch: {vfix['n_selected']} vs {exp020_fine['n_selected']}")
    tfix = fixed_threshold_eval(conf, thr)
    tfix_rounded = fixed_threshold_eval(conf, ROUNDED_THRESHOLD)

    vt = {f"exact_at_{k}": two_proportion_z(vfix[f"exact_at_{k}_hits"], vfix["n_selected"],
                                            tfix[f"exact_at_{k}_hits"], tfix["n_selected"]) for k in KS}
    vt["coverage"] = two_proportion_z(vfix["n_selected"], vfix["n_total"], tfix["n_selected"], tfix["n_total"])

    tcurve, tcum = curve(conf)
    vcurve, vcum = curve(vconf)

    vb = bucket_crosscheck(vconf.merge(pd.read_csv(args.mask_csv, usecols=["sample_id", "label_char"], dtype=str),
                                       on="sample_id", how="left"),
                           args.mask_csv, (vconf["top1_score"] >= thr).to_numpy(), "valid")
    conf["label_char"] = raw["label_char"].astype(str).str.strip()
    tb = bucket_crosscheck(conf, args.mask_csv, (conf["top1_score"] >= thr).to_numpy(), "test")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    png, pdf = out / "exp022_coverage_accuracy_valid_vs_test.png", out / "exp022_coverage_accuracy_valid_vs_test.pdf"
    plot_overlay(vcum, tcum, vfix, tfix, png, pdf)
    tcurve.to_csv(out / "exp022_test_coverage_accuracy.csv", index=False, encoding="utf-8")
    conf.drop(columns=["label_char"]).to_csv(out / "exp022_test_sample_confidence.csv", index=False, encoding="utf-8")

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    summary = {
        "experiment_id": "EXP-022",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": "notebooks/best/exp012_sillok_ft_full_predictions.csv",
        "mask_csv": rel(args.mask_csv),
        "exp020_json": rel(EXP020_JSON),
        "exp020_sample_confidence": rel(EXP020_CONF),
        "n_test": n,
        "threshold": thr,
        "threshold_source": "exp020_results.json max_coverage.top1_score['exact_at_5>=90%'].fine.threshold (valid-fixed, not re-tuned)",
        "sanity": {"passed": passed, "overall": overall, "official_exp012": OFFICIAL_EXP012,
                   "rank_mismatch_rows": rank_mismatch, "notes": sanity_notes},
        "fixed_threshold": {"valid": vfix, "test": tfix, "test_rounded_0617": tfix_rounded},
        "valid_vs_test": vt,
        "bucket_crosscheck": {"valid": vb, "test": tb},
        "outputs": {
            "png": rel(png), "pdf": rel(pdf),
            "test_curve_csv": rel(out / "exp022_test_coverage_accuracy.csv"),
            "test_sample_confidence_csv": rel(out / "exp022_test_sample_confidence.csv"),
            "md": rel(out / "exp022_results.md"), "json": rel(out / "exp022_results.json"),
        },
    }
    (out / "exp022_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                             encoding="utf-8")
    md = render_md(summary, tcurve, vcurve)
    (out / "exp022_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
