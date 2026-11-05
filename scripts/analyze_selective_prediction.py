"""EXP-020: selective prediction (coverage-accuracy) analysis on EXP-015 valid predictions.

Post-processing only. No model loading, no re-inference, no test split.

Usage:
    python scripts/analyze_selective_prediction.py
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

from src.eval.stratified_analysis import FREQ_BUCKETS, _freq_bucket  # noqa: E402

DEFAULT_PRED = ROOT / "outputs/predictions/exp015_sillok_ft_valid_predictions.csv"
DEFAULT_MASK = ROOT / "data/model_inputs/artificial_mask_dataset.csv"
DEFAULT_OUT = ROOT / "reports/exp020"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansKR-Regular.ttf")

REQUIRED_COLUMNS = ["sample_id", "label_char", "top10_chars", "top10_scores"]
COVERAGES = [round(0.05 * i, 2) for i in range(1, 21)]
TARGETS = {
    "exact_at_5": [0.90, 0.80, 0.70],
    "exact_at_1": [0.90, 0.80],
    "exact_at_10": [0.90],
}
KS = (1, 5, 10)

METRICS = {
    "top1_score": "top-1 확률",
    "margin": "top-1 − top-2 확률 차",
    "entropy": "top-10 재정규화 엔트로피",
}
SUPPLEMENTARY = {"entropy_softmax_literal": "top-10 score에 softmax 재적용한 엔트로피 (참고)"}


def _split(raw: object) -> list[str]:
    if not isinstance(raw, str):
        return []
    return [part.strip() for part in raw.split(";") if part.strip()]


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[STOP] prediction CSV not found: {path}")
    df = pd.read_csv(path, dtype={"label_char": str, "top10_chars": str, "top10_scores": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[STOP] missing columns: {missing}")
    return df


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else float("inf")


def add_confidence(df: pd.DataFrame) -> pd.DataFrame:
    chars = df["top10_chars"].map(_split)
    scores = df["top10_scores"].map(lambda s: [float(x) for x in _split(s)])
    gold = df["label_char"].astype(str).str.strip()

    rank = []
    for g, cs in zip(gold, chars):
        rank.append(cs.index(g) + 1 if g in cs else 0)
    rank = np.array(rank)
    out = df[["sample_id", "label_char"]].copy()
    out["rank_recomputed"] = rank
    for k in KS:
        out[f"hit_at_{k}"] = (rank >= 1) & (rank <= k)

    top1, margin, ent, ent_lit, n_cand, mass = [], [], [], [], [], []
    for sc in scores:
        arr = np.array(sc, dtype=float)
        n_cand.append(arr.size)
        mass.append(float(arr.sum()) if arr.size else 0.0)
        if arr.size == 0:
            top1.append(-np.inf)
            margin.append(-np.inf)
            ent.append(np.inf)
            ent_lit.append(np.inf)
            continue
        top1.append(arr[0])
        margin.append(arr[0] - (arr[1] if arr.size > 1 else 0.0))
        ent.append(entropy(arr / arr.sum()) if arr.sum() > 0 else np.inf)
        z = np.exp(arr - arr.max())
        ent_lit.append(entropy(z / z.sum()))
    out["n_candidates"] = n_cand
    out["topk_mass"] = mass
    out["top1_score"] = top1
    out["margin"] = margin
    out["entropy"] = ent
    out["entropy_softmax_literal"] = ent_lit
    return out


def confidence_order(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Indices sorted most-confident first; ties broken by sample_id for determinism."""
    key = df[metric].to_numpy()
    if metric.startswith("entropy"):
        key = -key
    tie = df["sample_id"].astype(str).to_numpy()
    return np.lexsort((tie, -key))


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def curve_table(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    order = confidence_order(df, metric)
    n = len(df)
    cum = {}
    for k in KS:
        hits = df[f"hit_at_{k}"].to_numpy()[order].astype(int)
        cum[k] = np.cumsum(hits)
    conf_sorted = df[metric].to_numpy()[order]
    rows = []
    for c in COVERAGES:
        m = max(1, int(round(c * n)))
        row = {
            "metric": metric,
            "coverage": c,
            "n_selected": m,
            "threshold": float(conf_sorted[m - 1]),
        }
        for k in KS:
            row[f"exact_at_{k}"] = cum[k][m - 1] / m
        rows.append(row)
    return pd.DataFrame(rows), {"order": order, **{f"cum{k}": cum[k] for k in KS}, "conf": conf_sorted}


def max_coverage(cum: np.ndarray, conf: np.ndarray, target: float, n: int, grid: bool) -> dict:
    """Largest coverage whose selected-prefix accuracy >= target.

    grid=True evaluates only the 5% coverage grid; grid=False scans every prefix size
    but requires the cut to fall on a confidence-value boundary (ties are not split).
    """
    sizes = np.arange(1, n + 1)
    acc = cum / sizes
    if grid:
        cand = [max(1, int(round(c * n))) for c in COVERAGES]
    else:
        boundary = np.append(conf[:-1] != conf[1:], True)
        cand = list(sizes[boundary])
    ok = [m for m in cand if acc[m - 1] >= target]
    if not ok:
        best = int(np.argmax(acc[[m - 1 for m in cand]]))
        m = cand[best]
        return {
            "reached": False,
            "best_accuracy": float(acc[m - 1]),
            "best_coverage": m / n,
            "best_n": int(m),
        }
    m = max(ok)
    lo, hi = wilson(int(cum[m - 1]), int(m))
    return {
        "reached": True,
        "coverage": m / n,
        "n_selected": int(m),
        "accuracy": float(acc[m - 1]),
        "wilson95": [lo, hi],
        "threshold": float(conf[m - 1]),
    }


def aurc_like(cum: np.ndarray, n: int) -> float:
    """Mean selected-prefix accuracy over all coverages (higher = better curve)."""
    return float((cum / np.arange(1, n + 1)).mean())


def bucket_crosscheck(
    conf_df: pd.DataFrame, mask_csv: Path, order: np.ndarray, m: int
) -> dict:
    mask = pd.read_csv(mask_csv, usecols=["sample_id", "split", "label_char"], dtype=str)
    counts = Counter(mask.loc[mask["split"] == "train", "label_char"].astype(str))
    valid_ids = set(mask.loc[mask["split"] == "valid", "sample_id"])
    not_valid = int((~conf_df["sample_id"].isin(valid_ids)).sum())
    buckets = conf_df["label_char"].astype(str).map(lambda ch: _freq_bucket(counts.get(ch, 0)))
    selected = np.zeros(len(conf_df), dtype=bool)
    selected[order[:m]] = True

    result = {"n_total": len(conf_df), "n_selected": int(m), "rows_not_in_valid_split": not_valid, "buckets": {}}
    for name, _ in FREQ_BUCKETS:
        in_b = (buckets == name).to_numpy()
        sel_b = in_b & selected
        n_b, n_sel = int(in_b.sum()), int(sel_b.sum())
        entry = {
            "n_all": n_b,
            "share_all": n_b / len(conf_df),
            "n_selected": n_sel,
            "share_selected": n_sel / m if m else 0.0,
            "bucket_coverage": n_sel / n_b if n_b else 0.0,
        }
        for k in KS:
            hits = conf_df[f"hit_at_{k}"].to_numpy()
            entry[f"exact_at_{k}_all"] = float(hits[in_b].mean()) if n_b else None
            entry[f"exact_at_{k}_selected"] = float(hits[sel_b].mean()) if n_sel else None
        result["buckets"][name] = entry
    return result


def plot_curve(cum: dict, n: int, metric: str, cov90: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    if not FONT_PATH.exists():
        raise SystemExit(f"[STOP] Korean font not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 15})

    sizes = np.arange(1, n + 1)
    x = sizes / n * 100
    start = max(1, int(0.01 * n)) - 1
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    styles = {1: ("#9aa5b1", "Exact@1"), 5: ("#c0392b", "Exact@5"), 10: ("#2c3e50", "Exact@10")}
    for k in KS:
        color, label = styles[k]
        ax.plot(x[start:], (cum[f"cum{k}"] / sizes * 100)[start:], color=color,
                lw=3.2 if k == 5 else 2.2, label=label)

    ax.axhline(90, color="#c0392b", ls="--", lw=1.4, alpha=0.8)
    ax.text(99, 91, "Exact@5 = 90%", color="#c0392b", ha="right", va="bottom", fontsize=13)
    if cov90.get("reached"):
        cx = cov90["coverage"] * 100
        ax.axvline(cx, color="#c0392b", ls=":", lw=1.4)
        ax.scatter([cx], [cov90["accuracy"] * 100], color="#c0392b", zorder=5, s=60)
        ax.annotate(
            f"커버리지 {cx:.1f}% (n={cov90['n_selected']:,})\nExact@5 {cov90['accuracy'] * 100:.1f}%",
            xy=(cx, cov90["accuracy"] * 100), xytext=(cx + 8, 25),
            fontsize=14, color="#c0392b",
            arrowprops={"arrowstyle": "->", "color": "#c0392b"},
        )
    ax.scatter([100], [cum["cum5"][-1] / n * 100], color="#c0392b", zorder=5, s=40)
    ax.annotate(f"전체 {cum['cum5'][-1] / n * 100:.1f}%", xy=(100, cum["cum5"][-1] / n * 100),
                xytext=(88, cum["cum5"][-1] / n * 100 - 9), fontsize=12, color="#c0392b")

    ax.set_xlim(0, 101)
    ax.set_ylim(0, 102)
    ax.set_xlabel("커버리지 (모델이 후보를 제시하는 결손 비율, %)")
    ax.set_ylabel("정확도 (%)")
    ax.set_title(f"선택적 예측: 확신도({METRICS[metric]}) 상위 구간의 정확도  ·  valid n={n:,}", fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    fig.text(0.99, 0.01, "EXP-020 · EXP-015 valid 예측 후처리 (재추론 없음)", ha="right", fontsize=10, color="#666")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def render_md(summary: dict, curves: pd.DataFrame, bucket: dict) -> str:
    L: list[str] = []
    L.append("# EXP-020 — 선택적 예측(coverage-accuracy) 분석 (EXP-015 valid 후처리)\n")
    L.append(f"- 생성: {summary['generated_at']}")
    L.append(f"- 입력: `{summary['input_csv']}` (n={summary['n']:,}, valid only; test 미사용; 재추론 없음)")
    L.append(f"- 재계산 전체 Exact@1/5/10: {pct(summary['overall']['exact_at_1'])} / "
             f"{pct(summary['overall']['exact_at_5'])} / {pct(summary['overall']['exact_at_10'])} "
             f"(EXP-015 공식값과 일치 여부: {summary['overall']['matches_exp015']})\n")

    L.append("## 1. 입력 확인\n")
    for line in summary["input_check"]:
        L.append(f"- {line}")
    L.append("")

    L.append("## 2. 지표별 coverage-accuracy (10% 단위)\n")
    for metric in list(METRICS) + list(SUPPLEMENTARY):
        label = METRICS.get(metric) or SUPPLEMENTARY[metric]
        L.append(f"### {metric} — {label}\n")
        L.append("| 커버리지 | n | 임계값 | Exact@1 | Exact@5 | Exact@10 |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        sub = curves[curves["metric"] == metric]
        for _, r in sub.iterrows():
            if round(r["coverage"] * 100) % 10 == 0:
                L.append(f"| {r['coverage'] * 100:.0f}% | {r['n_selected']:,} | {r['threshold']:.4f} | "
                         f"{pct(r['exact_at_1'])} | {pct(r['exact_at_5'])} | {pct(r['exact_at_10'])} |")
        L.append("")

    L.append("## 3. 목표 정확도별 최대 커버리지\n")
    L.append("샘플 단위 스캔(동점 confidence는 분할하지 않음). 괄호는 5% 격자 기준.\n")
    L.append("| 지표 | 기준 | 최대 커버리지 | n | 정확도 (95% Wilson) | 임계값 |")
    L.append("|---|---|--:|--:|--:|--:|")
    for metric in list(METRICS) + list(SUPPLEMENTARY):
        for key, res in summary["max_coverage"][metric].items():
            fine, grid = res["fine"], res["grid"]
            g = f"{grid['coverage'] * 100:.0f}%" if grid["reached"] else "도달 불가"
            if fine["reached"]:
                lo, hi = fine["wilson95"]
                L.append(f"| {metric} | {key} | **{fine['coverage'] * 100:.1f}%** ({g}) | {fine['n_selected']:,} | "
                         f"{pct(fine['accuracy'])} ({pct(lo)}–{pct(hi)}) | {fine['threshold']:.4f} |")
            else:
                L.append(f"| {metric} | {key} | 도달 불가 (최대 {pct(fine['best_accuracy'])} @ "
                         f"{fine['best_coverage'] * 100:.1f}%) | - | - | - |")
    L.append("")
    L.append("곡선 전체 평균 정확도(AURC 보수, 높을수록 좋음):\n")
    L.append("| 지표 | 평균 Exact@1 | 평균 Exact@5 | 평균 Exact@10 |")
    L.append("|---|--:|--:|--:|")
    for metric, v in summary["curve_mean_accuracy"].items():
        L.append(f"| {metric} | {pct(v['exact_at_1'])} | {pct(v['exact_at_5'])} | {pct(v['exact_at_10'])} |")
    L.append(f"\n최적 지표: **{summary['best_metric']}** (평균 Exact@5 기준)\n")

    L.append("## 4. 빈도 구간 교차 확인\n")
    L.append(f"기준: {summary['bucket_point']['description']} — 선택 n={bucket['n_selected']:,} / 전체 {bucket['n_total']:,}\n")
    L.append("| 구간 | 전체 n | 전체 비중 | 선택 n | 선택 비중 | 구간 내 선택률 | 전체 Exact@5 | 선택 Exact@5 |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for name, b in bucket["buckets"].items():
        L.append(f"| {name} | {b['n_all']:,} | {pct(b['share_all'])} | {b['n_selected']:,} | {pct(b['share_selected'])} | "
                 f"{pct(b['bucket_coverage'])} | {pct(b['exact_at_5_all'])} | {pct(b['exact_at_5_selected'])} |")
    L.append("")
    L.append(f"그래프: `{summary['outputs']['png']}`\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    raw = load_predictions(args.predictions)
    n = len(raw)
    conf = add_confidence(raw)

    scores_all = np.concatenate([np.array([float(x) for x in _split(s)]) for s in raw["top10_scores"]])
    input_check = [
        f"행 수 {n:,}; 컬럼: {', '.join(raw.columns)}",
        "top-k 후보 `top10_chars`(;구분)와 대응 score `top10_scores` 사용",
        f"score 범위 [{scores_all.min():.6f}, {scores_all.max():.6f}], 샘플별 top-10 합 최대 "
        f"{conf['topk_mass'].max():.4f} / 중앙값 {conf['topk_mass'].median():.4f} → **확률(logit 아님)**",
        "출처: `src/eval/predictors.py` `torch.softmax(token_logits)` — 전체 vocab softmax 확률. "
        "단일 한자 토큰만 남기도록 `##`·다중문자 토큰은 건너뛰므로 top-2는 '두 번째 단일 한자 후보'",
        "score는 소수 6자리 반올림 저장",
        f"후보 수 10 미만 행: {int((conf['n_candidates'] < 10).sum())}, 후보 0개 행: {int((conf['n_candidates'] == 0).sum())} "
        "(0개는 확신 최하위로 처리)",
        "entropy: score가 이미 확률이므로 softmax 재적용 대신 top-10 확률을 합 1로 재정규화해 계산(주 지표). "
        "지시문 그대로의 softmax(score) 엔트로피는 `entropy_softmax_literal`로 참고 병기. 둘 다 top-10만 쓴 근사치",
    ]
    rank_mismatch = None
    if "rank_of_gold" in raw.columns:
        stored = pd.to_numeric(raw["rank_of_gold"], errors="coerce").fillna(0).astype(int).to_numpy()
        stored = np.where(stored > 10, 0, stored)
        rank_mismatch = int((stored != conf["rank_recomputed"].to_numpy()).sum())
        input_check.append(f"저장된 `rank_of_gold`(≤10)와 재계산 rank 불일치 행: {rank_mismatch}")

    overall = {f"exact_at_{k}": float(conf[f"hit_at_{k}"].mean()) for k in KS}
    official = {"exact_at_1": 0.3377657083996629, "exact_at_5": 0.532353216593314, "exact_at_10": 0.608952149077629}
    overall["matches_exp015"] = all(abs(overall[k] - v) < 1e-9 for k, v in official.items())

    all_metrics = list(METRICS) + list(SUPPLEMENTARY)
    tables, cums = [], {}
    for metric in all_metrics:
        t, c = curve_table(conf, metric)
        tables.append(t)
        cums[metric] = c
    curves = pd.concat(tables, ignore_index=True)

    maxcov: dict = {}
    mean_acc: dict = {}
    for metric in all_metrics:
        c = cums[metric]
        maxcov[metric] = {}
        for acc_key, targets in TARGETS.items():
            k = int(acc_key.split("_")[-1])
            for t in targets:
                maxcov[metric][f"{acc_key}>={int(t * 100)}%"] = {
                    "fine": max_coverage(c[f"cum{k}"], c["conf"], t, n, grid=False),
                    "grid": max_coverage(c[f"cum{k}"], c["conf"], t, n, grid=True),
                }
        mean_acc[metric] = {f"exact_at_{k}": aurc_like(c[f"cum{k}"], n) for k in KS}

    best = max(METRICS, key=lambda m: mean_acc[m]["exact_at_5"])
    cov90 = maxcov[best]["exact_at_5>=90%"]["fine"]
    if cov90["reached"]:
        m_sel = cov90["n_selected"]
        desc = f"{best} 기준 Exact@5≥90% 최대 커버리지 {cov90['coverage'] * 100:.1f}%"
    else:
        m_sel = int(round(0.30 * n))
        desc = f"{best} 기준 Exact@5 90% 도달 불가 → 커버리지 30% 지점"
    bucket = bucket_crosscheck(conf, args.mask_csv, cums[best]["order"], m_sel)

    m30 = int(round(0.30 * n))
    bucket30 = bucket_crosscheck(conf, args.mask_csv, cums[best]["order"], m30) if m30 != m_sel else bucket

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"exp020_coverage_accuracy_{best}.png"
    plot_curve(cums[best], n, best, cov90, png)

    curves_csv = out / "exp020_coverage_accuracy.csv"
    curves.to_csv(curves_csv, index=False, encoding="utf-8")
    conf_csv = out / "exp020_sample_confidence.csv"
    conf.drop(columns=["label_char"]).to_csv(conf_csv, index=False, encoding="utf-8")

    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    summary = {
        "experiment_id": "EXP-020",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": rel(args.predictions),
        "mask_csv": rel(args.mask_csv),
        "n": n,
        "split": "valid",
        "score_type": "probability (full-vocab softmax, single-char filtered top-10, 6dp)",
        "input_check": input_check,
        "rank_mismatch_rows": rank_mismatch,
        "overall": overall,
        "max_coverage": maxcov,
        "curve_mean_accuracy": mean_acc,
        "best_metric": best,
        "bucket_point": {"description": desc, "n_selected": m_sel},
        "bucket_crosscheck": bucket,
        "bucket_crosscheck_cov30": bucket30,
        "outputs": {
            "curves_csv": rel(curves_csv),
            "sample_confidence_csv": rel(conf_csv),
            "png": rel(png),
            "md": rel(out / "exp020_results.md"),
            "json": rel(out / "exp020_results.json"),
        },
    }
    (out / "exp020_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    md = render_md(summary, curves, bucket)
    if bucket30 is not bucket:
        md += "\n### 참고: 커버리지 30% 지점 구간 분포\n\n"
        md += "| 구간 | 선택 n | 선택 비중 | 구간 내 선택률 | 선택 Exact@5 |\n|---|--:|--:|--:|--:|\n"
        for name, b in bucket30["buckets"].items():
            md += (f"| {name} | {b['n_selected']:,} | {pct(b['share_selected'])} | "
                   f"{pct(b['bucket_coverage'])} | {pct(b['exact_at_5_selected'])} |\n")
    (out / "exp020_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
