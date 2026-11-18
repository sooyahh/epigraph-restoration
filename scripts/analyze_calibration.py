"""EXP-034: calibration of SillokBERT top-1 confidence (post-processing only).

Question: when the fine-tuned model says its top-1 candidate has softmax score p,
is the candidate actually right about p of the time? If not, can a simple
calibrator fitted on validation data fix that on test without re-tuning?

Inputs (already in this repository, no model or raw data needed):
- valid: reports/exp020/exp020_sample_confidence.csv  (EXP-015 predictions, n=10,679)
- test:  reports/exp022/exp022_test_sample_confidence.csv (EXP-012 predictions, n=9,719)

Protocol:
1. Calibrators (Platt, isotonic, histogram binning) are compared by 5-fold
   cross-validated NLL on valid only; the winner is chosen before test is read.
2. Every calibrator is fitted on all of valid and applied unchanged to test.
3. Test is used once, for reporting. Nothing is re-tuned on test.

Usage:
    python scripts/analyze_calibration.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.calibration import (  # noqa: E402
    HistogramCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    summarize,
)

VALID_CSV = ROOT / "reports/exp020/exp020_sample_confidence.csv"
TEST_CSV = ROOT / "reports/exp022/exp022_test_sample_confidence.csv"
EXP020_JSON = ROOT / "reports/exp020/exp020_results.json"
OUT = ROOT / "reports/exp034"
SEED = 42
N_BINS = 15
N_BOOT = 1000

CALIBRATORS = {
    "platt": lambda: PlattCalibrator(),
    "isotonic": lambda: IsotonicCalibrator(),
    "histogram": lambda: HistogramCalibrator(n_bins=N_BINS),
}


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    hit1 = df["hit_at_1"].astype(str).str.lower().eq("true").to_numpy().astype(int)
    hit5 = df["hit_at_5"].astype(str).str.lower().eq("true").to_numpy().astype(int)
    return df["top1_score"].to_numpy(float), hit1, df["topk_mass"].to_numpy(float), hit5


def cv_nll(prob: np.ndarray, hit: np.ndarray, make, k: int = 5) -> float:
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(prob.size), k)
    losses = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        cal = make().fit(prob[tr], hit[tr])
        losses.append(negative_log_likelihood(cal.predict(prob[te]), hit[te]))
    return float(np.mean(losses))


def boot_ece_diff(raw: np.ndarray, cal: np.ndarray, hit: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, hit.size, hit.size)
        diffs.append(expected_calibration_error(cal[idx], hit[idx], N_BINS)
                     - expected_calibration_error(raw[idx], hit[idx], N_BINS))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}


def plot(valid_raw, valid_hit, test_raw, test_cal, test_hit, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    panels = [
        ("Before: raw top-1 softmax", [("valid", valid_raw, valid_hit, "#9aa5b1"), ("test", test_raw, test_hit, "#2f6db5")]),
        ("After: Platt fitted on valid, applied to test", [("test raw", test_raw, test_hit, "#9aa5b1"), ("test calibrated", test_cal, test_hit, "#c2410c")]),
    ]
    for ax, (title, series) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#555", label="perfect calibration")
        for label, p, y, color in series:
            bins = reliability_bins(p, y, N_BINS, "quantile")
            ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins],
                    marker="o", ms=4, lw=1.6, color=color,
                    label=f"{label} (ECE {expected_calibration_error(p, y, N_BINS) * 100:.1f}%)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("stated confidence (top-1)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("observed Exact@1")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_md(s: dict) -> str:
    raw_v, raw_t = s["raw"]["valid"], s["raw"]["test"]
    sel = s["selected_calibrator"]
    cal_t = s["calibrated_test"][sel]
    lines = [
        "# EXP-034 — top-1 확신도 보정(calibration) 분석",
        "",
        f"- 생성: {s['created_at']}",
        "- 유형: 후처리 분석 (모델 로딩·재추론 없음)",
        "- 입력: valid `reports/exp020/exp020_sample_confidence.csv` (n=10,679), test `reports/exp022/exp022_test_sample_confidence.csv` (n=9,719)",
        "- 원칙: 보정기 선택은 valid 5-fold 교차검증 NLL로만 결정, valid 전체로 학습한 뒤 test에는 그대로 한 번 적용",
        "",
        "## 1. 보정 전: 모델은 실제보다 자신 있게 말한다",
        "",
        f"| split | 평균 확신도 | 실제 Exact@1 | ECE (15 bins) | Brier |",
        "|---|---:|---:|---:|---:|",
        f"| valid | {pct(raw_v['mean_confidence'])} | {pct(raw_v['accuracy'])} | {pct(raw_v['ece_uniform'])} | {raw_v['brier']:.4f} |",
        f"| test | {pct(raw_t['mean_confidence'])} | {pct(raw_t['accuracy'])} | {pct(raw_t['ece_uniform'])} | {raw_t['brier']:.4f} |",
        "",
        "test 구간별 (확신도 10% 단위):",
        "",
        "| 확신도 구간 | n | 평균 확신도 | 실제 Exact@1 | 차이 |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in s["raw_test_bins_10"]:
        lines.append(f"| {b['lower']:.1f}–{b['upper']:.1f} | {b['n']} | {pct(b['mean_confidence'])} | {pct(b['accuracy'])} | {(b['accuracy'] - b['mean_confidence']) * 100:+.1f}%p |")
    lines += [
        "",
        "## 2. 보정기 선택 (valid만 사용)",
        "",
        "| 보정기 | valid 5-fold CV NLL |",
        "|---|---:|",
    ]
    for name, v in s["valid_cv_nll"].items():
        mark = " ← 선택" if name == sel else ""
        lines.append(f"| {name} | {v:.4f}{mark} |")
    lines += [
        f"| (보정 안 함) | {s['raw']['valid']['nll']:.4f} |",
        "",
        "## 3. test 결과 (valid로 학습한 보정기를 그대로 적용)",
        "",
        "| 방법 | 평균 확신도 | ECE (15 bins) | ECE (동일 개수 bins) | Brier | NLL |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 보정 안 함 | {pct(raw_t['mean_confidence'])} | {pct(raw_t['ece_uniform'])} | {pct(raw_t['ece_quantile'])} | {raw_t['brier']:.4f} | {raw_t['nll']:.4f} |",
    ]
    for name, m in s["calibrated_test"].items():
        lines.append(f"| {name}{' (선택)' if name == sel else ''} | {pct(m['mean_confidence'])} | {pct(m['ece_uniform'])} | {pct(m['ece_quantile'])} | {m['brier']:.4f} | {m['nll']:.4f} |")
    d = s["ece_diff_bootstrap"]
    lines += [
        "",
        f"- 선택한 보정기({sel})의 test ECE 변화: {d['mean'] * 100:+.2f}%p (bootstrap 95% CI {d['ci95'][0] * 100:+.2f} ~ {d['ci95'][1] * 100:+.2f}%p, {N_BOOT}회)",
        f"- test 실제 Exact@1은 {pct(raw_t['accuracy'])}이며, 보정 후 평균 확신도는 {pct(cal_t['mean_confidence'])}로 가까워졌다.",
        "",
        "![reliability diagram](exp034_reliability_valid_test.png)",
        "",
        "## 4. 기권 규칙과의 관계",
        "",
        f"- 보정은 점수의 순서를 바꾸지 않는 단조 변환이라, EXP-020/022의 기권 규칙이 고르는 표본(top-1 ≥ {s['exp020_threshold']:.6f})은 그대로다. 제안 비율과 정확도는 바뀌지 않는다.",
        f"- 바뀌는 것은 화면에 보여 줄 숫자다. 원래 점수 {s['exp020_threshold']:.3f}는 보정 후 약 {pct(s['threshold_calibrated'])}로 읽어야 한다.",
        f"- 즉 \"모델 점수 62%\"를 그대로 보여 주면 사용자는 실제보다 높게 믿게 된다. 보정값을 보여 주면 \"이 후보가 맞을 확률\"에 더 가까운 숫자를 줄 수 있다.",
        "",
        "## 5. (보조) 상위 5개 후보 안에 정답이 있을 확률",
        "",
        f"- 상위 10개 후보의 확률 합(`topk_mass`)을 그대로 Exact@5 확률로 읽으면 test ECE {pct(s['hit5']['raw_test']['ece_uniform'])}로 크게 어긋난다.",
        f"- valid로 학습한 Platt 보정을 적용하면 test ECE {pct(s['hit5']['platt_test']['ece_uniform'])}가 된다.",
        "",
        "## 6. 한계",
        "",
        "- 저장된 예측에는 top-10 확률만 있어 전체 어휘 logit이 필요한 temperature scaling은 하지 않았다. Platt 보정은 top-1 점수 하나에 대한 2-파라미터 보정이다.",
        "- 인공 single-mask 표본 기준이다. 실제 결손(정답 없음)에서의 보정 상태는 확인할 수 없다.",
        "- valid(EXP-015)와 test(EXP-012) 예측은 같은 체크포인트(`exp012_sillokbert_gpu_best_valid_loss`)에서 나왔다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vp, vh, vmass, vh5 = load(VALID_CSV)

    # Step 1: choose the calibrator on valid only.
    cv = {name: cv_nll(vp, vh, make) for name, make in CALIBRATORS.items()}
    selected = min(cv, key=cv.get)
    fitted = {name: make().fit(vp, vh) for name, make in CALIBRATORS.items()}

    # Step 2: read test once and apply unchanged.
    tp, th, tmass, th5 = load(TEST_CSV)
    calibrated = {name: summarize(cal.predict(tp), th, N_BINS) for name, cal in fitted.items()}
    test_sel = fitted[selected].predict(tp)

    exp020 = json.loads(EXP020_JSON.read_text(encoding="utf-8"))
    thr_val = float(exp020["max_coverage"]["top1_score"]["exact_at_5>=90%"]["fine"]["threshold"])

    hit5_cal = PlattCalibrator().fit(vmass, vh5)
    summary = {
        "experiment_id": "EXP-034",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"valid": str(VALID_CSV.relative_to(ROOT)), "test": str(TEST_CSV.relative_to(ROOT))},
        "n_bins": N_BINS,
        "raw": {"valid": summarize(vp, vh, N_BINS), "test": summarize(tp, th, N_BINS)},
        "raw_test_bins_10": reliability_bins(tp, th, 10, "uniform"),
        "valid_cv_nll": cv,
        "selected_calibrator": selected,
        "selected_params": {"a": fitted["platt"].a, "b": fitted["platt"].b} if selected == "platt" else None,
        "calibrated_test": calibrated,
        "ece_diff_bootstrap": boot_ece_diff(tp, test_sel, th),
        "exp020_threshold": thr_val,
        "threshold_calibrated": float(fitted[selected].predict([thr_val])[0]),
        "hit5": {
            "raw_test": summarize(tmass, th5, N_BINS),
            "platt_test": summarize(hit5_cal.predict(tmass), th5, N_BINS),
        },
    }
    (OUT / "exp034_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "exp034_results.md").write_text(render_md(summary), encoding="utf-8")
    plot(vp, vh, tp, test_sel, th, OUT / "exp034_reliability_valid_test.png")
    print(json.dumps({k: summary[k] for k in ("selected_calibrator", "valid_cv_nll", "ece_diff_bootstrap", "threshold_calibrated")}, indent=2))


if __name__ == "__main__":
    main()
