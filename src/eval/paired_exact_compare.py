"""Paired Exact@k comparison (McNemar) between two prediction CSVs.

Use for iteration go/no-go on the same validation samples.
Do not treat models as independent proportions — join on sample_id and
compare discordant pairs only.

McNemar statistic uses continuity correction when b+c >= 1:
  chi2 = (abs(b - c) - 1)^2 / (b + c)
p-value is from chi-square with 1 df (scipy if available, else erfc approx).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _hit_column(k: int) -> str:
    if k not in (1, 5, 10):
        raise ValueError(f"k must be 1, 5, or 10 (got {k})")
    return f"is_exact_at_{k}"


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            1: True,
            0: False,
        }
    )
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:5]
        raise ValueError(f"Non-boolean hit values: {bad!r}")
    return mapped.astype(bool)


def mcnemar_chi2_continuity(b: int, c: int) -> float:
    """McNemar chi-square with continuity correction. Returns 0.0 if b+c==0."""
    disc = b + c
    if disc == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / disc


def chi2_sf_1df(chi2: float) -> float:
    """Survival function P(X > chi2) for chi-square with 1 degree of freedom."""
    if chi2 <= 0:
        return 1.0
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore

        return float(chi2_dist.sf(chi2, df=1))
    except Exception:
        # chi2_1 = Z^2; P(Z^2 > x) = erfc(sqrt(x/2))
        return math.erfc(math.sqrt(chi2 / 2.0))


def bootstrap_diff_ci(
    hit_a: list[bool],
    hit_b: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for (rate_b - rate_a). Returns (diff, lo, hi)."""
    import numpy as np

    a = np.asarray(hit_a, dtype=float)
    b = np.asarray(hit_b, dtype=float)
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    observed = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(b[idx].mean() - a[idx].mean())
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return observed, lo, hi


def paired_exact_compare(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    k: int = 5,
    label_a: str = "A",
    label_b: str = "B",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two prediction tables on shared sample_id with Exact@k hits."""
    col = _hit_column(k)
    for name, df in ((label_a, pred_a), (label_b, pred_b)):
        if "sample_id" not in df.columns:
            raise ValueError(f"{name}: missing sample_id")
        if col not in df.columns:
            raise ValueError(f"{name}: missing {col}")

    a = pred_a[["sample_id", col]].copy()
    b = pred_b[["sample_id", col]].copy()
    a[col] = _as_bool_series(a[col])
    b[col] = _as_bool_series(b[col])
    a = a.drop_duplicates(subset=["sample_id"], keep="first")
    b = b.drop_duplicates(subset=["sample_id"], keep="first")

    merged = a.merge(b, on="sample_id", how="inner", suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("No overlapping sample_id between prediction CSVs")

    hit_a = merged[f"{col}_a"].tolist()
    hit_b = merged[f"{col}_b"].tolist()
    both_true = sum(1 for x, y in zip(hit_a, hit_b) if x and y)
    both_false = sum(1 for x, y in zip(hit_a, hit_b) if (not x) and (not y))
    only_a = sum(1 for x, y in zip(hit_a, hit_b) if x and (not y))
    only_b = sum(1 for x, y in zip(hit_a, hit_b) if (not x) and y)
    n = len(merged)
    rate_a = sum(hit_a) / n
    rate_b = sum(hit_b) / n
    chi2 = mcnemar_chi2_continuity(only_a, only_b)
    p_value = chi2_sf_1df(chi2)
    diff, ci_lo, ci_hi = bootstrap_diff_ci(
        hit_a, hit_b, n_boot=n_boot, seed=seed
    )

    return {
        "k": k,
        "hit_column": col,
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": n,
        "n_a_only_rows": int(pred_a["sample_id"].nunique()),
        "n_b_only_rows": int(pred_b["sample_id"].nunique()),
        "both_correct": both_true,
        "both_incorrect": both_false,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        f"exact_at_{k}_a": rate_a,
        f"exact_at_{k}_b": rate_b,
        "diff_b_minus_a": diff,
        "diff_pp": diff * 100.0,
        "mcnemar": {
            "continuity_correction": True,
            "chi2": chi2,
            "p_value": p_value,
            "formula": "(abs(b-c)-1)^2 / (b+c) with b=only_a, c=only_b",
        },
        "bootstrap_diff_b_minus_a": {
            "n_boot": n_boot,
            "seed": seed,
            "alpha": 0.05,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        },
    }


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Baseline predictions CSV")
    parser.add_argument("--b", type=Path, required=True, help="Candidate predictions CSV")
    parser.add_argument("--k", type=int, default=5, choices=[1, 5, 10])
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the comparison JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = paired_exact_compare(
        load_predictions(args.a),
        load_predictions(args.b),
        k=args.k,
        label_a=args.label_a,
        label_b=args.label_b,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
