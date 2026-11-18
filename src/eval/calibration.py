"""Confidence calibration metrics and post-hoc calibrators (numpy only).

The model's top-1 softmax value is a score, not a calibrated probability.
These helpers measure how far the score is from the observed accuracy and fit
simple calibrators on validation data so they can be applied unchanged to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _as_arrays(prob, hit) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    y = np.asarray(hit, dtype=float)
    if p.shape != y.shape:
        raise ValueError("prob and hit must have the same shape")
    if p.size == 0:
        raise ValueError("empty input")
    return p, y


def reliability_bins(prob, hit, n_bins: int = 15, strategy: str = "uniform") -> list[dict]:
    """Group predictions into confidence bins and report mean confidence vs accuracy.

    strategy="uniform": equal-width bins on [0, 1].
    strategy="quantile": bins with (almost) equal numbers of samples.
    Empty bins are omitted.
    """
    p, y = _as_arrays(prob, hit)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "n": n,
            "mean_confidence": float(p[mask].mean()),
            "accuracy": float(y[mask].mean()),
        })
    return out


def expected_calibration_error(prob, hit, n_bins: int = 15, strategy: str = "uniform") -> float:
    """Sample-weighted mean |accuracy - confidence| over bins."""
    p, _ = _as_arrays(prob, hit)
    bins = reliability_bins(prob, hit, n_bins=n_bins, strategy=strategy)
    return float(sum(b["n"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins) / p.size)


def brier_score(prob, hit) -> float:
    p, y = _as_arrays(prob, hit)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(prob, hit) -> float:
    p, y = _as_arrays(prob, hit)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    """Logistic regression on logit(score): calibrated = sigmoid(a * logit(p) + b)."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, prob, hit, n_iter: int = 100) -> "PlattCalibrator":
        p, y = _as_arrays(prob, hit)
        x = _logit(p)
        a, b = 1.0, 0.0
        for _ in range(n_iter):  # Newton-Raphson
            z = np.clip(a * x + b, -35, 35)
            q = 1.0 / (1.0 + np.exp(-z))
            w = q * (1 - q) + 1e-12
            g = np.array([np.sum((q - y) * x), np.sum(q - y)])
            h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
            step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
            a, b = a - step[0], b - step[1]
            if np.max(np.abs(step)) < 1e-10:
                break
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, prob) -> np.ndarray:
        x = _logit(np.clip(np.asarray(prob, dtype=float), 0.0, 1.0))
        return 1.0 / (1.0 + np.exp(-np.clip(self.a * x + self.b, -35, 35)))


@dataclass
class IsotonicCalibrator:
    """Monotone non-decreasing step function fitted with pool-adjacent-violators."""

    x_: np.ndarray | None = None
    y_: np.ndarray | None = None

    def fit(self, prob, hit) -> "IsotonicCalibrator":
        p, y = _as_arrays(prob, hit)
        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]
        # Collapse ties so equal scores get one value.
        ux, inv = np.unique(xs, return_inverse=True)
        sums = np.bincount(inv, weights=ys)
        cnts = np.bincount(inv).astype(float)
        vals, wts, starts = [], [], []
        for i in range(len(ux)):
            vals.append(sums[i] / cnts[i])
            wts.append(cnts[i])
            starts.append(i)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2 = vals.pop(), wts.pop()
                starts.pop()
                v1, w1 = vals.pop(), wts.pop()
                vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
                wts.append(w1 + w2)
        fitted = np.empty(len(ux))
        bounds = starts + [len(ux)]
        for v, s, e in zip(vals, bounds[:-1], bounds[1:]):
            fitted[s:e] = v
        self.x_, self.y_ = ux, fitted
        return self

    def predict(self, prob) -> np.ndarray:
        if self.x_ is None:
            raise RuntimeError("fit before predict")
        p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
        return np.interp(p, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


@dataclass
class HistogramCalibrator:
    """Replace each score with the validation accuracy of its equal-mass bin."""

    n_bins: int = 15
    edges_: np.ndarray | None = None
    values_: np.ndarray | None = None

    def fit(self, prob, hit) -> "HistogramCalibrator":
        p, y = _as_arrays(prob, hit)
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, self.n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        values = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(edges) - 1)])
        # Fill empty bins from neighbours so predict never returns NaN.
        if np.isnan(values).any():
            good = ~np.isnan(values)
            values = np.interp(np.arange(len(values)), np.flatnonzero(good), values[good])
        self.edges_, self.values_ = edges, values
        return self

    def predict(self, prob) -> np.ndarray:
        if self.edges_ is None:
            raise RuntimeError("fit before predict")
        p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
        idx = np.clip(np.searchsorted(self.edges_, p, side="right") - 1, 0, len(self.edges_) - 2)
        return self.values_[idx]


def summarize(prob, hit, n_bins: int = 15) -> dict:
    p, y = _as_arrays(prob, hit)
    return {
        "n": int(p.size),
        "mean_confidence": float(p.mean()),
        "accuracy": float(y.mean()),
        "ece_uniform": expected_calibration_error(p, y, n_bins, "uniform"),
        "ece_quantile": expected_calibration_error(p, y, n_bins, "quantile"),
        "brier": brier_score(p, y),
        "nll": negative_log_likelihood(p, y),
    }
