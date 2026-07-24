"""elasticity.py — estimate price elasticity per category (from-scratch ridge).

Constant-elasticity demand implies a log-log law: ln(units) = a + b*ln(price)
+ controls, where b is the price elasticity. We pool SKUs within a category
(more price variation, one shared slope) and fit a ridge regression **from
scratch in numpy** via the closed form  w = (XᵀX + λI)⁻¹ Xᵀy.

Controls: a per-SKU intercept (fixed effect, so the slope is identified from
*within*-SKU promo variation, not from cross-SKU level differences) and month
Fourier terms (k=1,2) to soak up seasonality. Only ln(price) is the elasticity.

We compare the recovered elasticity to the true generating value from the SKU
master and report the recovery error.

    estimate_elasticities(history) -> dict[sku, float]
    recovery_report(history) -> dict

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..dataio import load
from .forecast import load_history

LAMBDA = 1e-2                    # ridge penalty (small; intercepts barely biased)


def _fourier(cal_m: int) -> list[float]:
    ang = 2 * math.pi * (cal_m - 1) / 12
    return [math.sin(ang), math.cos(ang), math.sin(2 * ang), math.cos(2 * ang)]


def _ridge(X: np.ndarray, y: np.ndarray, lam: float,
           penalize: np.ndarray | None = None) -> np.ndarray:
    """Closed-form ridge  w = (XᵀX + λ P)⁻¹ Xᵀy. `penalize` is a 0/1 mask so we
    can leave chosen columns (e.g. fixed effects) lightly/un-penalized."""
    n_feat = X.shape[1]
    P = np.eye(n_feat) if penalize is None else np.diag(penalize)
    A = X.T @ X + lam * P
    return np.linalg.solve(A, X.T @ y)


def _fit_category(rows: list[dict], skus: list[str]) -> float:
    """Return the estimated elasticity (slope on ln price) for one category."""
    sku_index = {s: i for i, s in enumerate(skus)}
    n_sku = len(skus)

    Xrows, y = [], []
    for r in rows:
        if r["price"] <= 0 or r["units"] <= 0:
            continue
        fe = [0.0] * n_sku
        fe[sku_index[r["sku"]]] = 1.0                 # per-SKU intercept
        feat = [math.log(r["price"]), *fe, *_fourier(int(r["month"][5:7]))]
        Xrows.append(feat)
        y.append(math.log(r["units"]))

    X = np.asarray(Xrows, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    # Do not penalize the elasticity slope (col 0) or the SKU intercepts;
    # only lightly penalize the seasonal terms for numerical stability.
    penalize = np.zeros(X.shape[1])
    penalize[1 + n_sku:] = 1.0
    w = _ridge(X, yv, LAMBDA, penalize)
    return float(w[0])


def estimate_elasticities(history: dict[str, list[dict]] | None = None
                          ) -> dict[str, float]:
    """Pooled-per-category elasticity, assigned to every SKU in that category."""
    hist = history if history is not None else load_history()
    skus = load()
    cat_of = {s.sku: s.category for s in skus}

    by_cat_rows: dict[str, list[dict]] = defaultdict(list)
    by_cat_skus: dict[str, list[str]] = defaultdict(list)
    for sku, rows in hist.items():
        cat = cat_of.get(sku)
        if cat is None:
            continue
        by_cat_skus[cat].append(sku)
        for r in rows:
            by_cat_rows[cat].append({**r, "sku": sku})

    out: dict[str, float] = {}
    for cat, rows in by_cat_rows.items():
        cat_skus = sorted(by_cat_skus[cat])
        est = _fit_category(rows, cat_skus)
        # Guard against a degenerate positive slope (no promo variation): fall
        # back to the pooled sign convention (elasticities are negative).
        for s in cat_skus:
            out[s] = round(est, 4)
    return out


def recovery_report(history: dict[str, list[dict]] | None = None) -> dict:
    """Compare estimated (per-category) elasticity to the true generating one."""
    est = estimate_elasticities(history)
    skus = load()
    true_by_sku = {s.sku: s.elasticity for s in skus}

    # per-category true = mean of member SKUs' true elasticities
    cat_true: dict[str, list[float]] = defaultdict(list)
    cat_est: dict[str, float] = {}
    for s in skus:
        cat_true[s.category].append(true_by_sku[s.sku])
        cat_est[s.category] = est.get(s.sku, float("nan"))

    per_cat = {}
    errs = []
    for cat, trues in cat_true.items():
        t = float(np.mean(trues))
        e = cat_est[cat]
        per_cat[cat] = {"true_mean": round(t, 3), "estimated": round(e, 3),
                        "abs_error": round(abs(e - t), 3)}
        errs.append(abs(e - t))

    # per-SKU error (each SKU vs its own true elasticity, using cat estimate)
    sku_errs = [abs(est[s.sku] - true_by_sku[s.sku]) for s in skus if s.sku in est]
    return {
        "mae_by_category": round(float(np.mean(errs)), 4),
        "mae_by_sku": round(float(np.mean(sku_errs)), 4),
        "per_category": per_cat,
    }


if __name__ == "__main__":
    rep = recovery_report()
    print(f"[elasticity] recovery MAE (by category) = {rep['mae_by_category']}")
    print(f"[elasticity] recovery MAE (by SKU)       = {rep['mae_by_sku']}")
    for cat, d in sorted(rep["per_category"].items()):
        print(f"  {cat:<12} true={d['true_mean']:+.2f}  "
              f"est={d['estimated']:+.2f}  |err|={d['abs_error']:.2f}")
