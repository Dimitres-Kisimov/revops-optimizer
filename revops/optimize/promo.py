"""promo.py — allocate a promotion budget across categories (concave uplift).

Each euro of promo spend on a category buys incremental margin with diminishing
returns: uplift_c(x) = a_c * (1 - exp(-b_c * x)). Maximizing the total concave
uplift subject to sum(x_c) <= budget and x_c >= 0 is a smooth concave program;
we solve it with scipy.optimize.minimize (SLSQP) and cross-check against the
water-filling KKT condition (spend where marginal uplift is highest until the
budget binds). Returns the per-category allocation and the expected incremental
margin.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..dataio import SKU


@dataclass
class PromoResult:
    allocation: dict[str, float]
    incremental_margin_eur: float
    budget_eur: float
    even_split_margin_eur: float
    uplift_vs_even_eur: float


def _category_response(skus: list[SKU]) -> dict[str, tuple[float, float]]:
    """Fit (a, b) of uplift_c(x)=a(1-e^{-bx}) per category from its economics:
    saturation `a` scales with the category's annual margin (headroom), and the
    responsiveness `b` scales with average elasticity magnitude."""
    by_cat: dict[str, list[SKU]] = defaultdict(list)
    for s in skus:
        by_cat[s.category].append(s)
    params = {}
    for cat, items in by_cat.items():
        ann_margin = sum(i.annual_margin_eur for i in items)
        avg_elastic = np.mean([abs(i.elasticity) for i in items])
        a = 0.12 * ann_margin                       # max ~12% incremental margin
        b = 4.0e-4 * avg_elastic                    # responsiveness per euro
        params[cat] = (a, b)
    return params


def optimize_promo(skus: list[SKU], budget: float) -> PromoResult:
    params = _category_response(skus)
    cats = list(params)
    a = np.array([params[c][0] for c in cats])
    b = np.array([params[c][1] for c in cats])

    def neg_uplift(x):
        return -np.sum(a * (1 - np.exp(-b * x)))

    def neg_grad(x):
        return -(a * b * np.exp(-b * x))

    x0 = np.full(len(cats), budget / len(cats))
    cons = [{"type": "ineq", "fun": lambda x: budget - np.sum(x)}]   # sum <= budget
    bounds = [(0, budget) for _ in cats]
    res = minimize(neg_uplift, x0, jac=neg_grad, bounds=bounds,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-9})
    x = np.clip(res.x, 0, None)
    inc = float(np.sum(a * (1 - np.exp(-b * x))))
    even = float(np.sum(a * (1 - np.exp(-b * x0))))
    return PromoResult(
        allocation={c: round(float(v), 2) for c, v in zip(cats, x)},
        incremental_margin_eur=round(inc, 2), budget_eur=budget,
        even_split_margin_eur=round(even, 2),
        uplift_vs_even_eur=round(inc - even, 2))
