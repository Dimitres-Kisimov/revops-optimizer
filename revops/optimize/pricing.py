"""pricing.py — elasticity-based price optimization with business guardrails.

Constant-elasticity demand: Q(P) = Q0 * (P/P0)^e, e < 0. Profit
π(P) = (P - c) * Q(P). Unconstrained, the optimum is the Lerner markup
P* = c * |e| / (|e| - 1) (finite only when |e| > 1). We then clamp the
recommendation to a guardrail band (default +/-15% of the current price) so the
model never proposes something commercially absurd, and report the expected
margin uplift vs the current price. Inelastic SKUs (|e| <= 1) are flagged, not
priced up to infinity.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..dataio import SKU


@dataclass
class PriceLine:
    sku: str
    current_price: float
    elasticity: float
    unconstrained_optimal: float | None
    recommended_price: float
    price_change_pct: float
    expected_margin_now: float
    expected_margin_new: float
    margin_uplift_eur: float
    flag: str


def _demand(q0: float, p0: float, p: float, e: float) -> float:
    return max(0.0, q0 * (p / p0) ** e)


def optimize_price(s: SKU, guardrail: float = 0.15) -> PriceLine:
    e = s.elasticity
    c, p0, q0 = s.unit_cost_eur, s.current_price_eur, s.avg_monthly_demand
    lo, hi = p0 * (1 - guardrail), p0 * (1 + guardrail)

    if e < -1:                                   # elastic: interior optimum exists
        p_star = c * abs(e) / (abs(e) - 1)
        rec = min(hi, max(lo, p_star))
        flag = "elastic"
    else:                                        # inelastic: margin rises with price
        p_star = None
        rec = hi                                 # push to the top of the guardrail
        flag = "inelastic-capped"

    m_now = (p0 - c) * q0
    m_new = (rec - c) * _demand(q0, p0, rec, e)
    return PriceLine(
        sku=s.sku, current_price=round(p0, 2), elasticity=e,
        unconstrained_optimal=round(p_star, 2) if p_star else None,
        recommended_price=round(rec, 2),
        price_change_pct=round(100 * (rec - p0) / p0, 2),
        expected_margin_now=round(m_now, 2), expected_margin_new=round(m_new, 2),
        margin_uplift_eur=round(m_new - m_now, 2), flag=flag)


def optimize_prices(skus: list[SKU], guardrail: float = 0.15) -> list[PriceLine]:
    return [optimize_price(s, guardrail) for s in skus]
