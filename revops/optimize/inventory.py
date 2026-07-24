"""inventory.py — how much of each carried SKU to hold. Newsvendor + safety stock.

For each SKU we solve the single-period newsvendor optimum: the order-up-to
level that balances the cost of a stockout (lost unit margin) against the cost
of overstock (holding). The critical fractile is cu/(cu+co); the optimal level
is the demand quantile at that fractile. We add the statistical safety
stock/reorder point (Z from the target service level) and a service-vs-cost
frontier so the planner can see the trade-off.

    from revops.optimize.inventory import newsvendor, plan_inventory
    plan_inventory(load(), service_level=0.95)

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from ..dataio import SKU


@dataclass
class InventoryLine:
    sku: str
    critical_fractile: float
    order_up_to: float
    safety_stock: float
    reorder_point: float
    expected_holding_eur: float
    expected_shortage_units: float


def newsvendor(mean: float, std: float, unit_margin: float, unit_cost: float,
               holding_rate_monthly: float) -> tuple[float, float]:
    """Return (order_up_to_level, critical_fractile). cu = lost margin on a
    stockout; co = one period's holding cost on an unsold unit."""
    cu = max(1e-6, unit_margin)
    co = max(1e-6, unit_cost * holding_rate_monthly)
    cf = cu / (cu + co)
    z = NormalDist().inv_cdf(min(0.9999, max(0.5, cf)))
    return mean + z * std, cf


def plan_inventory(skus: list[SKU], service_level: float = 0.95) -> list[InventoryLine]:
    z_service = NormalDist().inv_cdf(min(0.9999, max(0.5, service_level)))
    out: list[InventoryLine] = []
    for s in skus:
        hr_m = s.annual_holding_rate / 12
        S, cf = newsvendor(s.avg_monthly_demand, s.demand_std,
                           s.unit_margin_eur, s.unit_cost_eur, hr_m)
        # safety stock accounting for both demand and lead-time variability
        lt = max(0.1, s.lead_time_months)
        ss = z_service * math.sqrt(lt * s.demand_std ** 2 +
                                   (s.avg_monthly_demand ** 2) * s.lead_time_std ** 2)
        rop = s.avg_monthly_demand * lt + ss
        exp_hold = max(0.0, S - s.avg_monthly_demand) * s.unit_cost_eur * hr_m
        exp_short = _expected_shortage(s.avg_monthly_demand, s.demand_std, S)
        out.append(InventoryLine(
            sku=s.sku, critical_fractile=round(cf, 3), order_up_to=round(S, 1),
            safety_stock=round(ss, 1), reorder_point=round(rop, 1),
            expected_holding_eur=round(exp_hold, 2),
            expected_shortage_units=round(exp_short, 2)))
    return out


def _expected_shortage(mean: float, std: float, level: float) -> float:
    """E[(D - level)+] for D ~ Normal(mean, std): std * (phi(k) - k*(1-Phi(k)))
    with k = (level - mean)/std. The expected units short per period."""
    if std <= 0:
        return max(0.0, mean - level)
    k = (level - mean) / std
    nd = NormalDist()
    phi = math.exp(-0.5 * k * k) / math.sqrt(2 * math.pi)
    return max(0.0, std * (phi - k * (1 - nd.cdf(k))))


def service_cost_frontier(s: SKU, levels=(0.80, 0.90, 0.95, 0.98, 0.99)) -> list[dict]:
    """Trade-off curve: safety-stock capital vs service level for one SKU."""
    lt = max(0.1, s.lead_time_months)
    out = []
    for sl in levels:
        z = NormalDist().inv_cdf(sl)
        ss = z * math.sqrt(lt * s.demand_std ** 2 +
                           (s.avg_monthly_demand ** 2) * s.lead_time_std ** 2)
        out.append({"service_level": sl, "safety_stock": round(ss, 1),
                    "safety_capital_eur": round(ss * s.unit_cost_eur, 2)})
    return out
