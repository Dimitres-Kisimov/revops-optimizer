"""assortment.py — which SKUs should we carry? A real 0/1 knapsack MILP.

Strategic range decision: pick the subset of SKUs to stock that maximizes total
annual gross margin, subject to a working-capital budget (the money tied up in
average inventory) and a minimum breadth per category (you can't drop a whole
category to zero). This is a mixed-integer linear program, solved with
scipy.optimize.milp (HiGHS). We also compare against the naive "rank by GMROI
and fill the budget" greedy heuristic to show the MILP earns its keep.

    from revops.optimize.assortment import optimize_assortment
    res = optimize_assortment(load(), budget=60_000, min_per_category=3)

Author: Dimitres Kisimov.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, milp

from ..dataio import SKU


@dataclass
class AssortmentResult:
    carried: list[str]
    dropped: list[str]
    total_margin_eur: float
    capital_used_eur: float
    budget_eur: float
    n_carried: int
    status: str
    greedy_margin_eur: float
    milp_uplift_vs_greedy_eur: float


def optimize_assortment(skus: list[SKU], budget: float,
                        min_per_category: int = 3,
                        shelf_capacity_m3: float | None = None) -> AssortmentResult:
    n = len(skus)
    value = np.array([s.annual_margin_eur for s in skus])       # maximize
    weight = np.array([s.avg_inventory_cost_eur for s in skus])  # working capital
    # cube each SKU needs on the shelf ~ a cycle+safety of stock
    space = np.array([(s.avg_monthly_demand * max(0.5, s.lead_time_months) * 0.5
                       + 1.65 * s.demand_std) * s.shelf_m3_per_unit for s in skus])

    # milp minimizes c @ x, so minimize -value
    c = -value
    integrality = np.ones(n)                     # all binary
    bounds_lb = np.zeros(n)
    bounds_ub = np.ones(n)

    constraints = [LinearConstraint(weight, ub=budget)]         # capital budget
    if shelf_capacity_m3 is not None:
        constraints.append(LinearConstraint(space, ub=shelf_capacity_m3))  # 2nd knapsack dim

    # breadth: at least `min_per_category` carried SKUs per category
    cats: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(skus):
        cats[s.category].append(i)
    for _cat, idxs in cats.items():
        row = np.zeros(n)
        row[idxs] = 1.0
        need = min(min_per_category, len(idxs))
        constraints.append(LinearConstraint(row, lb=need))

    from scipy.optimize import Bounds
    res = milp(c=c, integrality=integrality, bounds=Bounds(bounds_lb, bounds_ub),
               constraints=constraints, options={"disp": False})

    x = np.round(res.x).astype(int) if res.x is not None else np.zeros(n, int)
    carried = [skus[i].sku for i in range(n) if x[i] == 1]
    dropped = [skus[i].sku for i in range(n) if x[i] == 0]
    total_margin = float(value[x == 1].sum())
    capital = float(weight[x == 1].sum())

    greedy = _greedy(skus, budget, cats, min_per_category, space,
                     shelf_capacity_m3)
    return AssortmentResult(
        carried=carried, dropped=dropped,
        total_margin_eur=round(total_margin, 2), capital_used_eur=round(capital, 2),
        budget_eur=budget, n_carried=len(carried),
        status="optimal" if res.success else str(res.status),
        greedy_margin_eur=round(greedy, 2),
        milp_uplift_vs_greedy_eur=round(total_margin - greedy, 2),
    )


def _greedy(skus: list[SKU], budget: float, cats: dict[str, list[int]],
            min_per_category: int, space, shelf_capacity_m3: float | None) -> float:
    """Rank by GMROI (margin per euro of capital), fill until a budget binds —
    the common heuristic the MILP is benchmarked against. With two constraints
    (capital + shelf) this greedy is genuinely sub-optimal, which is the point."""
    import math
    chosen: set[int] = set()
    spent = 0.0
    used_space = 0.0
    cap_space = shelf_capacity_m3 if shelf_capacity_m3 is not None else math.inf
    for idxs in cats.values():
        for i in sorted(idxs, key=lambda i: -skus[i].gmroi)[:min_per_category]:
            chosen.add(i)
            spent += skus[i].avg_inventory_cost_eur
            used_space += space[i]
    for i in sorted(range(len(skus)), key=lambda i: -skus[i].gmroi):
        if i in chosen:
            continue
        w = skus[i].avg_inventory_cost_eur
        if spent + w <= budget and used_space + space[i] <= cap_space:
            chosen.add(i)
            spent += w
            used_space += space[i]
    return sum(skus[i].annual_margin_eur for i in chosen)
