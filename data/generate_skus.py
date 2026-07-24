"""generate_skus.py — synthesize a distributor SKU master for optimization.

Deterministic (fixed seed). Each SKU carries what the optimizers need: cost,
current price, demand level + variability, lead time, MOQ, shelf footprint, a
price-elasticity estimate, and a churn-linked demand risk. ~240 SKUs across the
usual industrial categories, with a long tail of slow movers (so ABC/assortment
decisions actually bite).

    python data/generate_skus.py
Out: data/sku_master.csv

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 7
OUT = Path(__file__).resolve().parent / "sku_master.csv"

CATS = {
    # category: (avg unit cost, base margin, elasticity range, popularity weight)
    "fasteners":   (0.55, 0.42, (-2.4, -1.3), 6),
    "power tools": (95.0, 0.28, (-1.8, -1.1), 2),
    "chemicals":   (3.6, 0.46, (-2.0, -1.2), 3),
    "ppe":         (2.1, 0.38, (-2.6, -1.4), 3),
    "hand tools":  (15.0, 0.34, (-1.6, -1.05), 2),
    "abrasives":   (0.7, 0.44, (-2.2, -1.3), 3),
    "electrical":  (4.2, 0.36, (-1.9, -1.1), 3),
    "plumbing":    (5.5, 0.40, (-1.7, -1.05), 2),
}


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    sku_id = 0
    cats = list(CATS)
    weights = [CATS[c][3] for c in cats]
    for _ in range(240):
        cat = rng.choices(cats, weights=weights, k=1)[0]
        base_cost, base_margin, el_range, _ = CATS[cat]
        sku_id += 1
        cost = round(base_cost * rng.uniform(0.6, 2.2), 2)
        margin = min(0.62, max(0.08, base_margin + rng.uniform(-0.08, 0.08)))
        price = round(cost / (1 - margin), 2)
        # demand: log-normal-ish, long tail of slow movers
        avg_demand = max(1.0, round(rng.lognormvariate(3.2, 1.1), 1))
        demand_cv = rng.uniform(0.15, 1.4)             # steady -> erratic
        demand_std = round(avg_demand * demand_cv, 2)
        lead_time = round(rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]), 1)   # months
        lead_std = round(lead_time * rng.uniform(0.1, 0.4), 2)
        moq = rng.choice([1, 1, 5, 10, 25, 50])
        shelf = round(rng.uniform(0.001, 0.05), 4)     # m^3 per unit (footprint)
        elasticity = round(rng.uniform(*el_range), 2)
        holding_rate = rng.uniform(0.18, 0.30)         # annual holding cost fraction
        churn_risk = round(rng.betavariate(2, 5), 3)   # demand-erosion risk 0..1
        rows.append({
            "sku": f"S{sku_id:04d}", "category": cat,
            "unit_cost_eur": cost, "current_price_eur": price,
            "avg_monthly_demand": avg_demand, "demand_std": demand_std,
            "lead_time_months": lead_time, "lead_time_std": lead_std,
            "moq": moq, "shelf_m3_per_unit": shelf, "elasticity": elasticity,
            "annual_holding_rate": round(holding_rate, 3), "demand_risk": churn_risk,
        })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rev = sum(r["current_price_eur"] * r["avg_monthly_demand"] for r in rows)
    print(f"{len(rows)} SKUs across {len(CATS)} categories, "
          f"~{rev/1000:.0f}k EUR/mo revenue potential -> {OUT}")


if __name__ == "__main__":
    main()
