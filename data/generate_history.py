"""generate_history.py — synthesize 24 months of demand & price history per SKU.

For every SKU in the master we emit 24 monthly observations that carry a real,
learnable structure:

  * seasonality  — a spring (April) and autumn (October) peak,
  * a gentle per-SKU trend whose sign is tied to the SKU's demand_risk (so
    high-risk SKUs visibly decline — the decline classifier has honest labels),
  * noise around the SKU's avg_monthly_demand / demand_std,
  * the price actually charged that month (current price, discounted on the odd
    promo month), and
  * the resulting units, produced by applying the SKU's *true* elasticity
    Q = baseline * (price/current_price)^e — so a price<->quantity relationship
    genuinely exists for the elasticity estimator to recover.

Deterministic (fixed seed).

    python data/generate_history.py
Out: data/demand_history.csv  (columns: sku, month, price_eur, units, demand_baseline)

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

SEED = 20240724
N_MONTHS = 24
START_YEAR, START_MONTH = 2024, 7          # 2024-07 .. 2026-06 (24 months)

HERE = Path(__file__).resolve().parent
SRC = HERE / "sku_master.csv"
OUT = HERE / "demand_history.csv"


def _month_labels() -> list[tuple[str, int]]:
    """Return (YYYY-MM, calendar_month) for each of the N_MONTHS periods."""
    out = []
    y, m = START_YEAR, START_MONTH
    for _ in range(N_MONTHS):
        out.append((f"{y:04d}-{m:02d}", m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _seasonal(cal_month: int) -> float:
    """Two peaks a year: April (4) and October (10). Amplitude ~15%."""
    return 1.0 + 0.15 * math.cos(2 * math.pi * 2 * (cal_month - 4) / 12)


def main() -> None:
    rng = np.random.default_rng(SEED)
    months = _month_labels()

    with SRC.open(encoding="utf-8-sig", newline="") as f:
        skus = list(csv.DictReader(f))

    rows = []
    for r in skus:
        sku = r["sku"]
        avg = float(r["avg_monthly_demand"])
        std = float(r["demand_std"])
        price0 = float(r["current_price_eur"])
        elasticity = float(r["elasticity"])
        risk = float(r["demand_risk"])

        # Per-SKU trend: healthy SKUs drift up gently, high-risk SKUs decline.
        # base ~ +0.4%/mo, minus 3%/mo scaled by demand_risk.
        trend = float(rng.normal(0.004, 0.003)) - 0.030 * risk

        for t, (label, cal_m) in enumerate(months):
            season = _seasonal(cal_m)
            baseline = avg * season * (1.0 + trend * t)
            baseline += float(rng.normal(0.0, std * 0.4))         # demand noise
            baseline = max(0.1, baseline)

            # Price: mostly the current price, discounted on ~28% of months.
            if rng.random() < 0.28:
                disc = float(rng.uniform(0.05, 0.25))
                price = price0 * (1.0 - disc)
            else:
                price = price0
            price = round(price, 2)

            # Realized units apply the TRUE elasticity to the price ratio.
            units = baseline * (price / price0) ** elasticity
            units *= math.exp(float(rng.normal(0.0, 0.05)))       # small mult. noise
            units = max(0.1, round(units, 1))

            rows.append({
                "sku": sku, "month": label,
                "price_eur": price, "units": units,
                "demand_baseline": round(baseline, 2),
            })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sku", "month", "price_eur",
                                          "units", "demand_baseline"])
        w.writeheader()
        w.writerows(rows)

    n_sku = len(skus)
    print(f"{len(rows)} rows = {n_sku} SKUs x {N_MONTHS} months "
          f"({months[0][0]}..{months[-1][0]}) -> {OUT}")


if __name__ == "__main__":
    main()
