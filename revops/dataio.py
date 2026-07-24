"""dataio.py — load the SKU master and derive the fields the optimizers reuse.

Pure stdlib. Everything downstream (assortment MILP, newsvendor inventory,
price optimization, promo allocation) reads these typed records.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data" / "sku_master.csv"


@dataclass
class SKU:
    sku: str
    category: str
    unit_cost_eur: float
    current_price_eur: float
    avg_monthly_demand: float
    demand_std: float
    lead_time_months: float
    lead_time_std: float
    moq: int
    shelf_m3_per_unit: float
    elasticity: float
    annual_holding_rate: float
    demand_risk: float

    # ---- derived commercial figures ----
    @property
    def unit_margin_eur(self) -> float:
        return self.current_price_eur - self.unit_cost_eur

    @property
    def margin_pct(self) -> float:
        return self.unit_margin_eur / self.current_price_eur if self.current_price_eur else 0.0

    @property
    def annual_demand(self) -> float:
        return self.avg_monthly_demand * 12

    @property
    def annual_margin_eur(self) -> float:
        return self.unit_margin_eur * self.annual_demand

    @property
    def avg_inventory_cost_eur(self) -> float:
        """Rough average inventory value at cost = half a lead time of demand
        plus a safety buffer, valued at unit cost. Enough for GMROI/knapsack."""
        cycle = self.avg_monthly_demand * max(0.5, self.lead_time_months) * 0.5
        safety = self.demand_std * 1.65
        return (cycle + safety) * self.unit_cost_eur

    @property
    def gmroi(self) -> float:
        inv = self.avg_inventory_cost_eur
        return self.annual_margin_eur / inv if inv else 0.0


def load(path: str | Path | None = None) -> list[SKU]:
    p = Path(path) if path else _DATA
    out: list[SKU] = []
    with p.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(SKU(
                sku=r["sku"], category=r["category"],
                unit_cost_eur=float(r["unit_cost_eur"]),
                current_price_eur=float(r["current_price_eur"]),
                avg_monthly_demand=float(r["avg_monthly_demand"]),
                demand_std=float(r["demand_std"]),
                lead_time_months=float(r["lead_time_months"]),
                lead_time_std=float(r["lead_time_std"]),
                moq=int(r["moq"]), shelf_m3_per_unit=float(r["shelf_m3_per_unit"]),
                elasticity=float(r["elasticity"]),
                annual_holding_rate=float(r["annual_holding_rate"]),
                demand_risk=float(r["demand_risk"]),
            ))
    return out
