"""build_star.py — export the prescriptive plan as a Power BI star schema.

Power BI Desktop imports these CSVs and models them into a classic star:

    fact_prescription  (grain = SKU)  -- the measures live here
        |  sku          -> dim_sku[sku]
        |  category     -> dim_category[category]
        |  date_key     -> dim_date[date_key]

Run after the deliverables exist:

    python -m revops.report          # writes deliverables/prescription.json
    python powerbi/build_star.py     # writes powerbi/data/*.csv

No Power BI licence or tenant is needed to *generate* the model — the point of
this pack is to demonstrate dimensional modelling + DAX. See powerbi/README.md.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PLAN = _ROOT / "deliverables" / "prescription.json"
_MASTER = _ROOT / "data" / "sku_master.csv"
_HISTORY = _ROOT / "data" / "demand_history.csv"
_FORECASTS = _ROOT / "data" / "forecasts.csv"
_SIM = _ROOT / "deliverables" / "uplift_simulation.csv"
_OUT = Path(__file__).resolve().parent / "data"

# metrics lifted from the Monte-Carlo deliverable into a disconnected KPI table
_RISK_METRICS = (
    "baseline_point_estimate_eur", "total_p10_eur", "total_p50_eur",
    "total_p90_eur", "total_mean_eur", "total_var10_eur", "total_cvar10_eur",
    "total_prob_at_or_above_hurdle", "n_draws",
)


def _load_plan() -> dict:
    if not _PLAN.exists():
        raise SystemExit(
            f"{_PLAN} not found — run `python -m revops.report` first.")
    with _PLAN.open(encoding="utf-8") as f:
        return json.load(f)


def _load_master() -> dict[str, dict]:
    out = {}
    with _MASTER.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[r["sku"]] = r
    return out


def _load_forecasts() -> dict[str, float]:
    out: dict[str, float] = {}
    if _FORECASTS.exists():
        with _FORECASTS.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                out[r["sku"]] = float(r["forecast_next_month"])
    return out


def _months() -> list[str]:
    seen: dict[str, None] = {}
    if _HISTORY.exists():
        with _HISTORY.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                seen[r["month"]] = None
    return sorted(seen)


def _next_month(label: str) -> str:
    y, m = int(label[:4]), int(label[5:7])
    m += 1
    if m > 12:
        m, y = 1, y + 1
    return f"{y:04d}-{m:02d}"


_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def build() -> dict[str, Path]:
    plan = _load_plan()
    master = _load_master()
    forecasts = _load_forecasts()
    _OUT.mkdir(parents=True, exist_ok=True)

    carried = set(plan["plan"]["assortment"]["carried"])
    inv_by = {li["sku"]: li for li in plan["plan"]["inventory"]["lines"]}
    price_by = {li["sku"]: li for li in plan["plan"]["pricing"]["lines"]}
    promo_alloc = plan["plan"]["promo"]["allocation"]

    months = _months()
    forecast_month = _next_month(months[-1]) if months else "2026-01"
    plan_date_key = int(forecast_month.replace("-", "") + "01")

    # ---- dim_date ---------------------------------------------------------
    dim_date = _OUT / "dim_date.csv"
    cal = months + [forecast_month]
    with dim_date.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date_key", "date", "year", "month_num", "month_name",
                    "quarter", "is_forecast_month"])
        for label in cal:
            y, m = int(label[:4]), int(label[5:7])
            key = int(f"{y:04d}{m:02d}01")
            w.writerow([key, f"{label}-01", y, m, _MONTH_NAMES[m],
                        f"Q{(m - 1) // 3 + 1}",
                        1 if label == forecast_month else 0])

    # ---- dim_category -----------------------------------------------------
    cat_skus: dict[str, list[str]] = defaultdict(list)
    cat_elastic: dict[str, list[float]] = defaultdict(list)
    for sku, r in master.items():
        cat_skus[r["category"]].append(sku)
        cat_elastic[r["category"]].append(float(r["elasticity"]))
    dim_category = _OUT / "dim_category.csv"
    with dim_category.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_skus", "avg_elasticity",
                    "promo_allocation_eur"])
        for cat in sorted(cat_skus):
            el = cat_elastic[cat]
            w.writerow([cat, len(cat_skus[cat]),
                        round(sum(el) / len(el), 3),
                        round(promo_alloc.get(cat, 0.0), 2)])

    # ---- dim_sku ----------------------------------------------------------
    dim_sku = _OUT / "dim_sku.csv"
    with dim_sku.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "category", "unit_cost_eur", "lead_time_months",
                    "moq", "shelf_m3_per_unit", "elasticity_true",
                    "annual_holding_rate"])
        for sku in sorted(master):
            r = master[sku]
            w.writerow([sku, r["category"], r["unit_cost_eur"],
                        r["lead_time_months"], r["moq"], r["shelf_m3_per_unit"],
                        r["elasticity"], r["annual_holding_rate"]])

    # ---- fact_prescription (grain = SKU) ----------------------------------
    fact = _OUT / "fact_prescription.csv"
    with fact.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sku", "category", "date_key", "carried",
            "forecast_demand", "current_price", "recommended_price",
            "price_change_pct", "margin_uplift_monthly_eur",
            "margin_uplift_annual_eur", "order_up_to", "safety_stock",
            "reorder_point", "expected_shortage_units", "annual_margin_eur",
            "capital_at_cost_eur", "promo_category", "promo_allocation_eur",
        ])
        for sku in sorted(master):
            r = master[sku]
            cat = r["category"]
            iv = inv_by.get(sku, {})
            pr = price_by.get(sku, {})
            unit_cost = float(r["unit_cost_eur"])
            cur_price = float(r["current_price_eur"])
            fdem = forecasts.get(sku, float(r["avg_monthly_demand"]))
            ann_margin = round((cur_price - unit_cost)
                               * float(r["avg_monthly_demand"]) * 12, 2)
            mu_m = pr.get("margin_uplift_eur", "")
            w.writerow([
                sku, cat, plan_date_key, 1 if sku in carried else 0,
                round(fdem, 2), round(cur_price, 2),
                pr.get("recommended_price", ""), pr.get("price_change_pct", ""),
                mu_m, round(mu_m * 12, 2) if mu_m != "" else "",
                iv.get("order_up_to", ""), iv.get("safety_stock", ""),
                iv.get("reorder_point", ""),
                iv.get("expected_shortage_units", ""),
                ann_margin, round(_capital(r), 2),
                cat if sku in carried else "",
                round(promo_alloc.get(cat, 0.0), 2) if sku in carried else "",
            ])

    # ---- kpi_headline (one row of plan-level scalars) ---------------------
    # A disconnected 1-row table so DAX can surface scalars that don't live at
    # the SKU grain (the MILP-vs-greedy gap, promo incremental, forecast MASE).
    kpi = _OUT / "kpi_headline.csv"
    up = plan["expected_uplift"]
    fq = plan["model_quality"]["forecast"]
    rq = plan["model_quality"]["decline_risk"]
    a = plan["plan"]["assortment"]
    pm = plan["plan"]["promo"]
    with kpi.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_expected_uplift_eur", up["total_expected_uplift_eur"]])
        w.writerow(["pricing_uplift_annual_eur", up["pricing_uplift_annual_eur"]])
        w.writerow(["promo_incremental_eur", up["promo_incremental_eur"]])
        w.writerow(["assortment_milp_vs_greedy_eur",
                    up["assortment_milp_vs_greedy_eur"]])
        w.writerow(["baseline_annual_margin_eur",
                    plan["baseline_current_annual_margin_eur"]])
        w.writerow(["forecast_mase", fq["mase"]])
        w.writerow(["seasonal_naive_mase", fq["naive_mase"]])
        w.writerow(["decline_roc_auc", rq["roc_auc"]])
        w.writerow(["capital_budget_eur", a["budget_eur"]])
        w.writerow(["capital_used_eur", a["capital_used_eur"]])
        w.writerow(["promo_budget_eur", pm["budget_eur"]])

    out = {"fact_prescription": fact, "dim_sku": dim_sku,
           "dim_category": dim_category, "dim_date": dim_date,
           "kpi_headline": kpi}

    # ---- kpi_uplift_risk (optional) ---------------------------------------
    # A second disconnected scalar table with the Monte-Carlo risk band, emitted
    # only when the simulation deliverable exists (so CI, which does not run the
    # simulation, is unaffected). Same (metric, value) shape as kpi_headline.
    risk = _write_uplift_risk()
    if risk is not None:
        out["kpi_uplift_risk"] = risk
    return out


def _write_uplift_risk() -> Path | None:
    """Lift the P10/P50/P90 band + VaR/CVaR + clear-probability from
    ``deliverables/uplift_simulation.csv`` into a Power BI KPI table. Returns
    None (no-op) when the simulation has not been run."""
    if not _SIM.exists():
        return None
    with _SIM.open(encoding="utf-8-sig", newline="") as f:
        vals = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    p = _OUT / "kpi_uplift_risk.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for m in _RISK_METRICS:
            if m in vals:
                w.writerow([m, vals[m]])
    return p


def _capital(r: dict) -> float:
    """Average inventory value at cost — mirrors dataio.SKU.avg_inventory_cost_eur."""
    d = float(r["avg_monthly_demand"])
    lt = max(0.5, float(r["lead_time_months"]))
    cycle = d * lt * 0.5
    safety = float(r["demand_std"]) * 1.65
    return (cycle + safety) * float(r["unit_cost_eur"])


if __name__ == "__main__":
    paths = build()
    for name, p in paths.items():
        n = sum(1 for _ in p.open(encoding="utf-8")) - 1
        print(f"  ok {name:<20} {n:>4} rows  -> {p.relative_to(_ROOT)}")
