"""prescribe.py — the AI -> optimization pipeline (the centerpiece).

`prescribe()` runs the trained predictive layer, feeds its outputs into the
existing optimization core, and assembles a single prescriptive plan:

    predict                              optimize (unchanged core)
    -------                              --------------------------
    demand forecaster  ->  forecast demand replaces the raw average in a copy
                           of the SKUs used by the newsvendor + assortment MILP
    elasticity ridge   ->  estimated elasticity replaces the true one in pricing
    decline classifier ->  high-risk SKUs get a demand haircut, so the MILP
                           deprioritizes / drops fragile ranges

The plan reports the total expected € uplift vs the current baseline (current
prices, no promo optimization, naive GMROI assortment) and a set of plain-English
decision cards. The return value is a JSON-serializable dict.

Author: Dimitres Kisimov.
"""
from __future__ import annotations

from dataclasses import asdict, replace

from .dataio import SKU, load
from .optimize.assortment import optimize_assortment
from .optimize.inventory import plan_inventory
from .optimize.pricing import optimize_prices
from .optimize.promo import optimize_promo
from .predict import elasticity as el
from .predict import forecast as fc
from .predict import risk as rk

# demand haircut applied to a SKU at decline-probability 1.0 (linear in risk)
RISK_HAIRCUT = 0.5
# SKUs above this decline probability are called out as "at risk"
RISK_ALERT = 0.5


def _adjusted_skus(skus: list[SKU], forecasts: dict[str, float],
                   elasticities: dict[str, float], risks: dict[str, float]
                   ) -> tuple[list[SKU], list[SKU]]:
    """Two SKU copies fed to the optimizers:
      * `for_ops`   — forecast demand, risk-haircut  (assortment + inventory)
      * `for_price` — forecast demand, ESTIMATED elasticity (pricing)."""
    for_ops, for_price = [], []
    for s in skus:
        f = forecasts.get(s.sku, s.avg_monthly_demand)
        r = risks.get(s.sku, 0.0)
        e = elasticities.get(s.sku, s.elasticity)
        demand_eff = max(0.1, f * (1.0 - RISK_HAIRCUT * r))
        for_ops.append(replace(s, avg_monthly_demand=demand_eff))
        for_price.append(replace(s, avg_monthly_demand=f, elasticity=e))
    return for_ops, for_price


def _decision_cards(assort, prices, promo, inv, risks, forecasts,
                    fc_q, el_q, rk_q, uplift, carried_set) -> list[str]:
    cards: list[str] = []

    n_risk_dropped = sum(1 for s in assort.dropped if risks.get(s, 0) >= RISK_ALERT)
    cards.append(
        f"Assortment: carry {assort.n_carried} of "
        f"{assort.n_carried + len(assort.dropped)} SKUs within "
        f"EUR {assort.budget_eur:,.0f} working capital "
        f"(EUR {assort.capital_used_eur:,.0f} used). The MILP earns "
        f"EUR {assort.milp_uplift_vs_greedy_eur:,.0f}/yr over the GMROI heuristic; "
        f"{n_risk_dropped} of the dropped SKUs are flagged declining.")

    moves = sorted((p for p in prices if abs(p.price_change_pct) >= 1.0),
                   key=lambda p: -abs(p.margin_uplift_eur))[:3]
    for p in moves:
        direction = "raise" if p.price_change_pct > 0 else "cut"
        cards.append(
            f"Price: {direction} {p.sku} by {p.price_change_pct:+.1f}% "
            f"(EUR {p.current_price:.2f} -> {p.recommended_price:.2f}) "
            f"-> EUR {p.margin_uplift_eur * 12:,.0f}/yr "
            f"[estimated elasticity {p.elasticity:.2f}, {p.flag}].")

    alloc = sorted(promo.allocation.items(), key=lambda kv: -kv[1])
    top = ", ".join(f"{c} EUR {v:,.0f}" for c, v in alloc[:4] if v > 1)
    cards.append(
        f"Promo: split EUR {promo.budget_eur:,.0f} as {top} "
        f"-> EUR {promo.incremental_margin_eur:,.0f} incremental margin "
        f"(EUR {promo.uplift_vs_even_eur:,.0f} better than an even split).")

    avg_ss = sum(li.safety_stock for li in inv) / max(1, len(inv))
    cards.append(
        f"Inventory: set order-up-to & reorder points on {len(inv)} carried "
        f"SKUs from the demand forecast (avg safety stock {avg_ss:.1f} units).")

    n_alert = sum(1 for s, r in risks.items()
                  if r >= RISK_ALERT and s in carried_set)
    cards.append(
        f"Risk: {sum(1 for r in risks.values() if r >= RISK_ALERT)} SKUs flagged "
        f"declining (prob >= {RISK_ALERT}); {n_alert} are still carried but "
        f"demand-haircut. Decline classifier test ROC-AUC {rk_q['roc_auc']}.")

    cards.append(
        f"Forecast quality: demand MASE {fc_q['mase']} vs seasonal-naive "
        f"{fc_q['naive_mase']} ({'beats' if fc_q['beats_naive'] else 'below'} "
        f"naive); elasticity recovered to EUR-free MAE {el_q['mae_by_category']}.")

    cards.append(
        f"Bottom line: total expected uplift EUR "
        f"{uplift['total_expected_uplift_eur']:,.0f}/yr "
        f"(pricing {uplift['pricing_uplift_annual_eur']:,.0f} + promo "
        f"{uplift['promo_incremental_eur']:,.0f} + assortment "
        f"{uplift['assortment_uplift_vs_greedy_eur']:,.0f}).")
    return cards


def prescribe(budget: float = 60_000.0,
              shelf_capacity_m3: float | None = 500.0,
              service_level: float = 0.95,
              promo_budget: float = 40_000.0,
              price_guardrail: float = 0.15,
              verbose: bool = True) -> dict:
    history = fc.load_history()
    skus = load()

    # ---------------- predict ----------------
    model = fc.train(history, verbose=verbose)
    forecasts = {sku: fc.forecast_next(model, rows)
                 for sku, rows in history.items()}
    fc.build_forecast_cache(model, history)               # refresh data/forecasts.csv
    elasticities = el.estimate_elasticities(history)
    risks = rk.score_decline_risk(history)

    # honest model-quality metrics
    fc_q = fc.backtest(history)
    el_q = el.recovery_report(history)
    rk_q = rk.quality_report(history)

    # ---------------- feed the optimizers ----------------
    for_ops, for_price = _adjusted_skus(skus, forecasts, elasticities, risks)

    assort = optimize_assortment(for_ops, budget, min_per_category=3,
                                 shelf_capacity_m3=shelf_capacity_m3)
    carried_set = set(assort.carried)
    carried_ops = [s for s in for_ops if s.sku in carried_set]
    carried_price = [s for s in for_price if s.sku in carried_set]

    inv = plan_inventory(carried_ops, service_level)
    prices = optimize_prices(carried_price, price_guardrail)
    promo = optimize_promo(carried_ops, promo_budget)

    # ---------------- uplift vs current baseline ----------------
    pricing_uplift_annual = sum(p.margin_uplift_eur for p in prices) * 12
    uplift = {
        "pricing_uplift_annual_eur": round(pricing_uplift_annual, 2),
        "promo_incremental_eur": round(promo.incremental_margin_eur, 2),
        "assortment_uplift_vs_greedy_eur": round(assort.milp_uplift_vs_greedy_eur, 2),
        "total_expected_uplift_eur": round(
            pricing_uplift_annual + promo.incremental_margin_eur
            + assort.milp_uplift_vs_greedy_eur, 2),
    }

    baseline_margin = round(sum(s.annual_margin_eur for s in skus), 2)

    cards = _decision_cards(assort, prices, promo, inv, risks, forecasts,
                            fc_q, el_q, rk_q, uplift, carried_set)

    return {
        "inputs": {
            "budget_eur": budget, "shelf_capacity_m3": shelf_capacity_m3,
            "service_level": service_level, "promo_budget_eur": promo_budget,
            "price_guardrail": price_guardrail,
        },
        "model_quality": {
            "forecast": fc_q,
            "elasticity": {"mae_by_category": el_q["mae_by_category"],
                           "mae_by_sku": el_q["mae_by_sku"],
                           "per_category": el_q["per_category"]},
            "decline_risk": rk_q,
        },
        "expected_uplift": uplift,
        "baseline_current_annual_margin_eur": baseline_margin,
        "plan": {
            "assortment": {
                "n_carried": assort.n_carried,
                "n_dropped": len(assort.dropped),
                "carried": assort.carried,
                "dropped": assort.dropped,
                "capital_used_eur": assort.capital_used_eur,
                "budget_eur": assort.budget_eur,
                "optimized_annual_margin_eur": assort.total_margin_eur,
                "greedy_annual_margin_eur": assort.greedy_margin_eur,
                "milp_uplift_vs_greedy_eur": assort.milp_uplift_vs_greedy_eur,
                "status": assort.status,
            },
            "pricing": {
                "n_moves": sum(1 for p in prices if abs(p.price_change_pct) >= 1.0),
                "lines": [asdict(p) for p in prices],
            },
            "inventory": {"lines": [asdict(li) for li in inv]},
            "promo": {
                "allocation": promo.allocation,
                "incremental_margin_eur": promo.incremental_margin_eur,
                "uplift_vs_even_eur": promo.uplift_vs_even_eur,
                "budget_eur": promo.budget_eur,
            },
        },
        "decision_cards": cards,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(prescribe(verbose=False)["expected_uplift"], indent=2))
