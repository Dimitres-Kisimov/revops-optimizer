"""Tests for the optimization core — MILP, newsvendor, pricing, promo."""
from __future__ import annotations

import json

from conftest import make_sku

from revops.dataio import load
from revops.optimize.assortment import optimize_assortment
from revops.optimize.inventory import newsvendor, plan_inventory
from revops.optimize.pricing import optimize_price, optimize_prices
from revops.optimize.promo import optimize_promo


# --------------------------------------------------------------------------- #
# assortment MILP                                                              #
# --------------------------------------------------------------------------- #
def test_milp_reports_optimal_status():
    skus = load()
    res = optimize_assortment(skus, budget=60_000, min_per_category=3,
                              shelf_capacity_m3=10.0)
    assert res.status == "optimal"
    # capital budget respected
    assert res.capital_used_eur <= 60_000 + 1e-6


def test_milp_strictly_beats_greedy_under_two_constraints():
    """With BOTH a capital budget and a shelf-capacity budget binding, the MILP
    is genuinely better than the GMROI greedy heuristic."""
    skus = load()
    res = optimize_assortment(skus, budget=60_000, min_per_category=3,
                              shelf_capacity_m3=10.0)
    assert res.milp_uplift_vs_greedy_eur > 0
    # a material, five-figure gap is the headline claim
    assert res.milp_uplift_vs_greedy_eur >= 10_000
    assert res.total_margin_eur >= res.greedy_margin_eur


def test_milp_result_is_json_serializable():
    skus = load()
    res = optimize_assortment(skus, budget=40_000, shelf_capacity_m3=8.0)
    from dataclasses import asdict
    json.dumps(asdict(res))


# --------------------------------------------------------------------------- #
# newsvendor / inventory                                                       #
# --------------------------------------------------------------------------- #
def test_newsvendor_critical_fractile_in_unit_interval():
    S, cf = newsvendor(mean=100, std=25, unit_margin=8.0, unit_cost=10.0,
                       holding_rate_monthly=0.02)
    assert 0.0 < cf < 1.0


def test_order_up_to_at_least_mean_when_fractile_above_half():
    # high margin vs holding -> cf > 0.5 -> order-up-to above the mean
    mean = 100.0
    S, cf = newsvendor(mean=mean, std=25, unit_margin=15.0, unit_cost=10.0,
                       holding_rate_monthly=0.02)
    assert cf > 0.5
    assert S >= mean


def test_plan_inventory_fields_sane():
    skus = [make_sku(sku=f"S{i}", avg_monthly_demand=50 + i,
                     demand_std=10 + i) for i in range(5)]
    lines = plan_inventory(skus, service_level=0.95)
    assert len(lines) == 5
    for li in lines:
        assert 0.0 < li.critical_fractile < 1.0
        assert li.safety_stock >= 0.0
        assert li.expected_shortage_units >= 0.0


# --------------------------------------------------------------------------- #
# pricing (elasticity / Lerner guardrail)                                      #
# --------------------------------------------------------------------------- #
def test_pricing_hits_lerner_optimum_when_inside_band():
    # e=-2, cost=10 -> Lerner P* = c|e|/(|e|-1) = 10*2/1 = 20
    s = make_sku(unit_cost_eur=10.0, current_price_eur=19.0, elasticity=-2.0)
    line = optimize_price(s, guardrail=0.15)     # band [16.15, 21.85] contains 20
    assert line.unconstrained_optimal == 20.0
    assert abs(line.recommended_price - 20.0) < 1e-6
    assert line.flag == "elastic"


def test_pricing_respects_guardrail_band():
    # Lerner optimum (20) far below the band -> clamp to the lower guardrail
    s = make_sku(unit_cost_eur=10.0, current_price_eur=100.0, elasticity=-2.0)
    line = optimize_price(s, guardrail=0.15)     # band [85, 115]
    lo, hi = 100.0 * 0.85, 100.0 * 1.15
    assert lo - 1e-6 <= line.recommended_price <= hi + 1e-6
    assert abs(line.recommended_price - lo) < 1e-6


def test_inelastic_sku_pushed_to_top_of_band():
    s = make_sku(unit_cost_eur=10.0, current_price_eur=20.0, elasticity=-0.5)
    line = optimize_price(s, guardrail=0.15)
    assert line.flag == "inelastic-capped"
    assert abs(line.recommended_price - 20.0 * 1.15) < 1e-6


def test_optimize_prices_batch():
    skus = [make_sku(sku=f"S{i}", elasticity=-1.5 - 0.1 * i) for i in range(4)]
    lines = optimize_prices(skus, guardrail=0.15)
    assert len(lines) == 4


# --------------------------------------------------------------------------- #
# promo allocation                                                             #
# --------------------------------------------------------------------------- #
def test_promo_within_budget_and_beats_even_split():
    # two asymmetric categories so an even split is provably sub-optimal
    skus = ([make_sku(sku=f"A{i}", category="alpha", avg_monthly_demand=200,
                      current_price_eur=40, unit_cost_eur=10, elasticity=-2.5)
             for i in range(6)]
            + [make_sku(sku=f"B{i}", category="beta", avg_monthly_demand=20,
                        current_price_eur=15, unit_cost_eur=10, elasticity=-1.2)
               for i in range(6)])
    res = optimize_promo(skus, budget=30_000)
    assert sum(res.allocation.values()) <= 30_000 + 1.0
    assert res.uplift_vs_even_eur >= 0.0
    assert res.incremental_margin_eur >= res.even_split_margin_eur
