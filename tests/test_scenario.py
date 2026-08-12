"""Tests for the what-if scenario & one-way sensitivity (tornado) layer.

Fast and key-free: every check runs on a tiny synthetic ``PredictContext`` built
directly from ``make_sku`` (no forecaster training), so the whole file solves a
handful of small MILPs and finishes in well under a second.
"""
from __future__ import annotations

from conftest import make_sku

from revops.scenario import (
    DEFAULT_DRIVERS,
    DEFAULT_SCENARIOS,
    PredictContext,
    Scenario,
    evaluate,
    render_tornado_svg,
    scenario_library,
    sensitivity_tornado,
    write_scenario_csv,
    write_tornado_csv,
)


def _ctx() -> PredictContext:
    """A small, deterministic catalogue across three categories."""
    cats = ["alpha", "beta", "gamma"]
    skus = []
    for ci, cat in enumerate(cats):
        for i in range(5):
            idx = ci * 5 + i
            skus.append(make_sku(
                sku=f"{cat[0].upper()}{i}", category=cat,
                unit_cost_eur=8.0 + idx * 0.7,
                current_price_eur=18.0 + idx * 1.3,
                avg_monthly_demand=60.0 + 8 * i + 12 * ci,
                demand_std=10.0 + i, shelf_m3_per_unit=0.01 + 0.004 * i,
                elasticity=-1.4 - 0.15 * i, annual_holding_rate=0.24))
    forecasts = {s.sku: s.avg_monthly_demand for s in skus}
    elasticities = {s.sku: s.elasticity for s in skus}
    risks = {s.sku: 0.1 for s in skus}
    return PredictContext(skus=skus, forecasts=forecasts,
                          elasticities=elasticities, risks=risks)


# --------------------------------------------------------------------------- #
# evaluate                                                                     #
# --------------------------------------------------------------------------- #
def test_evaluate_total_is_sum_of_components():
    r = evaluate(_ctx(), Scenario("Baseline"))
    parts = r.pricing_uplift_eur + r.promo_uplift_eur + r.assortment_uplift_eur
    assert abs(parts - r.total_uplift_eur) < 1.0
    # the MILP is optimal, so its margin can never trail the greedy heuristic
    assert r.assortment_uplift_eur >= -1e-6
    assert r.n_carried >= 1


def test_evaluate_is_deterministic():
    ctx = _ctx()
    a = evaluate(ctx, Scenario("Baseline"))
    b = evaluate(ctx, Scenario("Baseline"))
    assert a.__dict__ == b.__dict__


def test_promo_budget_monotonic():
    """Promo budget touches only the concave allocation (not the assortment), so
    a bigger budget can only buy at least as much incremental margin."""
    ctx = _ctx()
    lo = evaluate(ctx, Scenario("lo", promo_budget=20_000.0)).promo_uplift_eur
    hi = evaluate(ctx, Scenario("hi", promo_budget=60_000.0)).promo_uplift_eur
    assert hi >= lo - 1e-6


def test_price_guardrail_monotonic():
    """A wider price guardrail is a superset band, so the elasticity optimizer's
    achievable margin uplift cannot fall (assortment/promo are untouched)."""
    ctx = _ctx()
    narrow = evaluate(ctx, Scenario("n", price_guardrail=0.05)).pricing_uplift_eur
    wide = evaluate(ctx, Scenario("w", price_guardrail=0.25)).pricing_uplift_eur
    assert wide >= narrow - 1e-6


def test_default_baseline_matches_evaluated_baseline():
    """The tornado's default baseline is exactly the unperturbed plan's total."""
    ctx = _ctx()
    base = evaluate(ctx, Scenario("Baseline")).total_uplift_eur
    rows = sensitivity_tornado(ctx)
    assert rows
    assert all(r.baseline_uplift_eur == round(base, 2) for r in rows)


# --------------------------------------------------------------------------- #
# scenario library                                                            #
# --------------------------------------------------------------------------- #
def test_scenario_library_baseline_is_zero_delta():
    lib = scenario_library(_ctx())
    assert len(lib) == len(DEFAULT_SCENARIOS)
    assert lib[0].name == "Baseline"
    assert lib[0].delta_vs_baseline_eur == 0.0
    base = lib[0].total_uplift_eur
    for r in lib:
        assert r.delta_vs_baseline_eur == round(r.total_uplift_eur - base, 2)
        parts = r.pricing_uplift_eur + r.promo_uplift_eur + r.assortment_uplift_eur
        assert abs(parts - r.total_uplift_eur) < 1.0


def test_scenario_library_is_deterministic():
    ctx = _ctx()
    a = [r.__dict__ for r in scenario_library(ctx)]
    b = [r.__dict__ for r in scenario_library(ctx)]
    assert a == b


# --------------------------------------------------------------------------- #
# tornado                                                                      #
# --------------------------------------------------------------------------- #
def test_tornado_structure_and_ranking():
    rows = sensitivity_tornado(_ctx())
    assert len(rows) == len(DEFAULT_DRIVERS)
    # ranked by swing, widest first
    swings = [r.swing_eur for r in rows]
    assert swings == sorted(swings, reverse=True)
    for r in rows:
        assert r.low_uplift_eur <= r.high_uplift_eur + 1e-9
        assert r.swing_eur >= -1e-9
        # downside/upside decompose the bar around the baseline
        assert abs(r.downside_eur - (r.baseline_uplift_eur - r.low_uplift_eur)) < 1e-6
        assert abs(r.upside_eur - (r.high_uplift_eur - r.baseline_uplift_eur)) < 1e-6
        assert abs(r.swing_eur - (r.high_uplift_eur - r.low_uplift_eur)) < 1e-6


def test_tornado_respects_supplied_baseline():
    ctx = _ctx()
    rows = sensitivity_tornado(ctx, baseline=123_456.0)
    assert all(r.baseline_uplift_eur == 123_456.0 for r in rows)


def test_tornado_is_deterministic():
    ctx = _ctx()
    a = [r.__dict__ for r in sensitivity_tornado(ctx)]
    b = [r.__dict__ for r in sensitivity_tornado(ctx)]
    assert a == b


# --------------------------------------------------------------------------- #
# deterministic deliverables                                                  #
# --------------------------------------------------------------------------- #
def test_scenario_csv_byte_identical(tmp_path):
    lib = scenario_library(_ctx())
    a = write_scenario_csv(lib, tmp_path / "a.csv").read_bytes()
    b = write_scenario_csv(lib, tmp_path / "b.csv").read_bytes()
    assert a == b
    assert a.splitlines()[0].startswith(b"scenario,")


def test_tornado_csv_and_svg_byte_identical(tmp_path):
    ctx = _ctx()
    rows = sensitivity_tornado(ctx)
    baseline = scenario_library(ctx)[0].total_uplift_eur

    c1 = write_tornado_csv(rows, tmp_path / "a.csv").read_bytes()
    c2 = write_tornado_csv(rows, tmp_path / "b.csv").read_bytes()
    assert c1 == c2

    s1 = render_tornado_svg(rows, baseline, tmp_path / "a.svg").read_bytes()
    s2 = render_tornado_svg(rows, baseline, tmp_path / "b.svg").read_bytes()
    assert s1 == s2
    assert s1.lstrip().startswith(b"<svg")
    assert s1.rstrip().endswith(b"</svg>")
    # the honesty label and every driver name survive any redesign
    assert b"illustrative, not a forecast" in s1
    for r in rows:
        assert r.driver.encode() in s1
