"""Tests for the A-vs-B plan compare & € bridge (revops/compare.py).

Fast and key-free: every check runs on a tiny synthetic ``PredictContext`` (no
forecaster training) and a small draw count. The bridge identity is asserted
*exactly* — in cents, not to a tolerance — because that is the module's whole
claim: every euro between plan A's total and plan B's is attributed to a named
driver. The per-line CSV is pinned to the bridge it explains, the degenerate
cases (a plan against itself; a compare that moves exactly one lever) are
hand-checked, and the deliverables are byte-stable.
"""
from __future__ import annotations

import pytest
from conftest import make_sku

from revops.compare import (
    BRIDGE_DRIVERS,
    DEFAULT_A_NAME,
    DEFAULT_B_NAME,
    DRIVER_ASSORTMENT,
    DRIVER_PRICE_ACTION,
    DRIVER_PRICE_CONDITIONS,
    DRIVER_PRICE_MIX,
    DRIVER_PROMO_BUDGET,
    DRIVER_PROMO_MIX,
    KIND_ANCHOR,
    KIND_DRIVER,
    PlanComparison,
    compare_plans,
    render_bridge_svg,
    scenario_by_name,
    summary_rows,
    write_bridge_csv,
    write_lines_csv,
    write_summary_csv,
)
from revops.scenario import (
    BASELINE_PROMO_BUDGET,
    DEFAULT_SCENARIOS,
    PredictContext,
    Scenario,
    evaluate,
)


def _ctx() -> PredictContext:
    """A small, deterministic catalogue across three categories (the same one
    the scenario/simulate/robustness tests use, so every layer exercises the
    identical core)."""
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


_A = Scenario("Baseline")
_B = Scenario("Cost inflation +8%", cost_mult=1.08)


def _drivers(cmp: PlanComparison) -> dict[str, float]:
    return {s.driver: s.delta_eur for s in cmp.bridge if s.kind == KIND_DRIVER}


def _cents(eur: float) -> int:
    return int(round(eur * 100))


# --------------------------------------------------------------------------- #
# the bridge identity — the module's central claim                             #
# --------------------------------------------------------------------------- #
def test_bridge_reconciles_exactly():
    """total_A + every named driver == total_B, to the cent, with no residual."""
    cmp = compare_plans(_ctx(), _A, _B)
    steps = cmp.bridge
    assert [s.driver for s in steps] == [_A.name, *BRIDGE_DRIVERS, _B.name]
    assert [s.kind for s in steps] == [KIND_ANCHOR] + [KIND_DRIVER] * 6 + [KIND_ANCHOR]
    assert steps[0].running_eur == cmp.total_a_eur
    assert steps[-1].running_eur == cmp.total_b_eur

    total = _cents(cmp.total_a_eur) + sum(_cents(s.delta_eur) for s in steps)
    assert total == _cents(cmp.total_b_eur)          # exact, in cents
    assert cmp.residual_eur == 0.0
    assert _cents(cmp.delta_eur) == _cents(cmp.total_b_eur) - _cents(cmp.total_a_eur)
    # anchors carry no delta of their own, so the column sums to the difference
    assert sum(_cents(s.delta_eur) for s in steps) == _cents(cmp.delta_eur)
    # the running column is the cumulative sum, step by step
    run = _cents(cmp.total_a_eur)
    for s in steps:
        run += _cents(s.delta_eur)
        assert _cents(s.running_eur) == run


def test_plan_totals_are_the_published_scenario_totals():
    """The anchors are not a second opinion: they are exactly what evaluate()
    publishes for the same two scenarios."""
    ctx = _ctx()
    cmp = compare_plans(ctx, _A, _B)
    assert cmp.total_a_eur == evaluate(ctx, _A).total_uplift_eur
    assert cmp.total_b_eur == evaluate(ctx, _B).total_uplift_eur
    assert cmp.pricing_a_eur == evaluate(ctx, _A).pricing_uplift_eur
    assert cmp.promo_b_eur == evaluate(ctx, _B).promo_uplift_eur
    assert cmp.assortment_b_eur == evaluate(ctx, _B).assortment_uplift_eur


def test_plan_against_itself_is_all_zero():
    cmp = compare_plans(_ctx(), _A, Scenario("Baseline"))
    assert cmp.total_a_eur == cmp.total_b_eur
    assert cmp.delta_eur == 0.0 and cmp.residual_eur == 0.0
    assert all(s.delta_eur == 0.0 for s in cmp.bridge)
    assert cmp.entered == [] and cmp.left == []
    assert cmp.n_price_changed == 0 and cmp.n_verdict_flips == 0
    # only the promo table survives, and every category is unmoved
    assert {r.kind for r in cmp.lines} <= {"promo"}
    assert all(r.change == "steady" and r.delta_eur == 0.0 for r in cmp.lines)


def test_promo_budget_only_move_lands_on_one_driver():
    """A compare that moves exactly one lever must put the whole difference on
    that lever's driver and leave the other five at zero."""
    ctx = _ctx()
    b = Scenario("promo push", promo_budget=BASELINE_PROMO_BUDGET * 1.5)
    cmp = compare_plans(ctx, _A, b)
    d = _drivers(cmp)
    assert d[DRIVER_PROMO_BUDGET] == cmp.delta_eur
    for name in (DRIVER_PRICE_ACTION, DRIVER_PRICE_CONDITIONS, DRIVER_PRICE_MIX,
                 DRIVER_PROMO_MIX, DRIVER_ASSORTMENT):
        assert d[name] == 0.0
    assert cmp.delta_eur == round(evaluate(ctx, b).total_uplift_eur
                                  - evaluate(ctx, _A).total_uplift_eur, 2)


def test_capital_only_move_leaves_the_shared_price_lines_alone():
    """Changing capital/shelf cannot change a *carried* SKU's price maths, so
    both same-SKU pricing drivers must be exactly zero and the pricing € can
    only move through assortment mix."""
    ctx = _ctx()
    b = Scenario("tight", budget=30_000.0, shelf_capacity_m3=8.0)
    cmp = compare_plans(ctx, _A, b)
    d = _drivers(cmp)
    assert d[DRIVER_PRICE_ACTION] == 0.0
    assert d[DRIVER_PRICE_CONDITIONS] == 0.0
    assert _cents(d[DRIVER_PRICE_MIX]) == _cents(cmp.pricing_b_eur) \
        - _cents(cmp.pricing_a_eur)


def test_lines_audit_the_pricing_drivers():
    """The per-line CSV explains the bridge it ships with: the line rows sum to
    the three pricing drivers exactly."""
    cmp = compare_plans(_ctx(), _A, _B)
    d = _drivers(cmp)
    cond = sum(_cents(r.conditions_eur) for r in cmp.lines)
    action = sum(_cents(r.action_eur) for r in cmp.lines)
    mix = sum(_cents(r.delta_eur) for r in cmp.lines if r.kind == "assortment")
    assert cond == _cents(d[DRIVER_PRICE_CONDITIONS])
    assert action == _cents(d[DRIVER_PRICE_ACTION])
    assert mix == _cents(d[DRIVER_PRICE_MIX])
    # and each price row's own two shares add up to its delta
    for r in cmp.lines:
        if r.kind == "price":
            assert _cents(r.conditions_eur) + _cents(r.action_eur) == _cents(r.delta_eur)


def test_line_rows_are_shaped_and_ordered():
    cmp = compare_plans(_ctx(), _A, _B)
    kinds = [r.kind for r in cmp.lines]
    assert kinds == sorted(kinds, key=lambda k: {"assortment": 0, "price": 1,
                                                 "promo": 2}[k])
    for r in cmp.lines:
        assert r.metric in ("price-line uplift EUR/yr", "promo budget EUR")
        assert _cents(r.delta_eur) == _cents(r.b_eur) - _cents(r.a_eur)
        if r.kind == "assortment":
            assert r.change in ("enters", "leaves")
            assert (r.key in cmp.entered) == (r.change == "enters")
        if r.kind == "price":
            assert r.change in ("repriced", "revalued", "steady")
    # inside a kind, the biggest absolute € reads first
    for kind in ("assortment", "price", "promo"):
        mags = [abs(_cents(r.delta_eur)) for r in cmp.lines if r.kind == kind]
        assert mags == sorted(mags, reverse=True)


def test_promo_rows_cover_every_category():
    ctx = _ctx()
    cmp = compare_plans(ctx, _A, _B)
    cats = {r.key for r in cmp.lines if r.kind == "promo"}
    assert cats == {s.category for s in ctx.skus}


# --------------------------------------------------------------------------- #
# the gate: verdict flips                                                      #
# --------------------------------------------------------------------------- #
def test_verdict_flips_are_reported_and_counted():
    cmp = compare_plans(_ctx(), _A, _B, gate_draws=6)
    assert cmp.gate_draws == 6
    seen = 0
    for r in cmp.lines:
        assert r.a_verdict in ("", "accept", "hold")
        assert r.b_verdict in ("", "accept", "hold")
        if r.verdict_flip:
            assert r.a_verdict and r.b_verdict and r.a_verdict != r.b_verdict
            assert r.verdict_flip == f"{r.a_verdict}->{r.b_verdict}"
            seen += 1
    assert seen == cmp.n_verdict_flips
    assert cmp.n_verdict_flips >= 1          # this pair does move a verdict


def test_no_gate_leaves_verdict_columns_empty():
    cmp = compare_plans(_ctx(), _A, _B, gate_draws=0)
    assert cmp.gate_draws == 0
    assert cmp.n_verdict_flips == 0
    assert all(r.a_verdict == "" and r.b_verdict == "" and r.verdict_flip == ""
               for r in cmp.lines)


def test_gating_does_not_move_the_bridge():
    """The verdicts annotate the diff; they must not touch a single euro of it."""
    ctx = _ctx()
    plain = compare_plans(ctx, _A, _B, gate_draws=0)
    gated = compare_plans(ctx, _A, _B, gate_draws=4)
    assert [s.__dict__ for s in plain.bridge] == [s.__dict__ for s in gated.bridge]
    assert plain.total_b_eur == gated.total_b_eur


def test_is_deterministic():
    ctx = _ctx()
    a = compare_plans(ctx, _A, _B, gate_draws=4)
    b = compare_plans(ctx, _A, _B, gate_draws=4)
    assert [s.__dict__ for s in a.bridge] == [s.__dict__ for s in b.bridge]
    assert [r.__dict__ for r in a.lines] == [r.__dict__ for r in b.lines]


# --------------------------------------------------------------------------- #
# the scenario lookup used by the CLI                                          #
# --------------------------------------------------------------------------- #
def test_scenario_by_name_and_published_defaults():
    assert scenario_by_name(DEFAULT_A_NAME).name == DEFAULT_A_NAME
    assert scenario_by_name(DEFAULT_B_NAME).name == DEFAULT_B_NAME
    assert {sc.name for sc in DEFAULT_SCENARIOS} >= {DEFAULT_A_NAME, DEFAULT_B_NAME}
    with pytest.raises(KeyError):
        scenario_by_name("no such plan")


# --------------------------------------------------------------------------- #
# deterministic deliverables                                                  #
# --------------------------------------------------------------------------- #
def test_summary_csv_is_tall_and_numeric(tmp_path):
    cmp = compare_plans(_ctx(), _A, _B)
    a = write_summary_csv(cmp, tmp_path / "a.csv").read_bytes()
    b = write_summary_csv(cmp, tmp_path / "b.csv").read_bytes()
    assert a == b
    rows = a.decode().splitlines()
    assert rows[0] == "metric,value"
    assert len(rows) == len(summary_rows(cmp)) + 1
    for row in rows[1:]:
        metric, value = row.split(",")
        assert metric
        assert isinstance(float(value), float)       # the value column is numeric
    assert "bridge_residual_eur,0.0" in rows


def test_bridge_csv_byte_identical(tmp_path):
    cmp = compare_plans(_ctx(), _A, _B)
    a = write_bridge_csv(cmp, tmp_path / "a.csv").read_bytes()
    b = write_bridge_csv(cmp, tmp_path / "b.csv").read_bytes()
    assert a == b
    rows = a.decode().splitlines()
    assert rows[0] == "order,driver,kind,delta_eur,running_eur"
    assert len(rows) == len(cmp.bridge) + 1
    # the anchors name the two plans, so the file is self-describing
    assert cmp.plan_a in rows[1] and cmp.plan_b in rows[-1]


def test_lines_csv_byte_identical(tmp_path):
    cmp = compare_plans(_ctx(), _A, _B, gate_draws=4)
    a = write_lines_csv(cmp, tmp_path / "a.csv").read_bytes()
    b = write_lines_csv(cmp, tmp_path / "b.csv").read_bytes()
    assert a == b
    rows = a.decode().splitlines()
    assert rows[0].startswith("kind,key,category,metric,change,")
    assert rows[0].endswith("a_verdict,b_verdict,verdict_flip")
    assert len(rows) == len(cmp.lines) + 1


def test_bridge_svg_byte_identical(tmp_path):
    cmp = compare_plans(_ctx(), _A, _B)
    s1 = render_bridge_svg(cmp, tmp_path / "a.svg").read_bytes()
    s2 = render_bridge_svg(cmp, tmp_path / "b.svg").read_bytes()
    assert s1 == s2
    assert s1.lstrip().startswith(b"<svg")
    assert s1.rstrip().endswith(b"</svg>")
    # the honesty framing and the exactness claim survive any redesign
    assert b"not a forecast" in s1
    assert b"Illustrative planning ranges on synthetic data" in s1
    assert b"closes exactly (residual EUR 0.00)" in s1
    # every driver is named on the chart and every step carries its own € text,
    # so the diverging colors never have to travel alone
    for name in BRIDGE_DRIVERS:
        assert name.encode() in s1
    for step in cmp.bridge:
        if step.kind == KIND_DRIVER and step.delta_eur:
            assert f"{step.delta_eur:+,.0f}".encode() in s1


def test_svg_escapes_plan_names(tmp_path):
    """Plan names are data: an ``&`` in one must not break the XML."""
    cmp = compare_plans(_ctx(), Scenario("A & B"), Scenario("A & B"))
    svg = render_bridge_svg(cmp, tmp_path / "a.svg").read_text(encoding="utf-8")
    assert "A &amp; B" in svg and "A & B" not in svg


def test_result_shape_roundtrips():
    """The result dataclass is JSON-serializable via asdict (CLI --json)."""
    import json
    from dataclasses import asdict
    cmp = compare_plans(_ctx(), _A, _B, gate_draws=2)
    payload = json.loads(json.dumps(asdict(cmp)))
    assert payload["gate_draws"] == 2
    assert len(payload["bridge"]) == len(BRIDGE_DRIVERS) + 2
    assert len(payload["lines"]) == len(cmp.lines)
