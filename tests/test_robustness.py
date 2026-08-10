"""Tests for the price-move robustness gate (revops/robustness.py).

Fast and key-free: every check runs on a tiny synthetic ``PredictContext`` (no
forecaster training) with a small draw count. The gate rule, the frozen-price
arithmetic and the draw loop are hand-checked; determinism, the
collapse-to-base-case gate and the two cross-layer pins (solve_plan == evaluate,
robustness draws == simulate draws) keep the layers from drifting apart.
"""
from __future__ import annotations

import pytest
from conftest import make_sku

import revops.robustness as rb
from revops.optimize.pricing import PriceLine
from revops.robustness import (
    ACCEPT_MIN_AGREEMENT,
    ACCEPT_MIN_CARRY,
    MOVE_THRESHOLD_PCT,
    RobustnessResult,
    _frozen_move_uplift,
    analyze,
    gate,
    render_moves_svg,
    write_moves_csv,
)
from revops.scenario import PlanSolution, PredictContext, Scenario, evaluate, solve_plan
from revops.simulate import DEFAULT_RISK_DRIVERS, RiskDriver, simulate


def _ctx() -> PredictContext:
    """A small, deterministic catalogue across three categories (mirrors the
    scenario/simulate tests so all three layers exercise the same core)."""
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


_FLAT = tuple(RiskDriver(d.name, d.field, 1.0, 1.0, 1.0)
              for d in DEFAULT_RISK_DRIVERS)


# --------------------------------------------------------------------------- #
# the gate rule and the frozen-price arithmetic — hand-checked                 #
# --------------------------------------------------------------------------- #
def test_gate_rule_and_reason_precedence():
    ok_c, ok_a = ACCEPT_MIN_CARRY, ACCEPT_MIN_AGREEMENT
    assert gate(1.0, 1.0, 100.0) == ("accept", "robust")
    assert gate(ok_c, ok_a, 0.01) == ("accept", "robust")     # thresholds inclusive
    assert gate(ok_c - 0.01, 1.0, 100.0) == ("hold", "delist-risk")
    assert gate(1.0, ok_a - 0.01, 100.0) == ("hold", "direction-flips")
    assert gate(1.0, 1.0, 0.0) == ("hold", "downside")        # P10 must be > 0
    assert gate(1.0, 1.0, -5.0) == ("hold", "downside")
    # precedence: delist-risk beats direction-flips beats downside
    assert gate(0.0, 0.0, -1.0) == ("hold", "delist-risk")
    assert gate(1.0, 0.0, -1.0) == ("hold", "direction-flips")


def test_frozen_move_uplift_hand_checked():
    # p0=20, c=10, q0=100, e=-2, frozen 22:
    #   now    = (20-10)*100                      = 1000
    #   frozen = (22-10)*100*(22/20)^-2 = 12*100/1.21 = 991.7355...
    got = _frozen_move_uplift(20.0, 10.0, 100.0, -2.0, 22.0)
    assert abs(got - (12 * 100 / 1.21 - 1000.0)) < 1e-9
    # charging the current price changes nothing
    assert abs(_frozen_move_uplift(20.0, 10.0, 100.0, -2.0, 20.0)) < 1e-12


def test_solve_plan_matches_evaluate():
    """The refactor pin: PlanSolution's totals are exactly what evaluate reports."""
    ctx = _ctx()
    sc = Scenario("x", demand_mult=1.1, cost_mult=1.05)
    sol = solve_plan(ctx, sc)
    r = evaluate(ctx, sc)
    assert round(sol.pricing_uplift_annual_eur, 2) == r.pricing_uplift_eur
    assert round(sol.promo.incremental_margin_eur, 2) == r.promo_uplift_eur
    assert round(sol.assortment.milp_uplift_vs_greedy_eur, 2) == r.assortment_uplift_eur
    assert round(sol.total_uplift_eur, 2) == r.total_uplift_eur
    # the pricing optimizer saw exactly the carried SKUs
    assert {s.sku for s in sol.price_inputs} == set(sol.assortment.carried)


# --------------------------------------------------------------------------- #
# the analysis                                                                 #
# --------------------------------------------------------------------------- #
def test_analyze_rejects_bad_draws():
    with pytest.raises(ValueError):
        analyze(_ctx(), draws=0)


def test_analyze_is_deterministic():
    ctx = _ctx()
    a = analyze(ctx, draws=12)
    b = analyze(ctx, draws=12)
    assert a.draws_total_eur == b.draws_total_eur
    assert [m.__dict__ for m in a.moves] == [m.__dict__ for m in b.moves]
    assert (a.n_accept, a.n_hold, a.accepted_book_p10_eur) == \
           (b.n_accept, b.n_hold, b.accepted_book_p10_eur)


def test_draws_match_simulate():
    """The cross-layer pin: with the same seed/draw count the gate replays
    byte-for-byte the draws behind the published Monte-Carlo risk band, so the
    accept/hold split decomposes *that* band, not a parallel model."""
    ctx = _ctx()
    res = analyze(ctx, draws=16)
    sim = simulate(ctx, draws=16)
    assert res.draws_total_eur == sim.draws_total_eur
    assert res.baseline_total_uplift_eur == sim.baseline_point_estimate_eur
    assert res.seed == sim.seed


def test_moves_are_the_baseline_plan_moves():
    ctx = _ctx()
    res = analyze(ctx, draws=8)
    base_lines = {p.sku: p for p in solve_plan(ctx, Scenario("Baseline")).prices
                  if abs(p.price_change_pct) >= MOVE_THRESHOLD_PCT}
    assert {m.sku for m in res.moves} == set(base_lines)
    assert res.n_moves == len(res.moves) == len(base_lines)
    for m in res.moves:
        line = base_lines[m.sku]
        assert m.baseline_change_pct == round(line.price_change_pct, 2)
        assert m.direction == ("raise" if line.price_change_pct > 0 else "cut")
        assert m.category != ""


def test_baseline_frozen_uplift_close_to_optimizer_line():
    """The frozen-price € at baseline differs from the optimizer's own uplift
    only by the cent-rounding of the published price. Bound the gap by hand:
    |Δπ| <= |Δp| · max|dπ/dP| with |Δp| <= 0.005 and
    |dπ/dP| = Q(P)·|1 + e(P−c)/P| <= q0·(0.85)^e·(1+|e|) on the guardrail band."""
    ctx = _ctx()
    res = analyze(ctx, draws=4)
    base_sol = solve_plan(ctx, Scenario("Baseline"))
    lines = {p.sku: p for p in base_sol.prices}
    inputs = {s.sku: s for s in base_sol.price_inputs}
    for m in res.moves:
        s = inputs[m.sku]
        tol = 12 * 0.005 * s.avg_monthly_demand \
            * 0.85 ** s.elasticity * (1 + abs(s.elasticity)) + 0.02
        assert abs(m.baseline_uplift_annual_eur
                   - lines[m.sku].margin_uplift_eur * 12) <= tol, m.sku


def test_stat_bounds_and_partition():
    res = analyze(_ctx(), draws=12)
    assert res.n_moves >= 1
    assert res.n_accept + res.n_hold == res.n_moves
    acc = sum(m.baseline_uplift_annual_eur for m in res.moves
              if m.verdict == "accept")
    held = sum(m.baseline_uplift_annual_eur for m in res.moves
               if m.verdict == "hold")
    assert abs(acc - res.accepted_baseline_uplift_eur) < 0.02
    assert abs(held - res.held_baseline_uplift_eur) < 0.02
    assert res.accepted_book_p10_eur <= res.accepted_book_p50_eur \
        <= res.accepted_book_p90_eur
    for m in res.moves:
        assert 0.0 <= m.carry_rate <= 1.0
        assert 0.0 <= m.direction_agreement <= m.carry_rate + 1e-9
        assert m.uplift_p10_eur <= m.uplift_p50_eur <= m.uplift_p90_eur
        # the gate rule is exactly reproduced by the published stats
        assert (m.verdict, m.reason) == gate(m.carry_rate, m.direction_agreement,
                                             m.uplift_p10_eur)


def test_collapses_to_base_case():
    """Zero uncertainty (degenerate bands) must land exactly on the base plan:
    every baseline move is accepted with agreement 1.0 and a point distribution
    at its own baseline €."""
    ctx = _ctx()
    res = analyze(ctx, draws=8, drivers=_FLAT)
    assert res.n_moves >= 1
    assert res.n_hold == 0
    assert res.held_baseline_uplift_eur == 0.0
    for m in res.moves:
        assert m.carry_rate == 1.0
        assert m.direction_agreement == 1.0
        assert m.verdict == "accept" and m.reason == "robust"
        assert m.uplift_p10_eur == m.uplift_p50_eur == m.uplift_p90_eur \
            == m.baseline_uplift_annual_eur
    book = sum(m.baseline_uplift_annual_eur for m in res.moves)
    for q in (res.accepted_book_p10_eur, res.accepted_book_p50_eur,
              res.accepted_book_p90_eur):
        assert abs(q - book) < 0.02
    assert abs(res.accepted_baseline_uplift_eur - book) < 0.02


def test_draw_loop_hand_checked(monkeypatch):
    """Stub the solver so the loop's bookkeeping is verifiable by hand: one +5%
    move on S1, carried in draws 0/2 and delisted in draws 1/3."""
    from types import SimpleNamespace

    s1 = make_sku(sku="S1", category="alpha", unit_cost_eur=10.0,
                  current_price_eur=20.0, avg_monthly_demand=100.0,
                  elasticity=-2.0)
    line = PriceLine(sku="S1", current_price=20.0, elasticity=-2.0,
                     unconstrained_optimal=None, recommended_price=21.0,
                     price_change_pct=5.0, expected_margin_now=1000.0,
                     expected_margin_new=1010.0, margin_uplift_eur=10.0,
                     flag="elastic")
    carried_sol = PlanSolution(
        assortment=SimpleNamespace(milp_uplift_vs_greedy_eur=0.0),
        prices=[line], promo=SimpleNamespace(incremental_margin_eur=0.0),
        price_inputs=[s1])
    empty_sol = PlanSolution(
        assortment=SimpleNamespace(milp_uplift_vs_greedy_eur=0.0),
        prices=[], promo=SimpleNamespace(incremental_margin_eur=0.0),
        price_inputs=[])

    def fake_solve(ctx, scenario):
        if scenario.name == "Baseline" or int(scenario.name[4:]) % 2 == 0:
            return carried_sol
        return empty_sol

    monkeypatch.setattr(rb, "solve_plan", fake_solve)
    ctx = PredictContext(skus=[s1], forecasts={"S1": 100.0},
                         elasticities={"S1": -2.0}, risks={"S1": 0.0})
    res = analyze(ctx, draws=4, drivers=_FLAT)

    # frozen +5% at p0=20, c=10, q0=100, e=-2:
    #   (21-10)*100*(21/20)^-2 - 1000 = 1100/1.1025 - 1000 = -2.2675736...
    u = round(((21.0 - 10.0) * 100.0 * (21.0 / 20.0) ** -2.0 - 1000.0) * 12, 2)
    assert res.n_moves == 1
    m = res.moves[0]
    assert m.sku == "S1" and m.direction == "raise"
    assert m.carry_rate == 0.5
    assert m.direction_agreement == 0.5
    assert m.baseline_uplift_annual_eur == u
    # per-draw series is (u, 0, u, 0) -> P10 = u, P50 = u/2, P90 = 0
    assert m.uplift_p10_eur == u
    assert abs(m.uplift_p50_eur - round(u / 2, 2)) <= 0.01
    assert m.uplift_p90_eur == 0.0
    assert (m.verdict, m.reason) == ("hold", "delist-risk")
    # totals bookkeeping: carried draws are worth 10*12, delisted draws 0
    assert res.draws_total_eur == [120.0, 0.0, 120.0, 0.0]
    assert res.n_accept == 0 and res.n_hold == 1
    assert res.accepted_book_p10_eur == res.accepted_book_p90_eur == 0.0
    assert res.held_baseline_uplift_eur == u


# --------------------------------------------------------------------------- #
# deterministic deliverables                                                  #
# --------------------------------------------------------------------------- #
def test_moves_csv_byte_identical(tmp_path):
    res = analyze(_ctx(), draws=8)
    a = write_moves_csv(res, tmp_path / "a.csv").read_bytes()
    b = write_moves_csv(res, tmp_path / "b.csv").read_bytes()
    assert a == b
    header = a.splitlines()[0].decode()
    assert header.startswith("sku,category,direction,")
    assert header.endswith("verdict,reason")
    assert len(a.splitlines()) == res.n_moves + 1


def test_moves_svg_byte_identical(tmp_path):
    res = analyze(_ctx(), draws=8)
    s1 = render_moves_svg(res, tmp_path / "a.svg").read_bytes()
    s2 = render_moves_svg(res, tmp_path / "b.svg").read_bytes()
    assert s1 == s2
    assert s1.lstrip().startswith(b"<svg")
    assert s1.rstrip().endswith(b"</svg>")


def test_result_shape_roundtrips():
    """The result dataclass is JSON-serializable via asdict (CLI --json)."""
    import json
    from dataclasses import asdict
    res = analyze(_ctx(), draws=4)
    assert isinstance(res, RobustnessResult)
    payload = json.loads(json.dumps(asdict(res)))
    assert payload["n_draws"] == 4
    assert len(payload["moves"]) == res.n_moves
