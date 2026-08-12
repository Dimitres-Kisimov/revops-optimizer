"""Tests for the joint Monte-Carlo uplift risk band (revops/simulate.py).

Fast and key-free: every check runs on a tiny synthetic ``PredictContext`` (no
forecaster training) with a small draw count, so the whole file solves a handful
of small MILPs and finishes quickly. Determinism, the collapses-to-base-case
gate, and the downside-metric ordering are all hand-checked.
"""
from __future__ import annotations

import numpy as np
from conftest import make_sku

from revops.scenario import DEFAULT_DRIVERS, PredictContext, Scenario, evaluate
from revops.simulate import (
    DEFAULT_RISK_DRIVERS,
    RiskDriver,
    latin_hypercube,
    render_distribution_svg,
    simulate,
    summary_rows,
    triangular_ppf,
    write_summary_csv,
)


def _ctx() -> PredictContext:
    """A small, deterministic catalogue across three categories (mirrors the
    scenario tests so the two layers exercise the same core)."""
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
# samplers — hand-checked                                                      #
# --------------------------------------------------------------------------- #
def test_triangular_ppf_endpoints_and_monotone():
    u = np.linspace(0.0, 1.0, 101)
    v = triangular_ppf(u, 0.8, 1.0, 1.2)
    assert abs(v[0] - 0.8) < 1e-9        # u=0 -> lo
    assert abs(v[-1] - 1.2) < 1e-9       # u=1 -> hi
    assert np.all(np.diff(v) >= -1e-12)  # non-decreasing
    # symmetric triangle: the median sits at the mode
    median = float(triangular_ppf(np.array([0.5]), 0.8, 1.0, 1.2)[0])
    assert abs(median - 1.0) < 1e-9


def test_triangular_ppf_degenerate_band_is_constant():
    u = np.linspace(0.0, 1.0, 11)
    v = triangular_ppf(u, 1.0, 1.0, 1.0)
    assert np.allclose(v, 1.0)


def test_latin_hypercube_is_stratified():
    rng = np.random.default_rng(0)
    n, d = 32, 4
    g = latin_hypercube(n, d, rng)
    assert g.shape == (n, d)
    assert g.min() >= 0.0 and g.max() < 1.0
    # exactly one sample per stratum in every column
    for j in range(d):
        strata = np.floor(g[:, j] * n).astype(int)
        assert sorted(strata.tolist()) == list(range(n))


# --------------------------------------------------------------------------- #
# the simulation                                                              #
# --------------------------------------------------------------------------- #
def test_simulate_is_deterministic():
    ctx = _ctx()
    a = simulate(ctx, draws=24)
    b = simulate(ctx, draws=24)
    assert a.draws_total_eur == b.draws_total_eur
    assert a.total.__dict__ == b.total.__dict__
    assert (a.var10_eur, a.cvar10_eur, a.prob_at_or_above_hurdle) == \
           (b.var10_eur, b.cvar10_eur, b.prob_at_or_above_hurdle)


def test_baseline_point_matches_evaluate():
    ctx = _ctx()
    res = simulate(ctx, draws=16)
    base = evaluate(ctx, Scenario("Baseline")).total_uplift_eur
    assert abs(res.baseline_point_estimate_eur - round(base, 2)) < 0.01


def test_percentiles_are_ordered():
    res = simulate(_ctx(), draws=24)
    for c in (res.total, res.pricing, res.promo, res.assortment):
        seq = [c.min, c.p05, c.p10, c.p25, c.p50, c.p75, c.p90, c.p95, c.max]
        assert seq == sorted(seq), c.label
        assert c.std >= 0.0


def test_total_mean_is_sum_of_component_means():
    """total = pricing + promo + assortment on every draw, so expectation is
    additive: the mean of the total equals the sum of the component means."""
    res = simulate(_ctx(), draws=24)
    parts = res.pricing.mean + res.promo.mean + res.assortment.mean
    assert abs(parts - res.total.mean) < 1.0


def test_downside_metric_ordering():
    res = simulate(_ctx(), draws=28)
    # expected shortfall is a mean over the worst tail, so it cannot exceed the
    # VaR (P10) it is conditioned on, which in turn cannot exceed the median
    assert res.cvar10_eur <= res.var10_eur + 1e-6
    assert res.var10_eur <= res.total.p50 + 1e-6
    assert 0.0 <= res.prob_at_or_above_hurdle <= 1.0
    # headline-at-risk is the (non-negative) gap from the point estimate to P10
    assert res.downside_point_vs_p10_eur >= -1e-6


def test_hurdle_probability_is_monotone():
    ctx = _ctx()
    lo = simulate(ctx, draws=20, hurdle=0.0).prob_at_or_above_hurdle
    hi = simulate(ctx, draws=20, hurdle=1e12).prob_at_or_above_hurdle
    assert lo == 1.0            # every re-optimized draw clears a zero hurdle
    assert hi == 0.0            # none clears an impossible one
    assert hi <= lo


def test_collapses_to_base_case():
    """Zero uncertainty (degenerate bands) must land exactly on the base plan:
    the whole distribution collapses to the deterministic headline."""
    ctx = _ctx()
    flat = tuple(RiskDriver(d.name, d.field, 1.0, 1.0, 1.0)
                 for d in DEFAULT_RISK_DRIVERS)
    res = simulate(ctx, draws=16, drivers=flat)
    point = res.baseline_point_estimate_eur
    assert res.total.std == 0.0
    for q in (res.total.p10, res.total.p50, res.total.p90,
              res.total.min, res.total.max, res.var10_eur, res.cvar10_eur):
        assert abs(q - point) < 0.01
    assert res.prob_at_or_above_hurdle == 1.0
    assert res.downside_point_vs_p10_eur == 0.0
    assert res.upside_p90_vs_point_eur == 0.0


def test_risk_bands_match_scenario_driver_bands():
    """The triangular bands here must mirror scenario.py's one-way driver bands
    exactly, so the two robustness layers never drift apart."""
    by_field = {d.field: d for d in DEFAULT_DRIVERS}
    for rd in DEFAULT_RISK_DRIVERS:
        sd = by_field[rd.field]
        lo, hi = sorted((sd.a_value, sd.b_value))
        assert (rd.lo, rd.hi) == (lo, hi), rd.field
        assert rd.lo <= rd.mode <= rd.hi


# --------------------------------------------------------------------------- #
# deterministic deliverables                                                  #
# --------------------------------------------------------------------------- #
def test_summary_csv_byte_identical(tmp_path):
    res = simulate(_ctx(), draws=20)
    a = write_summary_csv(res, tmp_path / "a.csv").read_bytes()
    b = write_summary_csv(res, tmp_path / "b.csv").read_bytes()
    assert a == b
    assert a.splitlines()[0] == b"metric,value"
    # the tall table carries the headline risk scalars
    keys = {m for m, _ in summary_rows(res)}
    assert {"total_p10_eur", "total_p50_eur", "total_p90_eur",
            "total_var10_eur", "total_cvar10_eur",
            "total_prob_at_or_above_hurdle"} <= keys


def test_distribution_svg_byte_identical(tmp_path):
    res = simulate(_ctx(), draws=24)
    s1 = render_distribution_svg(res, tmp_path / "a.svg").read_bytes()
    s2 = render_distribution_svg(res, tmp_path / "b.svg").read_bytes()
    assert s1 == s2
    assert s1.lstrip().startswith(b"<svg")
    assert s1.rstrip().endswith(b"</svg>")
    # the honesty label and the band/downside figures survive any redesign
    assert b"illustrative, not a forecast" in s1
    assert b"P10-P90 band" in s1
    assert b"VaR(10%)" in s1 and b"CVaR(10%)" in s1
