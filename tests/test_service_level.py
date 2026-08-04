"""Tests for the forecast-uncertainty -> service-level (fill-rate) layer.

These are fast and key-free: the sweep is exercised on deterministic synthetic
residuals (no forecaster training), and one integration check reuses the shared
prescribe() plan fixture.
"""
from __future__ import annotations

from statistics import NormalDist

from conftest import make_sku

from revops.optimize.service_level import (
    DEFAULT_LEVELS,
    demand_uncertainty,
    recommend_service_level,
    render_curve_svg,
    service_level_curve,
    write_curve_csv,
)

# a small, deterministic, mildly fat-tailed residual set (no RNG)
_RESID = [-9.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 11.0]


def _skus_and_residuals():
    skus = [make_sku(sku=f"S{i}", avg_monthly_demand=80.0 + 20 * i,
                     unit_cost_eur=10.0 + i, current_price_eur=25.0 + 2 * i,
                     annual_holding_rate=0.24) for i in range(4)]
    residuals = {f"S{i}": [r * (1.0 + 0.1 * i) for r in _RESID] for i in range(4)}
    return skus, residuals


# --------------------------------------------------------------------------- #
# empirical demand uncertainty                                                 #
# --------------------------------------------------------------------------- #
def test_demand_uncertainty_pools_and_sorts():
    _, residuals = _skus_and_residuals()
    unc = demand_uncertainty(residuals)
    assert set(unc["sigma_by_sku"]) == set(residuals)
    assert all(v > 0 for v in unc["sigma_by_sku"].values())
    pooled = unc["pooled_standardized"]
    assert unc["n_pooled"] == sum(len(v) for v in residuals.values())
    # sorted ascending
    assert list(pooled) == sorted(pooled)
    # standardized -> unit-ish spread
    assert 0.5 < unc["pooled_std"] < 2.0


# --------------------------------------------------------------------------- #
# the swept curve                                                              #
# --------------------------------------------------------------------------- #
def test_curve_length_and_levels():
    skus, residuals = _skus_and_residuals()
    curve = service_level_curve(skus, residuals)
    assert len(curve) == len(DEFAULT_LEVELS)
    assert [p.service_level for p in curve] == list(DEFAULT_LEVELS)


def test_curve_monotonicity_and_bounds():
    skus, residuals = _skus_and_residuals()
    curve = service_level_curve(skus, residuals)
    for a, b in zip(curve, curve[1:], strict=False):
        # higher service level costs more safety stock / holding, buys more fill
        assert b.empirical_z >= a.empirical_z - 1e-9
        assert b.safety_stock_units >= a.safety_stock_units - 1e-6
        assert b.holding_cost_eur_month >= a.holding_cost_eur_month - 1e-6
        assert b.expected_fill_rate >= a.expected_fill_rate - 1e-9
        # ... and fewer expected stockouts
        assert b.expected_stockout_units_month <= a.expected_stockout_units_month + 1e-6
    for p in curve:
        assert 0.0 <= p.expected_fill_rate <= 1.0
        assert p.total_cost_eur_month == round(
            p.holding_cost_eur_month + p.expected_stockout_cost_eur_month, 2)


def test_gaussian_z_matches_normal_dist():
    skus, residuals = _skus_and_residuals()
    curve = service_level_curve(skus, residuals)
    for p in curve:
        assert abs(p.gaussian_z - NormalDist().inv_cdf(p.service_level)) < 1e-3


def test_recommendation_is_the_cost_minimiser():
    skus, residuals = _skus_and_residuals()
    curve = service_level_curve(skus, residuals)
    reco = recommend_service_level(curve)
    cheapest = min(curve, key=lambda p: p.total_cost_eur_month)
    assert reco["service_level"] == cheapest.service_level
    assert reco["total_cost_eur_month"] == cheapest.total_cost_eur_month
    assert "illustrative" in reco["read"].lower()
    assert 0.80 <= reco["service_level"] <= 0.99


def test_penalty_multiplier_raises_recommended_level():
    skus, residuals = _skus_and_residuals()
    lo = recommend_service_level(service_level_curve(skus, residuals,
                                 stockout_penalty_mult=1.0))
    hi = recommend_service_level(service_level_curve(skus, residuals,
                                 stockout_penalty_mult=4.0))
    assert hi["service_level"] >= lo["service_level"]


def test_curve_is_deterministic():
    skus, residuals = _skus_and_residuals()
    a = service_level_curve(skus, residuals)
    b = service_level_curve(skus, residuals)
    assert [p.__dict__ for p in a] == [p.__dict__ for p in b]


# --------------------------------------------------------------------------- #
# deliverables are byte-stable                                                 #
# --------------------------------------------------------------------------- #
def test_csv_and_svg_are_byte_identical_across_runs(tmp_path):
    skus, residuals = _skus_and_residuals()
    curve = service_level_curve(skus, residuals)
    reco = recommend_service_level(curve)
    c1 = write_curve_csv(curve, tmp_path / "a.csv").read_bytes()
    c2 = write_curve_csv(curve, tmp_path / "b.csv").read_bytes()
    assert c1 == c2
    s1 = render_curve_svg(curve, reco, tmp_path / "a.svg").read_bytes()
    s2 = render_curve_svg(curve, reco, tmp_path / "b.svg").read_bytes()
    assert s1 == s2
    assert s1.lstrip().startswith(b"<svg")
    assert s1.rstrip().endswith(b"</svg>")


# --------------------------------------------------------------------------- #
# integration: it rides the real prescribe() plan                             #
# --------------------------------------------------------------------------- #
def test_plan_carries_service_level_layer(plan):
    inv = plan["plan"]["inventory"]
    curve = inv["service_level_curve"]
    reco = inv["recommended_service_level"]
    assert len(curve) == len(DEFAULT_LEVELS)
    assert 0.80 <= reco["service_level"] <= 0.99
    assert 0.0 < reco["expected_fill_rate"] <= 1.0
    assert reco["read"]
    # this forecaster's real residuals are fat-tailed: hitting a high service
    # level needs more buffer than a bell curve would (empirical z > Gaussian z),
    # so a normal assumption would silently under-stock.
    at95 = next(p for p in curve if abs(p["service_level"] - 0.95) < 1e-9)
    assert at95["empirical_z"] > at95["gaussian_z"]
