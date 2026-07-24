"""Tests for the predictive layer — forecaster, elasticity, decline risk.

The heavy forecaster training is exercised once through the shared `plan`
fixture (prescribe already backtests it), so these stay fast. The numpy
elasticity and decline models are cheap and tested directly.
"""
from __future__ import annotations

from revops.predict import elasticity as el
from revops.predict import risk as rk


def test_forecaster_beats_naive_mase(plan):
    """Sanity: the global forecaster's held-out MASE is below 1 (beats a
    seasonal-naive baseline) on the synthetic data."""
    fq = plan["model_quality"]["forecast"]
    assert fq["mase"] < 1.0
    assert fq["beats_naive"] is True


def test_elasticity_signs_are_negative(history):
    est = el.estimate_elasticities(history)
    assert est                                   # non-empty
    assert all(v < 0 for v in est.values())      # demand curves slope down


def test_elasticity_recovery_reasonable(history):
    rep = el.recovery_report(history)
    # recovered per-category elasticity within ~1 unit of the truth
    assert rep["mae_by_category"] < 1.0


def test_decline_risk_auc_above_threshold(history):
    rep = rk.quality_report(history)
    assert rep["roc_auc"] > 0.8
    assert 0.0 <= rep["pr_auc"] <= 1.0


def test_decline_scores_are_probabilities(history):
    scores = rk.score_decline_risk(history)
    assert scores
    assert all(0.0 <= p <= 1.0 for p in scores.values())
