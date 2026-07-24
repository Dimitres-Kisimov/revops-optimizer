"""Shared fixtures. The prescribe() run (which trains the forecaster) happens
once per session and is reused, keeping the whole suite fast (<60s)."""
from __future__ import annotations

import pytest

from revops.dataio import SKU


def make_sku(sku="S1", category="cat", unit_cost_eur=10.0, current_price_eur=20.0,
             avg_monthly_demand=100.0, demand_std=20.0, lead_time_months=1.0,
             lead_time_std=0.2, moq=1, shelf_m3_per_unit=0.01, elasticity=-2.0,
             annual_holding_rate=0.24, demand_risk=0.1) -> SKU:
    """Factory for a fully-specified SKU with sensible defaults."""
    return SKU(sku=sku, category=category, unit_cost_eur=unit_cost_eur,
               current_price_eur=current_price_eur,
               avg_monthly_demand=avg_monthly_demand, demand_std=demand_std,
               lead_time_months=lead_time_months, lead_time_std=lead_time_std,
               moq=moq, shelf_m3_per_unit=shelf_m3_per_unit, elasticity=elasticity,
               annual_holding_rate=annual_holding_rate, demand_risk=demand_risk)


@pytest.fixture(scope="session")
def plan() -> dict:
    """One real prescribe() run, shared across the session."""
    from revops.prescribe import prescribe
    return prescribe(verbose=False)


@pytest.fixture(scope="session")
def history() -> dict:
    from revops.predict.forecast import load_history
    return load_history()
