"""service_level.py — forecast uncertainty -> service-level (fill-rate) curve.

The newsvendor in ``inventory.py`` sets one order-up-to level from a *point*
forecast and the SKU's static ``demand_std``. But the number a planner actually
turns is the **target service level**, and the honest input to it is not the raw
historical spread — it is how wrong the *forecast* tends to be. This module
closes that seam:

  1. take the forecaster's own one-step residuals
     (``revops.predict.forecast.one_step_residuals``) as the empirical
     demand-uncertainty — no distributional assumption, no new heavy deps;
  2. standardize them per SKU and pool them, so the *shape* of the error
     distribution (fat tails, bias) comes from data, not a bell curve;
  3. sweep the target service level (80 -> 99%) and, at each point, report the
     required safety stock, the holding cost of carrying it, the expected unit
     fill rate, and the expected stockout cost — then pick the service level
     that minimizes total illustrative cost.

The cost rates are an **illustrative single-period (monthly) model**, not a
guarantee: overage = one month's holding on a safety-stock unit (each SKU's own
``annual_holding_rate``); underage = the lost unit margin on a missed sale,
optionally scaled by a goodwill/expedite penalty. See ``docs/OPTIMIZATION_MODELS.md``.

    from revops.predict.forecast import train, load_history, one_step_residuals
    from revops.optimize.service_level import service_level_curve, recommend_service_level
    hist = load_history(); model = train(hist, verbose=False)
    curve = service_level_curve(load(), one_step_residuals(model, hist))
    reco = recommend_service_level(curve)

Author: Dimitres Kisimov.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from ..dataio import SKU

# --------------------------------------------------------------------------- #
# illustrative cost model — labelled, NOT a guarantee                          #
# --------------------------------------------------------------------------- #
# A missed sale costs the lost unit margin; a goodwill/expedite penalty can be
# layered on top. Default 1.0 keeps the underage cost grounded purely in the
# data-derived margin (pure newsvendor); raising it pushes the recommended level
# up. The holding (overage) rate is each SKU's own annual_holding_rate / 12.
DEFAULT_STOCKOUT_PENALTY_MULT = 1.0        # x lost unit margin (illustrative)
# swept target service levels: 80% -> 99% in 1-point steps
DEFAULT_LEVELS: tuple[float, ...] = tuple(round(0.80 + 0.01 * i, 2) for i in range(20))


@dataclass
class ServiceLevelPoint:
    service_level: float
    empirical_z: float                     # residual quantile at SL (sigma units)
    gaussian_z: float                      # textbook Phi^-1(SL), for comparison
    safety_stock_units: float
    safety_capital_eur: float
    holding_cost_eur_month: float
    expected_fill_rate: float
    expected_stockout_units_month: float
    expected_stockout_cost_eur_month: float
    total_cost_eur_month: float


# --------------------------------------------------------------------------- #
# empirical demand uncertainty from forecast residuals                         #
# --------------------------------------------------------------------------- #
def demand_uncertainty(residuals: dict[str, list[float]]) -> dict:
    """Turn per-SKU forecast residuals into empirical demand-uncertainty.

    Returns the per-SKU residual sigma (absolute units) and the *pooled
    standardized* residuals (each SKU's residuals divided by its own sigma),
    whose quantiles describe the error-distribution shape across the catalogue.
    """
    sigma_by_sku: dict[str, float] = {}
    pooled: list[float] = []
    for sku, res in residuals.items():
        arr = np.asarray(res, dtype=np.float64)
        sigma = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        sigma_by_sku[sku] = sigma
        if sigma > 1e-9:
            pooled.extend((arr / sigma).tolist())
    pooled_arr = np.sort(np.asarray(pooled, dtype=np.float64))
    return {
        "sigma_by_sku": sigma_by_sku,
        "pooled_standardized": pooled_arr,
        "n_pooled": int(pooled_arr.size),
        "bias_mean": float(pooled_arr.mean()) if pooled_arr.size else 0.0,
        "pooled_std": float(pooled_arr.std(ddof=1)) if pooled_arr.size > 1 else 0.0,
    }


def _empirical_z(pooled: np.ndarray, sl: float) -> float:
    """Empirical SL-quantile of the pooled standardized residuals (the safety
    factor, in sigma units, that a Gaussian would call Phi^-1(SL))."""
    if pooled.size == 0:
        return NormalDist().inv_cdf(sl)
    return float(np.quantile(pooled, sl, method="linear"))


def _standardized_shortage(pooled: np.ndarray, b: float) -> float:
    """Empirical E[(r - b)+] over the pooled standardized residuals: expected
    units short per period, in sigma units, if the buffer is b sigmas."""
    if pooled.size == 0:
        return 0.0
    return float(np.mean(np.maximum(0.0, pooled - b)))


# --------------------------------------------------------------------------- #
# the service-level sweep                                                       #
# --------------------------------------------------------------------------- #
def service_level_curve(skus: list[SKU], residuals: dict[str, list[float]],
                        levels: tuple[float, ...] = DEFAULT_LEVELS,
                        stockout_penalty_mult: float = DEFAULT_STOCKOUT_PENALTY_MULT
                        ) -> list[ServiceLevelPoint]:
    """Sweep target service level over the managed SKUs, using the empirical
    forecast-error distribution. Portfolio-aggregated per period (month).

    The shape of the error distribution is pooled across *every* SKU present in
    ``residuals``; the per-SKU scale (sigma) and the euro economics are computed
    for the SKUs passed in ``skus`` (e.g. the carried assortment).
    """
    unc = demand_uncertainty(residuals)
    sigma_by_sku = unc["sigma_by_sku"]
    pooled = unc["pooled_standardized"]

    managed = [s for s in skus if sigma_by_sku.get(s.sku, 0.0) > 1e-9]
    total_demand = sum(s.avg_monthly_demand for s in managed) or 1.0

    out: list[ServiceLevelPoint] = []
    for sl in levels:
        b = _empirical_z(pooled, sl)
        short_std = _standardized_shortage(pooled, b)
        ss_units = hold = short_units = short_cost = safety_cap = 0.0
        for s in managed:
            sigma = sigma_by_sku[s.sku]
            ss = b * sigma
            ss_units += ss
            safety_cap += ss * s.unit_cost_eur
            hold += ss * s.unit_cost_eur * (s.annual_holding_rate / 12.0)
            su = sigma * short_std
            short_units += su
            short_cost += su * s.unit_margin_eur * stockout_penalty_mult
        fill = 1.0 - short_units / total_demand
        out.append(ServiceLevelPoint(
            service_level=round(sl, 4),
            empirical_z=round(b, 4),
            gaussian_z=round(NormalDist().inv_cdf(sl), 4),
            safety_stock_units=round(ss_units, 2),
            safety_capital_eur=round(safety_cap, 2),
            holding_cost_eur_month=round(hold, 2),
            expected_fill_rate=round(fill, 5),
            expected_stockout_units_month=round(short_units, 2),
            expected_stockout_cost_eur_month=round(short_cost, 2),
            total_cost_eur_month=round(hold + short_cost, 2),
        ))
    return out


def recommend_service_level(curve: list[ServiceLevelPoint]) -> dict:
    """Pick the swept service level that minimizes total illustrative cost and
    write a plain-language read of the trade-off it strikes."""
    best = min(curve, key=lambda p: p.total_cost_eur_month)
    sl_pct = best.service_level * 100
    read = (
        f"Recommended target service level: {sl_pct:.0f}%. Derived from the "
        f"forecaster's own residuals (not a bell-curve assumption), this level "
        f"holds ~{best.safety_stock_units:,.0f} units of safety stock "
        f"(EUR {best.safety_capital_eur:,.0f} capital, EUR "
        f"{best.holding_cost_eur_month:,.0f}/mo to carry) and yields an expected "
        f"unit fill rate of {best.expected_fill_rate * 100:.1f}% - i.e. a "
        f"{sl_pct:.0f}% cycle service level still meets "
        f"{best.expected_fill_rate * 100:.1f}% of demand, because most stockout "
        f"cycles miss by only a little. It minimizes total illustrative cost "
        f"(holding EUR {best.holding_cost_eur_month:,.0f}/mo + expected stockout "
        f"EUR {best.expected_stockout_cost_eur_month:,.0f}/mo = EUR "
        f"{best.total_cost_eur_month:,.0f}/mo). Cost rates are illustrative, not "
        f"a guarantee."
    )
    return {
        "service_level": best.service_level,
        "expected_fill_rate": best.expected_fill_rate,
        "safety_stock_units": best.safety_stock_units,
        "safety_capital_eur": best.safety_capital_eur,
        "holding_cost_eur_month": best.holding_cost_eur_month,
        "expected_stockout_cost_eur_month": best.expected_stockout_cost_eur_month,
        "total_cost_eur_month": best.total_cost_eur_month,
        "empirical_z": best.empirical_z,
        "gaussian_z": best.gaussian_z,
        "read": read,
    }


# --------------------------------------------------------------------------- #
# deterministic deliverables — CSV + hand-drawn SVG                            #
# --------------------------------------------------------------------------- #
_CSV_COLS = [
    "service_level", "empirical_z", "gaussian_z", "safety_stock_units",
    "safety_capital_eur", "holding_cost_eur_month", "expected_fill_rate",
    "expected_stockout_units_month", "expected_stockout_cost_eur_month",
    "total_cost_eur_month",
]


def write_curve_csv(curve: list[ServiceLevelPoint], path: str | Path) -> Path:
    """Write the swept curve to a byte-stable CSV (no wall-clock, no RNG)."""
    import csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLS)
        for pt in curve:
            row = asdict(pt)
            w.writerow([row[c] for c in _CSV_COLS])
    return p


# palette shared with report.py / the web dashboard (one project system)
_BLUE, _PINK, _GREEN, _INK, _GREY = (
    "#2f6bff", "#ea4b71", "#1d9e6f", "#1f2933", "#9aa5b1")


def _nice_top(v: float) -> float:
    """Round an axis maximum up to a clean 1/2/5 x 10^k step."""
    if v <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def render_curve_svg(curve: list[ServiceLevelPoint], recommended: dict,
                     path: str | Path) -> Path:
    """Hand-drawn SVG: two stacked panels sharing the service-level x-axis (one
    y-axis each — never a dual axis). Top: illustrative cost per month (total,
    holding, stockout). Bottom: expected unit fill rate. Deterministic output."""
    W, H = 760, 520
    ml, mr = 70, 24
    plot_w = W - ml - mr
    x0 = curve[0].service_level * 100
    x1 = curve[-1].service_level * 100
    reco_x = recommended["service_level"] * 100

    def sx(sl_pct: float) -> float:
        return ml + (sl_pct - x0) / (x1 - x0) * plot_w

    # ---- top panel: cost (EUR/month) ----
    t_top, t_h = 54, 200
    cost_max = _nice_top(max(p.total_cost_eur_month for p in curve))

    def ty(v: float) -> float:
        return t_top + t_h - (v / cost_max) * t_h

    # ---- bottom panel: fill rate (%) ----
    b_top, b_h = 320, 150
    fr_vals = [p.expected_fill_rate * 100 for p in curve]
    fr_lo = min(80.0, min(fr_vals) - 1)
    fr_hi = 100.0

    def by(v: float) -> float:
        return b_top + b_h - (v - fr_lo) / (fr_hi - fr_lo) * b_h

    def poly(sel) -> str:
        return " ".join(f"{sx(p.service_level * 100):.1f},{sel(p):.1f}" for p in curve)

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{ml}" y="26" font-size="16" font-weight="700" '
             f'fill="{_INK}">Forecast uncertainty - service-level trade-off</text>')
    s.append(f'<text x="{ml}" y="44" font-size="11" fill="{_GREY}">Empirical '
             f'forecast residuals; illustrative monthly cost rates (not a guarantee)</text>')

    # top gridlines + y labels
    for i in range(5):
        v = cost_max * i / 4
        y = ty(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" '
                 f'stroke="#eef1f5" stroke-width="1"/>')
        s.append(f'<text x="{ml - 6}" y="{y + 3:.1f}" font-size="9" '
                 f'text-anchor="end" fill="{_GREY}">{v:,.0f}</text>')
    s.append(f'<text x="{ml}" y="{t_top - 8}" font-size="10" fill="{_INK}" '
             f'font-weight="600">Cost EUR / month</text>')

    # recommended vertical marker (both panels)
    rx = sx(reco_x)
    s.append(f'<line x1="{rx:.1f}" y1="{t_top}" x2="{rx:.1f}" y2="{b_top + b_h}" '
             f'stroke="{_INK}" stroke-width="1" stroke-dasharray="4 3" opacity="0.55"/>')
    s.append(f'<text x="{rx:.1f}" y="{t_top - 8}" font-size="10" text-anchor="middle" '
             f'font-weight="700" fill="{_INK}">reco {reco_x:.0f}%</text>')

    # cost series: total (INK, thick), holding (BLUE), stockout (PINK, dashed)
    s.append(f'<polyline points="{poly(lambda p: ty(p.holding_cost_eur_month))}" '
             f'fill="none" stroke="{_BLUE}" stroke-width="2"/>')
    s.append(f'<polyline points="{poly(lambda p: ty(p.expected_stockout_cost_eur_month))}" '
             f'fill="none" stroke="{_PINK}" stroke-width="2" stroke-dasharray="5 3"/>')
    s.append(f'<polyline points="{poly(lambda p: ty(p.total_cost_eur_month))}" '
             f'fill="none" stroke="{_INK}" stroke-width="2.6"/>')
    # highlight recommended total-cost point
    rp = min(curve, key=lambda p: abs(p.service_level * 100 - reco_x))
    s.append(f'<circle cx="{rx:.1f}" cy="{ty(rp.total_cost_eur_month):.1f}" r="4.5" '
             f'fill="white" stroke="{_INK}" stroke-width="2"/>')
    # legend (identity not by color alone: each line labelled)
    lx, ly = ml + 8, t_top + 12
    for label, color, dash in (("Total", _INK, ""), ("Holding", _BLUE, ""),
                               ("Stockout", _PINK, "5 3")):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" '
                 f'stroke-width="2.4"{da}/>')
        s.append(f'<text x="{lx + 23}" y="{ly + 3}" font-size="10" fill="{_INK}">{label}</text>')
        lx += 78

    # bottom panel: fill rate
    for i in range(5):
        v = fr_lo + (fr_hi - fr_lo) * i / 4
        y = by(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" '
                 f'stroke="#eef1f5" stroke-width="1"/>')
        s.append(f'<text x="{ml - 6}" y="{y + 3:.1f}" font-size="9" '
                 f'text-anchor="end" fill="{_GREY}">{v:.0f}%</text>')
    s.append(f'<text x="{ml}" y="{b_top - 8}" font-size="10" fill="{_INK}" '
             f'font-weight="600">Expected unit fill rate</text>')
    s.append(f'<polyline points="{poly(lambda p: by(p.expected_fill_rate * 100))}" '
             f'fill="none" stroke="{_GREEN}" stroke-width="2.6"/>')
    s.append(f'<circle cx="{rx:.1f}" cy="{by(rp.expected_fill_rate * 100):.1f}" r="4.5" '
             f'fill="white" stroke="{_GREEN}" stroke-width="2"/>')
    s.append(f'<text x="{rx + 7:.1f}" y="{by(rp.expected_fill_rate * 100) - 6:.1f}" '
             f'font-size="10" font-weight="700" fill="{_GREEN}">'
             f'{rp.expected_fill_rate * 100:.1f}%</text>')

    # shared x-axis ticks (service level %)
    for p in curve:
        sl_pct = p.service_level * 100
        if int(round(sl_pct)) % 5 == 0:
            x = sx(sl_pct)
            s.append(f'<text x="{x:.1f}" y="{b_top + b_h + 16:.1f}" font-size="9" '
                     f'text-anchor="middle" fill="{_GREY}">{sl_pct:.0f}%</text>')
    s.append(f'<text x="{ml + plot_w / 2:.1f}" y="{H - 6}" font-size="10" '
             f'text-anchor="middle" fill="{_INK}">Target service level</text>')
    s.append('</svg>')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s) + "\n", encoding="utf-8")
    return p
