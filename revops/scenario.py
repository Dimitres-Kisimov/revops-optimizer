"""scenario.py — what-if scenarios & one-way sensitivity (robustness of the €).

The rest of the project answers *what should we do* and lands on a single number:
the total expected € uplift. A planner's next question is always *how much should
I trust that number, and which assumption moves it most?* This module answers
both, without re-inventing a single optimizer:

  * **Scenario library** — a handful of named business scenarios (demand
    downturn, cost inflation, tight capital, promo push, a more-elastic market)
    are each pushed back through the *existing* predict→optimize core and the
    headline uplift is recomputed and compared to the baseline plan.
  * **One-way sensitivity (a tornado)** — each driver is swept across a plausible
    planning band, one at a time, holding everything else at baseline, and the
    swing in total uplift is ranked. The widest bar is the assumption the plan's
    € is most exposed to — i.e. the thing to de-risk first.

The predictive layer (forecaster, elasticity ridge, decline classifier) is run
**once**; every scenario then perturbs its *outputs* — a demand multiplier on the
forecast, a cost multiplier on the SKU master, an elasticity/​risk multiplier —
and re-solves the assortment MILP, the elasticity pricing and the concave promo
allocation exactly as ``prescribe()`` does (it even reuses ``prescribe``'s own
predict→optimize adapter, so the *Baseline* scenario reproduces the headline
plan). Deterministic: no RNG, no wall-clock — byte-identical across re-runs.

The perturbation bands are **illustrative planning ranges, not a forecast of
what will happen**; the resulting uplift figures are what the model computes
under each assumption, not a guarantee.

    from revops.scenario import predict_context, scenario_library, sensitivity_tornado
    ctx = predict_context(verbose=False)
    lib = scenario_library(ctx)                 # named scenarios vs baseline
    tornado = sensitivity_tornado(ctx)          # ranked driver sensitivity

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .dataio import SKU, load
from .optimize.assortment import optimize_assortment
from .optimize.pricing import optimize_prices
from .optimize.promo import optimize_promo
from .prescribe import _adjusted_skus  # reuse the exact predict->optimize adapter

# baseline knobs mirror prescribe()'s defaults so the Baseline scenario matches
# the headline plan exactly.
BASELINE_BUDGET = 60_000.0
BASELINE_SHELF_M3: float | None = 10.0
BASELINE_PROMO_BUDGET = 40_000.0
BASELINE_PRICE_GUARDRAIL = 0.15
MIN_PER_CATEGORY = 3


# --------------------------------------------------------------------------- #
# predict once, perturb many                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class PredictContext:
    """The trained predictive layer's outputs, computed once and reused by every
    scenario. Tests can build one directly from tiny synthetic dicts (no
    training); the CLI builds one from the real models via ``predict_context``."""
    skus: list[SKU]
    forecasts: dict[str, float]
    elasticities: dict[str, float]
    risks: dict[str, float]


def predict_context(verbose: bool = False) -> PredictContext:
    """Run the real predictive layer once (trains the forecaster) and return its
    outputs. This is the only expensive step; every scenario reuses it."""
    from .predict import elasticity as el
    from .predict import forecast as fc
    from .predict import risk as rk

    history = fc.load_history()
    skus = load()
    model = fc.train(history, verbose=verbose)
    forecasts = {sku: fc.forecast_next(model, rows) for sku, rows in history.items()}
    elasticities = el.estimate_elasticities(history)
    risks = rk.score_decline_risk(history)
    return PredictContext(skus=skus, forecasts=forecasts,
                          elasticities=elasticities, risks=risks)


@dataclass(frozen=True)
class Scenario:
    """A named what-if. The four multipliers perturb the *predictions*; the four
    knobs are the decision levers ``prescribe()`` already exposes. All defaults
    reproduce the baseline plan."""
    name: str
    demand_mult: float = 1.0            # x forecast demand (market up/downturn)
    cost_mult: float = 1.0             # x unit cost (input-price inflation)
    elasticity_mult: float = 1.0        # x |elasticity| estimate (est. uncertainty)
    risk_mult: float = 1.0             # x decline probability (risk severity)
    budget: float = BASELINE_BUDGET
    shelf_capacity_m3: float | None = BASELINE_SHELF_M3
    promo_budget: float = BASELINE_PROMO_BUDGET
    price_guardrail: float = BASELINE_PRICE_GUARDRAIL


@dataclass
class ScenarioResult:
    name: str
    total_uplift_eur: float
    pricing_uplift_eur: float
    promo_uplift_eur: float
    assortment_uplift_eur: float
    delta_vs_baseline_eur: float
    n_carried: int
    capital_used_eur: float
    inputs: dict


def evaluate(ctx: PredictContext, scenario: Scenario) -> ScenarioResult:
    """Apply a scenario's perturbations to the (already computed) predictions and
    re-solve the optimization core, returning the recomputed uplift.

    The uplift is assembled with the *identical* arithmetic ``prescribe()`` uses
    (components rounded to cents, the total summed from the unrounded parts)."""
    pskus = [replace(s, unit_cost_eur=max(0.01, s.unit_cost_eur * scenario.cost_mult))
             for s in ctx.skus]
    p_forecasts = {k: v * scenario.demand_mult for k, v in ctx.forecasts.items()}
    p_elastic = {k: v * scenario.elasticity_mult for k, v in ctx.elasticities.items()}
    p_risks = {k: min(1.0, max(0.0, v * scenario.risk_mult)) for k, v in ctx.risks.items()}

    for_ops, for_price = _adjusted_skus(pskus, p_forecasts, p_elastic, p_risks)

    assort = optimize_assortment(for_ops, scenario.budget,
                                 min_per_category=MIN_PER_CATEGORY,
                                 shelf_capacity_m3=scenario.shelf_capacity_m3)
    carried = set(assort.carried)
    carried_price = [s for s in for_price if s.sku in carried]
    carried_ops = [s for s in for_ops if s.sku in carried]

    prices = optimize_prices(carried_price, scenario.price_guardrail)
    promo = optimize_promo(carried_ops, scenario.promo_budget)

    pricing_annual = sum(p.margin_uplift_eur for p in prices) * 12
    promo_inc = promo.incremental_margin_eur
    assort_gap = assort.milp_uplift_vs_greedy_eur
    total = pricing_annual + promo_inc + assort_gap

    return ScenarioResult(
        name=scenario.name,
        total_uplift_eur=round(total, 2),
        pricing_uplift_eur=round(pricing_annual, 2),
        promo_uplift_eur=round(promo_inc, 2),
        assortment_uplift_eur=round(assort_gap, 2),
        delta_vs_baseline_eur=0.0,        # filled by scenario_library
        n_carried=assort.n_carried,
        capital_used_eur=assort.capital_used_eur,
        inputs={
            "demand_mult": scenario.demand_mult, "cost_mult": scenario.cost_mult,
            "elasticity_mult": scenario.elasticity_mult, "risk_mult": scenario.risk_mult,
            "budget": scenario.budget, "shelf_capacity_m3": scenario.shelf_capacity_m3,
            "promo_budget": scenario.promo_budget,
            "price_guardrail": scenario.price_guardrail,
        },
    )


# --------------------------------------------------------------------------- #
# the named scenario library                                                   #
# --------------------------------------------------------------------------- #
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Baseline"),
    Scenario("Demand downturn -15%", demand_mult=0.85, risk_mult=1.30),
    Scenario("Cost inflation +8%", cost_mult=1.08),
    Scenario("Tight capital -25%", budget=BASELINE_BUDGET * 0.75,
             shelf_capacity_m3=(BASELINE_SHELF_M3 or 10.0) * 0.80),
    Scenario("Promo push +50%", promo_budget=BASELINE_PROMO_BUDGET * 1.50),
    Scenario("More elastic market +20%", elasticity_mult=1.20),
)


def scenario_library(ctx: PredictContext,
                     scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS
                     ) -> list[ScenarioResult]:
    """Evaluate every named scenario and stamp its delta vs the first (baseline).
    The first scenario is treated as the baseline reference."""
    results = [evaluate(ctx, sc) for sc in scenarios]
    base = results[0].total_uplift_eur
    return [replace(r, delta_vs_baseline_eur=round(r.total_uplift_eur - base, 2))
            for r in results]


# --------------------------------------------------------------------------- #
# one-way sensitivity (tornado)                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Driver:
    """One assumption to sweep. ``a_value``/``b_value`` are the low/high ends of
    a plausible planning band applied to ``field``; the labels are for display."""
    name: str
    field: str
    a_value: float
    b_value: float
    a_label: str
    b_label: str


# Plausible planning bands (illustrative, not a forecast). Absolute knob bands
# are ±25% of the baseline value; the prediction multipliers use their own
# per-driver ranges (demand ±15%, cost ±8%, elasticity ±20%, risk ±30%).
DEFAULT_DRIVERS: tuple[Driver, ...] = (
    Driver("Demand level", "demand_mult", 0.85, 1.15, "-15%", "+15%"),
    Driver("Unit cost", "cost_mult", 1.08, 0.92, "+8%", "-8%"),
    Driver("Elasticity estimate", "elasticity_mult", 0.80, 1.20, "-20%", "+20%"),
    Driver("Capital budget", "budget", BASELINE_BUDGET * 0.75,
           BASELINE_BUDGET * 1.25, "-25%", "+25%"),
    Driver("Shelf capacity", "shelf_capacity_m3", (BASELINE_SHELF_M3 or 10.0) * 0.75,
           (BASELINE_SHELF_M3 or 10.0) * 1.25, "-25%", "+25%"),
    Driver("Promo budget", "promo_budget", BASELINE_PROMO_BUDGET * 0.75,
           BASELINE_PROMO_BUDGET * 1.25, "-25%", "+25%"),
    Driver("Price guardrail", "price_guardrail", 0.10, 0.20, "band 10%", "band 20%"),
    Driver("Decline risk", "risk_mult", 1.30, 0.70, "+30%", "-30%"),
)


@dataclass
class TornadoRow:
    driver: str
    baseline_uplift_eur: float
    a_label: str
    a_uplift_eur: float
    b_label: str
    b_uplift_eur: float
    low_uplift_eur: float
    high_uplift_eur: float
    downside_eur: float          # baseline - low  (>= 0)
    upside_eur: float            # high - baseline (>= 0)
    swing_eur: float             # high - low  (the tornado bar width)


def sensitivity_tornado(ctx: PredictContext,
                        drivers: tuple[Driver, ...] = DEFAULT_DRIVERS,
                        baseline: float | None = None) -> list[TornadoRow]:
    """Sweep each driver one at a time over its band and rank the swing in total
    uplift (widest first). ``baseline`` defaults to the unperturbed plan."""
    base = (baseline if baseline is not None
            else evaluate(ctx, Scenario("Baseline")).total_uplift_eur)
    ref = Scenario("Baseline")
    rows: list[TornadoRow] = []
    for d in drivers:
        ua = evaluate(ctx, replace(ref, **{d.field: d.a_value})).total_uplift_eur
        ub = evaluate(ctx, replace(ref, **{d.field: d.b_value})).total_uplift_eur
        low, high = (ua, ub) if ua <= ub else (ub, ua)
        rows.append(TornadoRow(
            driver=d.name, baseline_uplift_eur=round(base, 2),
            a_label=d.a_label, a_uplift_eur=round(ua, 2),
            b_label=d.b_label, b_uplift_eur=round(ub, 2),
            low_uplift_eur=round(low, 2), high_uplift_eur=round(high, 2),
            downside_eur=round(base - low, 2), upside_eur=round(high - base, 2),
            swing_eur=round(high - low, 2)))
    rows.sort(key=lambda r: (-r.swing_eur, r.driver))
    return rows


# --------------------------------------------------------------------------- #
# deterministic deliverables — CSV + hand-drawn tornado SVG                    #
# --------------------------------------------------------------------------- #
_SCEN_COLS = ["scenario", "total_uplift_eur", "pricing_uplift_eur",
              "promo_uplift_eur", "assortment_uplift_eur", "delta_vs_baseline_eur",
              "n_carried", "capital_used_eur"]
_TORN_COLS = ["driver", "baseline_uplift_eur", "a_label", "a_uplift_eur",
              "b_label", "b_uplift_eur", "low_uplift_eur", "high_uplift_eur",
              "downside_eur", "upside_eur", "swing_eur"]


def write_scenario_csv(lib: list[ScenarioResult], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_SCEN_COLS)
        for r in lib:
            w.writerow([r.name, r.total_uplift_eur, r.pricing_uplift_eur,
                        r.promo_uplift_eur, r.assortment_uplift_eur,
                        r.delta_vs_baseline_eur, r.n_carried, r.capital_used_eur])
    return p


def write_tornado_csv(rows: list[TornadoRow], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_TORN_COLS)
        for r in rows:
            d = asdict(r)
            w.writerow([d[c] for c in _TORN_COLS])
    return p


# palette shared with report.py / service_level.py / the web dashboard
_BLUE, _PINK, _GREEN, _INK, _GREY = (
    "#2f6bff", "#ea4b71", "#1d9e6f", "#1f2933", "#9aa5b1")


def _axis_bounds(vmin: float, vmax: float) -> tuple[float, float]:
    """Deterministic padded axis: pad ~6% each side, snap to a 5,000 grid."""
    span = max(1.0, vmax - vmin)
    lo = vmin - 0.06 * span
    hi = vmax + 0.06 * span
    step = 5_000.0
    import math
    return (math.floor(lo / step) * step, math.ceil(hi / step) * step)


def render_tornado_svg(rows: list[TornadoRow], baseline: float,
                       path: str | Path) -> Path:
    """Hand-drawn horizontal tornado: drivers ranked by swing, each a bar from
    its low- to high-uplift outcome, split at the baseline into a downside (pink)
    and an upside (green) segment. One x-axis (total uplift €). Deterministic."""
    W = 760
    top, row_h, gap = 92, 26, 12
    n = len(rows)
    plot_h = n * (row_h + gap)
    H = top + plot_h + 60
    ml, mr = 168, 96
    plot_w = W - ml - mr

    lo_v = min([r.low_uplift_eur for r in rows] + [baseline])
    hi_v = max([r.high_uplift_eur for r in rows] + [baseline])
    ax_lo, ax_hi = _axis_bounds(lo_v, hi_v)

    def x(v: float) -> float:
        return ml + (v - ax_lo) / (ax_hi - ax_lo) * plot_w

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{ml - 140}" y="30" font-size="16" font-weight="700" '
             f'fill="{_INK}">Sensitivity of the expected uplift (tornado)</text>')
    s.append(f'<text x="{ml - 140}" y="48" font-size="11" fill="{_GREY}">One-way '
             f'sweep of each driver over a plausible planning band; illustrative, '
             f'not a forecast</text>')
    s.append(f'<text x="{ml - 140}" y="66" font-size="11" fill="{_INK}">Baseline '
             f'total EUR {baseline:,.0f}/yr - bars show downside (pink) vs upside '
             f'(green)</text>')

    # x gridlines + labels
    for i in range(5):
        v = ax_lo + (ax_hi - ax_lo) * i / 4
        gx = x(v)
        s.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
                 f'stroke="#eef1f5" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{top + plot_h + 18:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="{_GREY}">EUR {v / 1000:.0f}k</text>')

    # baseline reference line
    bx = x(baseline)
    s.append(f'<line x1="{bx:.1f}" y1="{top - 6}" x2="{bx:.1f}" y2="{top + plot_h}" '
             f'stroke="{_INK}" stroke-width="1.3" stroke-dasharray="4 3" opacity="0.7"/>')
    s.append(f'<text x="{bx:.1f}" y="{top - 10:.1f}" font-size="10" '
             f'text-anchor="middle" font-weight="700" fill="{_INK}">baseline</text>')

    for i, r in enumerate(rows):
        cy = top + i * (row_h + gap) + gap / 2
        yc = cy + row_h / 2
        lo_x, hi_x, b_x = x(r.low_uplift_eur), x(r.high_uplift_eur), bx
        # downside segment (low .. min(baseline, high))
        d_hi = min(b_x, hi_x)
        if lo_x < d_hi - 0.1:
            s.append(f'<rect x="{lo_x:.1f}" y="{cy:.1f}" width="{d_hi - lo_x:.1f}" '
                     f'height="{row_h}" fill="{_PINK}" opacity="0.85"/>')
        # upside segment (max(baseline, low) .. high)
        u_lo = max(b_x, lo_x)
        if hi_x > u_lo + 0.1:
            s.append(f'<rect x="{u_lo:.1f}" y="{cy:.1f}" width="{hi_x - u_lo:.1f}" '
                     f'height="{row_h}" fill="{_GREEN}" opacity="0.85"/>')
        # driver label (left) + swing (right)
        s.append(f'<text x="{ml - 12}" y="{yc + 3:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{_INK}">{r.driver}</text>')
        s.append(f'<text x="{ml + plot_w + 8}" y="{yc + 3:.1f}" font-size="10" '
                 f'fill="{_INK}">EUR {r.swing_eur:,.0f}</text>')
        # endpoint value labels
        s.append(f'<text x="{lo_x - 4:.1f}" y="{yc + 3:.1f}" font-size="8.5" '
                 f'text-anchor="end" fill="{_GREY}">{r.low_uplift_eur / 1000:.0f}k</text>')
        s.append(f'<text x="{hi_x + 4:.1f}" y="{yc + 3:.1f}" font-size="8.5" '
                 f'text-anchor="start" fill="{_GREY}">{r.high_uplift_eur / 1000:.0f}k</text>')

    s.append(f'<text x="{ml + plot_w / 2:.1f}" y="{H - 8}" font-size="10" '
             f'text-anchor="middle" fill="{_INK}">Total expected uplift (EUR / year)</text>')
    s.append('</svg>')
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s) + "\n", encoding="utf-8")
    return p


def write_scenario_deliverables(lib: list[ScenarioResult],
                                tornado: list[TornadoRow],
                                outdir: str | Path = "deliverables") -> dict:
    """Emit the three deterministic scenario deliverables and return their paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = lib[0].total_uplift_eur if lib else 0.0
    return {
        "scenario_summary.csv": write_scenario_csv(lib, out / "scenario_summary.csv"),
        "sensitivity_tornado.csv": write_tornado_csv(tornado, out / "sensitivity_tornado.csv"),
        "sensitivity_tornado.svg": render_tornado_svg(tornado, baseline,
                                                      out / "sensitivity_tornado.svg"),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _print_tables(lib: list[ScenarioResult], tornado: list[TornadoRow]) -> None:
    line = "=" * 78
    print(line)
    print(" REVOPS — scenario & sensitivity (robustness of the expected uplift)")
    print(line)
    print(f" {'Scenario':<26}{'Total EUR/yr':>14}{'vs base':>12}"
          f"{'carried':>9}{'capital EUR':>14}")
    print("-" * 78)
    for r in lib:
        print(f" {r.name:<26}{r.total_uplift_eur:>14,.0f}"
              f"{r.delta_vs_baseline_eur:>+12,.0f}{r.n_carried:>9}"
              f"{r.capital_used_eur:>14,.0f}")
    print("-" * 78)
    print(" Tornado — drivers ranked by swing in total uplift (widest = most exposed):")
    for r in tornado:
        print(f"   {r.driver:<22} EUR {r.low_uplift_eur:>10,.0f} .. "
              f"{r.high_uplift_eur:>10,.0f}   swing EUR {r.swing_eur:>10,.0f}")
    print(line)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(prog="revops.scenario", description=__doc__)
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full scenario + tornado result to this JSON file")
    ap.add_argument("--outdir", default="deliverables",
                    help="where to write the CSV/SVG deliverables")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the forecaster's training printout")
    args = ap.parse_args(argv)

    ctx = predict_context(verbose=not args.quiet)
    lib = scenario_library(ctx)
    tornado = sensitivity_tornado(ctx, baseline=lib[0].total_uplift_eur)

    paths = write_scenario_deliverables(lib, tornado, outdir=args.outdir)
    _print_tables(lib, tornado)
    for name, p in paths.items():
        print(f"  wrote {name:<28} {p}")

    if args.json:
        import json
        payload = {
            "scenarios": [asdict(r) for r in lib],
            "tornado": [asdict(r) for r in tornado],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull scenario result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
