"""simulate.py — a Monte-Carlo risk band on the headline uplift (interactions).

The scenario/tornado layer (``revops/scenario.py``) sweeps each assumption **one
at a time**, so — as its own docs state — it reads *local* sensitivity, **not
interactions**. A planner's next question is the joint one: *taking all the
prediction uncertainties together, what is the range of the € the plan can
deliver, and how bad is the downside?* This module answers that without touching
a single optimizer.

  * **Joint Monte-Carlo** — the four predictive uncertainties (demand, unit cost,
    estimated elasticity, decline risk) are drawn **together** from three-point
    (triangular) planning distributions and pushed back through the *existing*
    predict→optimize core (``scenario.evaluate`` re-solves the assortment MILP,
    the elasticity pricing and the concave promo exactly as ``prescribe()`` does).
    Because the draws are joint, the result captures interactions the one-way
    tornado cannot.
  * **A distribution, not a point** — the headline € becomes a full distribution:
    a P10–P90 confidence band, the median, and the classic downside pair a
    finance audience expects — **Value-at-Risk** (the P10 downside) and
    **Conditional VaR / expected shortfall** (the mean of the worst 10% of
    outcomes) — plus the probability the re-optimized outcome clears a hurdle.

The decision *levers* (capital budget, shelf capacity, promo budget, price
guardrail) are held at baseline: they are choices, not uncertainties. Each draw
is **re-optimized**, so the band is the range of *achievable* uplift across
plausible conditions, each solved to optimality — not the drift of a frozen plan.

Deterministic by construction: a **fixed seed** drives a Latin-Hypercube sample,
the draw count is fixed, and there is no wall-clock anywhere — the CSV/SVG
deliverables are byte-identical across re-runs. With ``demand_mult = 1``, the
central (mode) draw reproduces the headline plan to the cent.

The planning distributions are **illustrative ranges, not a forecast**; every
figure is what the model computes under a drawn assumption, not a guarantee, and
the whole thing runs on synthetic-but-structured data.

    from revops.simulate import predict_context, simulate
    ctx = predict_context(verbose=False)
    res = simulate(ctx, draws=256)      # P10/P50/P90 band + VaR/CVaR + P(clear)

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

# reuse the exact predict->optimize core the rest of the project already trusts
from .scenario import (
    PredictContext,
    Scenario,
    evaluate,
    predict_context,
)

# Fixed seed -> byte-identical outputs. No wall-clock, no unseeded RNG anywhere.
SEED = 20_260_807
DEFAULT_DRAWS = 256


# --------------------------------------------------------------------------- #
# the uncertain inputs — three-point (triangular) planning distributions       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskDriver:
    """One uncertain predictive input, as a three-point estimate (min, mode,
    max) on its ``Scenario`` multiplier. ``mode = 1.0`` means "the model's own
    estimate is the most likely value"; the ends are a plausible planning band.
    The bands mirror ``scenario.DEFAULT_DRIVERS`` so the two layers stay in step
    (a test pins them together)."""
    name: str
    field: str          # the Scenario multiplier this driver perturbs
    lo: float
    mode: float
    hi: float


# demand ±15%, cost ±8%, |elasticity| ±20%, decline risk ±30% — mode at 1.0.
DEFAULT_RISK_DRIVERS: tuple[RiskDriver, ...] = (
    RiskDriver("Demand level", "demand_mult", 0.85, 1.0, 1.15),
    RiskDriver("Unit cost", "cost_mult", 0.92, 1.0, 1.08),
    RiskDriver("Elasticity estimate", "elasticity_mult", 0.80, 1.0, 1.20),
    RiskDriver("Decline risk", "risk_mult", 0.70, 1.0, 1.30),
)


# --------------------------------------------------------------------------- #
# deterministic sampling — Latin Hypercube + triangular inverse-CDF            #
# --------------------------------------------------------------------------- #
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """An ``n x d`` Latin-Hypercube sample in the open unit cube: every column is
    stratified into ``n`` equal bins (one draw each) and the columns are shuffled
    independently, so marginals are smooth at a fraction of the draws plain MC
    would need. Deterministic given ``rng``."""
    grid = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(grid[:, j])       # in-place per-column permutation
    return grid


def triangular_ppf(u: np.ndarray, lo: float, mode: float, hi: float) -> np.ndarray:
    """Inverse CDF of a triangular(lo, mode, hi) distribution, evaluated at the
    uniforms ``u``. Vectorized and closed-form; collapses to the constant ``lo``
    when the band is degenerate (``hi == lo``), which is what makes the
    zero-uncertainty case land exactly on the base plan."""
    u = np.asarray(u, dtype=np.float64)
    if hi <= lo:
        return np.full_like(u, lo)
    fc = (mode - lo) / (hi - lo)      # CDF value at the mode
    left = lo + np.sqrt(np.clip(u * (hi - lo) * (mode - lo), 0.0, None))
    right = hi - np.sqrt(np.clip((1.0 - u) * (hi - lo) * (hi - mode), 0.0, None))
    return np.where(u <= fc, left, right)


# --------------------------------------------------------------------------- #
# the result of a run                                                          #
# --------------------------------------------------------------------------- #
_STATS = ("mean", "std", "min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max")


@dataclass
class Distribution:
    """Summary of one € series across the draws (all rounded to cents)."""
    label: str
    mean: float
    std: float
    min: float
    p05: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    max: float


def _distribution(label: str, arr: np.ndarray) -> Distribution:
    a = np.asarray(arr, dtype=np.float64)
    p05, p10, p25, p50, p75, p90, p95 = (
        float(np.percentile(a, q, method="linear")) for q in (5, 10, 25, 50, 75, 90, 95))
    return Distribution(
        label=label,
        mean=round(float(a.mean()), 2),
        std=round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 2),
        min=round(float(a.min()), 2),
        p05=round(p05, 2), p10=round(p10, 2), p25=round(p25, 2), p50=round(p50, 2),
        p75=round(p75, 2), p90=round(p90, 2), p95=round(p95, 2),
        max=round(float(a.max()), 2),
    )


@dataclass
class SimResult:
    n_draws: int
    seed: int
    hurdle_eur: float
    baseline_point_estimate_eur: float      # the deterministic headline (mode draw)
    total: Distribution
    pricing: Distribution
    promo: Distribution
    assortment: Distribution
    var10_eur: float                        # P10 of total — the 90%-confident floor
    cvar10_eur: float                       # mean of the worst 10% (expected shortfall)
    prob_at_or_above_hurdle: float          # P(total >= hurdle)
    downside_point_vs_p10_eur: float        # point - P10  (>= 0): headline-at-risk
    upside_p90_vs_point_eur: float          # P90 - point  (>= 0)
    draws_total_eur: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# the simulation                                                              #
# --------------------------------------------------------------------------- #
def simulate(ctx: PredictContext,
             draws: int = DEFAULT_DRAWS,
             drivers: tuple[RiskDriver, ...] = DEFAULT_RISK_DRIVERS,
             hurdle: float | None = None,
             seed: int = SEED) -> SimResult:
    """Draw ``draws`` joint scenarios from the triangular planning distributions,
    re-solve the predict→optimize core for each, and summarize the resulting
    distribution of total uplift (and its three components).

    ``hurdle`` defaults to the deterministic headline point estimate, so
    ``prob_at_or_above_hurdle`` reads "chance the re-optimized outcome meets or
    beats the headline number". Deterministic given ``seed`` and ``draws``."""
    n = int(draws)
    if n < 1:
        raise ValueError("draws must be >= 1")
    rng = np.random.default_rng(seed)
    d = len(drivers)
    lh = latin_hypercube(n, d, rng)
    cols = [triangular_ppf(lh[:, j], dr.lo, dr.mode, dr.hi)
            for j, dr in enumerate(drivers)]
    fields = [dr.field for dr in drivers]

    base = Scenario("Baseline")
    point = evaluate(ctx, base).total_uplift_eur

    tot = np.empty(n)
    pri = np.empty(n)
    pro = np.empty(n)
    ass = np.empty(n)
    for i in range(n):
        kw = {fields[j]: float(cols[j][i]) for j in range(d)}
        r = evaluate(ctx, replace(base, name=f"draw{i}", **kw))
        tot[i] = r.total_uplift_eur
        pri[i] = r.pricing_uplift_eur
        pro[i] = r.promo_uplift_eur
        ass[i] = r.assortment_uplift_eur

    hv = point if hurdle is None else float(hurdle)
    var10 = float(np.percentile(tot, 10, method="linear"))
    tail = tot[tot <= var10]
    cvar10 = float(tail.mean()) if tail.size else float(tot.min())
    p90 = float(np.percentile(tot, 90, method="linear"))

    return SimResult(
        n_draws=n, seed=seed, hurdle_eur=round(hv, 2),
        baseline_point_estimate_eur=round(point, 2),
        total=_distribution("total", tot),
        pricing=_distribution("pricing", pri),
        promo=_distribution("promo", pro),
        assortment=_distribution("assortment", ass),
        var10_eur=round(var10, 2),
        cvar10_eur=round(cvar10, 2),
        prob_at_or_above_hurdle=round(float(np.mean(tot >= hv)), 4),
        downside_point_vs_p10_eur=round(point - var10, 2),
        upside_p90_vs_point_eur=round(p90 - point, 2),
        draws_total_eur=[round(float(x), 2) for x in tot],
    )


# --------------------------------------------------------------------------- #
# deterministic deliverables — tall CSV (DAX-ready) + hand-drawn SVG           #
# --------------------------------------------------------------------------- #
def summary_rows(res: SimResult) -> list[tuple[str, float]]:
    """The (metric, value) rows written to the CSV — a tall table matching the
    ``kpi_headline`` shape so Power BI can read a scalar with one filter."""
    rows: list[tuple[str, float]] = [
        ("baseline_point_estimate_eur", res.baseline_point_estimate_eur),
        ("hurdle_eur", res.hurdle_eur),
        ("total_prob_at_or_above_hurdle", res.prob_at_or_above_hurdle),
        ("total_var10_eur", res.var10_eur),
        ("total_cvar10_eur", res.cvar10_eur),
        ("total_downside_point_vs_p10_eur", res.downside_point_vs_p10_eur),
        ("total_upside_p90_vs_point_eur", res.upside_p90_vs_point_eur),
    ]
    for comp in (res.total, res.pricing, res.promo, res.assortment):
        for stat in _STATS:
            rows.append((f"{comp.label}_{stat}_eur", getattr(comp, stat)))
    rows.append(("n_draws", res.n_draws))
    rows.append(("seed", res.seed))
    return rows


def write_summary_csv(res: SimResult, path: str | Path) -> Path:
    """Byte-stable tall CSV (metric, value). No wall-clock, no RNG at write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for metric, value in summary_rows(res):
            w.writerow([metric, value])
    return p


# "decision desk" tokens — dataviz-validated palette shared with the other SVG
# deliverables and the web dashboard. One series (the total-uplift draws) wears
# one hue: a blue sequential ramp, stepped darker inside the P10-P90 band so
# the band itself is the designed object, not a footnote.
_SURFACE, _INK, _SEC, _MUTED, _GRID = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9")
_BAND_IN, _BAND_OUT, _HEADLINE = "#5598e7", "#b7d3f6", "#104281"
_FONT = "system-ui, Segoe UI, sans-serif"


def _snapped_bounds(vmin: float, vmax: float, step: float = 5_000.0) -> tuple[float, float]:
    """Deterministic padded axis: pad ~4% each side, snap out to a grid step."""
    import math
    span = max(1.0, vmax - vmin)
    lo = vmin - 0.04 * span
    hi = vmax + 0.04 * span
    return (math.floor(lo / step) * step, math.ceil(hi / step) * step)


def _histogram(draws: list[float], lo: float, hi: float, bins: int) -> list[int]:
    """Fixed-edge counts (deterministic; no numpy histogram edge ambiguity)."""
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in draws:
        k = int((v - lo) / width)
        k = 0 if k < 0 else (bins - 1 if k >= bins else k)
        counts[k] += 1
    return counts


def _bar_top_path(x0: float, y0: float, w: float, h: float, r: float = 3.0) -> str:
    """Column path: rounded at the data end (top), square at the baseline."""
    r = min(r, w / 2, h)
    if r < 0.75:
        return f'M{x0:.1f} {y0 + h:.1f}V{y0:.1f}H{x0 + w:.1f}V{y0 + h:.1f}Z'
    return (f'M{x0:.1f} {y0 + h:.1f}V{y0 + r:.1f}'
            f'A{r:.1f} {r:.1f} 0 0 1 {x0 + r:.1f} {y0:.1f}H{x0 + w - r:.1f}'
            f'A{r:.1f} {r:.1f} 0 0 1 {x0 + w:.1f} {y0 + r:.1f}V{y0 + h:.1f}Z')


def render_distribution_svg(res: SimResult, path: str | Path, bins: int = 24) -> Path:
    """Hand-drawn histogram of the total-uplift draws. The P10–P90 band is the
    designed object: in-band bars wear the darker step of the one-hue ramp,
    out-of-band bars the lighter step; the median, the band edges and the
    deterministic headline (point estimate) are marked, with a downside caption
    (VaR/CVaR). One x-axis (total uplift €). Deterministic."""
    W, H = 760, 440
    ml, mr = 56, 24
    top, plot_h = 104, 246
    plot_w = W - ml - mr

    draws = res.draws_total_eur
    ax_lo, ax_hi = _snapped_bounds(min(draws), max(draws))
    counts = _histogram(draws, ax_lo, ax_hi, bins)
    cmax = max(counts) or 1

    def x(v: float) -> float:
        return ml + (v - ax_lo) / (ax_hi - ax_lo) * plot_w

    def y(c: float) -> float:
        return top + plot_h - (c / cmax) * plot_h

    point = res.baseline_point_estimate_eur
    p10, p50, p90 = res.total.p10, res.total.p50, res.total.p90

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="{_FONT}">')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" '
             f'fill="{_SURFACE}" stroke="{_GRID}"/>')
    s.append(f'<text x="{ml}" y="34" font-size="16" font-weight="700" '
             f'fill="{_INK}">Uplift risk band - joint Monte-Carlo ({res.n_draws} draws)</text>')
    s.append(f'<text x="{ml}" y="52" font-size="11" fill="{_SEC}">Demand, cost, '
             f'elasticity &amp; decline risk drawn together; illustrative, not a forecast</text>')
    s.append(f'<text x="{ml}" y="70" font-size="11" fill="{_INK}">P10-P90 band '
             f'EUR {p10:,.0f} .. {p90:,.0f}/yr &#183; median EUR {p50:,.0f} &#183; '
             f'mean EUR {res.total.mean:,.0f}</text>')

    # y gridlines + labels (solid hairlines, recessive)
    for i in range(5):
        c = cmax * i / 4
        gy = y(c)
        s.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + plot_w}" y2="{gy:.1f}" '
                 f'stroke="{_GRID}" stroke-width="1"/>')
        s.append(f'<text x="{ml - 6}" y="{gy + 3:.1f}" font-size="9" '
                 f'text-anchor="end" fill="{_MUTED}">{c:.0f}</text>')
    s.append(f'<text x="{ml}" y="{top - 8}" font-size="10" fill="{_INK}" '
             f'font-weight="600">Draws</text>')

    # histogram bars — one hue, darker step inside the P10-P90 band, 2px gaps
    bar_w = plot_w / bins
    for k, c in enumerate(counts):
        if c <= 0:
            continue
        bx = ml + k * bar_w
        by = y(c)
        mid = ax_lo + (k + 0.5) * (ax_hi - ax_lo) / bins
        fill = _BAND_IN if p10 <= mid <= p90 else _BAND_OUT
        s.append(f'<path d="{_bar_top_path(bx + 1.0, by, bar_w - 2.0, top + plot_h - by)}" '
                 f'fill="{fill}"/>')

    # markers: P10 / P90 band edges, median, and the deterministic headline.
    # Labels stagger onto two fixed rows so they never collide.
    def vline(v: float, color: str, label: str, dash: str, weight: float,
              row: int) -> None:
        vx = x(v)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        ly = top - 4 if row == 0 else top - 18
        s.append(f'<line x1="{vx:.1f}" y1="{top}" x2="{vx:.1f}" y2="{top + plot_h}" '
                 f'stroke="{color}" stroke-width="{weight}"{da}/>')
        s.append(f'<text x="{vx:.1f}" y="{ly:.1f}" font-size="9.5" '
                 f'text-anchor="middle" font-weight="700" fill="{_INK}">{label}</text>')

    vline(p10, _SEC, f"P10 {p10 / 1000:.0f}k", "4 3", 1.2, 0)
    vline(p90, _SEC, f"P90 {p90 / 1000:.0f}k", "4 3", 1.2, 0)
    vline(p50, _INK, f"median {p50 / 1000:.0f}k", "", 1.6, 0)
    vline(point, _HEADLINE, f"headline {point / 1000:.0f}k", "5 3", 1.8, 1)

    # x-axis ticks
    for i in range(6):
        v = ax_lo + (ax_hi - ax_lo) * i / 5
        tx = x(v)
        s.append(f'<text x="{tx:.1f}" y="{top + plot_h + 18:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="{_MUTED}">EUR {v / 1000:.0f}k</text>')
    s.append(f'<text x="{ml + plot_w / 2:.1f}" y="{top + plot_h + 36:.1f}" font-size="10" '
             f'text-anchor="middle" fill="{_INK}">Total expected uplift (EUR / year)</text>')

    # downside caption
    s.append(f'<text x="{ml}" y="{H - 14}" font-size="10.5" fill="{_INK}">Downside: '
             f'VaR(10%) EUR {res.var10_eur:,.0f} &#183; CVaR(10%) EUR {res.cvar10_eur:,.0f} '
             f'&#183; P(&#8805; headline) {res.prob_at_or_above_hurdle * 100:.0f}%</text>')
    s.append('</svg>')

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s) + "\n", encoding="utf-8")
    return p


def write_simulation_deliverables(res: SimResult,
                                  outdir: str | Path = "deliverables") -> dict:
    """Emit the two deterministic simulation deliverables and return their paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "uplift_simulation.csv": write_summary_csv(res, out / "uplift_simulation.csv"),
        "uplift_distribution.svg": render_distribution_svg(
            res, out / "uplift_distribution.svg"),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _print_summary(res: SimResult) -> None:
    line = "=" * 78
    print(line)
    print(" REVOPS — uplift risk band (joint Monte-Carlo, interactions)")
    print(line)
    print(f" Draws {res.n_draws}   seed {res.seed}   "
          f"headline point estimate EUR {res.baseline_point_estimate_eur:,.0f}/yr")
    print("-" * 78)
    print(f" {'Component':<14}{'P10':>12}{'median':>12}{'P90':>12}"
          f"{'mean':>12}{'std':>10}")
    for c in (res.total, res.pricing, res.promo, res.assortment):
        print(f" {c.label:<14}{c.p10:>12,.0f}{c.p50:>12,.0f}{c.p90:>12,.0f}"
              f"{c.mean:>12,.0f}{c.std:>10,.0f}")
    print("-" * 78)
    print(f" Total P10-P90 band     EUR {res.total.p10:>10,.0f} .. "
          f"{res.total.p90:>10,.0f}/yr")
    print(f" Downside VaR(10%)      EUR {res.var10_eur:>10,.0f}   "
          f"(headline-at-risk EUR {res.downside_point_vs_p10_eur:,.0f})")
    print(f" Expected shortfall     EUR {res.cvar10_eur:>10,.0f}   "
          f"(mean of the worst 10% of draws)")
    print(f" P(outcome >= hurdle)   {res.prob_at_or_above_hurdle * 100:>9.0f}%   "
          f"(hurdle EUR {res.hurdle_eur:,.0f})")
    print(line)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import contextlib
    import os
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(prog="revops.simulate", description=__doc__)
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help=f"number of joint Monte-Carlo draws (default {DEFAULT_DRAWS})")
    ap.add_argument("--hurdle", type=float, default=None, metavar="EUR",
                    help="report P(total uplift >= EUR); default is the headline point estimate")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full simulation result to this JSON file")
    ap.add_argument("--outdir", default="deliverables",
                    help="where to write the CSV/SVG deliverables")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the forecaster's training printout")
    args = ap.parse_args(argv)

    ctx = predict_context(verbose=not args.quiet)

    # The HiGHS build leaks solver chatter to the OS-level stdout; silence it at
    # the file-descriptor level only around the draw loop, so the summary table
    # below stays clean. Progress goes to stderr, which is left untouched.
    @contextlib.contextmanager
    def _quiet_fd():
        saved = None
        try:
            sys.stdout.flush()
            saved = os.dup(1)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
        except Exception:
            saved = None
        try:
            yield
        finally:
            if saved is not None:
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                os.dup2(saved, 1)
                os.close(saved)

    print(f"Running {args.draws} joint Monte-Carlo draws "
          f"(re-solving the MILP each draw)...", file=sys.stderr, flush=True)
    with _quiet_fd():
        res = simulate(ctx, draws=args.draws, hurdle=args.hurdle)

    paths = write_simulation_deliverables(res, outdir=args.outdir)
    _print_summary(res)
    for name, p in paths.items():
        print(f"  wrote {name:<28} {p}")

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(res), f, indent=2)
        print(f"\nFull simulation result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
