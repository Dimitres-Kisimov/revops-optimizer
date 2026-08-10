"""robustness.py — which recommended price moves survive the uncertainty?

The Monte-Carlo layer (``revops/simulate.py``) prices the *total* uplift as a
distribution — but a planner executes the plan one price move at a time, and a
band on the total says nothing about *which* of the recommended moves it is safe
to action. This module closes that gap with a **decision gate**: it replays the
*same* fixed-seed Latin-Hypercube draws the published risk band is built from
(same sampler, same seed, same triangular planning bands), re-solves the
identical predict→optimize core for each draw via ``scenario.solve_plan``, and
asks, per baseline price move: *is this exact move still what the optimizer
recommends under that draw?*

Per move (a baseline recommendation with a |change| ≥ 1%), across the draws:

  * ``carry_rate``           — share of draws in which the assortment MILP still
                               carries the SKU at all (a delisted SKU's move
                               earns nothing, whatever the price maths says);
  * ``direction_agreement``  — share of draws in which the pricing optimizer
                               still recommends the *same move* (same direction,
                               still ≥ the 1% move threshold);
  * ``uplift P10/P50/P90``   — the annual € of **executing the published price**
                               under each draw's drawn cost, demand and
                               elasticity. This is deliberately *not* the
                               re-optimized € (which is non-negative whenever the
                               SKU is carried, and so would make a downside test
                               vacuous): the action is frozen, the world moves.
                               Unconditional — counted as €0 in any draw where
                               the SKU is delisted — so the P10 already prices
                               in the delisting risk.

The gate is deliberately simple and stated in full:  **ACCEPT** a move iff
``carry_rate ≥ 90%`` *and* ``direction_agreement ≥ 80%`` *and* the P10 of its
annual uplift is > 0; otherwise **HOLD**, with a single named reason checked in
order — ``delist-risk``, then ``direction-flips``, then ``downside``. The
thresholds are module constants, not magic numbers buried in the code.

Because the draws are byte-for-byte the ones behind the published P10–P90 band
(a test pins the per-draw totals to ``simulate``'s), the accepted/held split is
an exact decomposition of that band into actions — not a second, parallel model.
Deterministic: fixed seed, no wall-clock; the CSV/SVG deliverables are
byte-identical across re-runs. With degenerate (zero-width) planning bands every
baseline move is accepted with agreement 1.0 and P10 = P50 = P90 = its baseline
uplift — the collapse-to-base-case gate.

The verdicts are **modelled on synthetic data under illustrative planning
ranges** — a screening discipline, not a guarantee that an accepted move will
earn its € (or that a held one would not have).

    from revops.robustness import analyze, predict_context
    ctx = predict_context(verbose=False)
    res = analyze(ctx, draws=256)       # per-move accept/hold + the accepted book

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

# reuse the exact predict->optimize core and the exact published draw design
from .optimize.pricing import _demand  # the constant-elasticity Q(P) of record
from .scenario import (
    PredictContext,
    Scenario,
    predict_context,
    solve_plan,
)
from .simulate import (
    DEFAULT_DRAWS,
    DEFAULT_RISK_DRIVERS,
    SEED,
    RiskDriver,
    latin_hypercube,
    triangular_ppf,
)

# a "move" is a baseline recommendation with |price change| >= this many percent
# (the same threshold prescribe() uses for its n_moves count).
MOVE_THRESHOLD_PCT = 1.0
# gate thresholds — stated once, tested, and printed on the deliverable.
ACCEPT_MIN_CARRY = 0.90         # the SKU must stay in the assortment in >=90% of draws
ACCEPT_MIN_AGREEMENT = 0.80     # the same move must be re-recommended in >=80% of draws

VERDICT_ACCEPT = "accept"
VERDICT_HOLD = "hold"
REASON_ROBUST = "robust"
REASON_DELIST = "delist-risk"
REASON_FLIPS = "direction-flips"
REASON_DOWNSIDE = "downside"


def gate(carry_rate: float, direction_agreement: float,
         uplift_p10_eur: float) -> tuple[str, str]:
    """The accept/hold rule, as a pure function so the logic is hand-testable.
    Reasons are checked in a fixed order (delist-risk beats direction-flips
    beats downside), so a move gets exactly one deterministic reason."""
    if carry_rate < ACCEPT_MIN_CARRY:
        return VERDICT_HOLD, REASON_DELIST
    if direction_agreement < ACCEPT_MIN_AGREEMENT:
        return VERDICT_HOLD, REASON_FLIPS
    if uplift_p10_eur <= 0.0:
        return VERDICT_HOLD, REASON_DOWNSIDE
    return VERDICT_ACCEPT, REASON_ROBUST


# --------------------------------------------------------------------------- #
# results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class MoveRobustness:
    """One baseline price move and how it held up across the joint draws."""
    sku: str
    category: str
    direction: str                      # "raise" | "cut"
    baseline_change_pct: float
    baseline_uplift_annual_eur: float
    carry_rate: float                   # share of draws the SKU is carried
    direction_agreement: float          # share of draws the same move recurs
    uplift_p10_eur: float               # frozen-price €; EUR 0 when delisted
    uplift_p50_eur: float
    uplift_p90_eur: float
    verdict: str                        # "accept" | "hold"
    reason: str                         # robust | delist-risk | direction-flips | downside


@dataclass
class RobustnessResult:
    n_draws: int
    seed: int
    baseline_total_uplift_eur: float    # the deterministic headline (pin vs simulate)
    n_moves: int
    n_accept: int
    n_hold: int
    accepted_baseline_uplift_eur: float  # baseline € of the accepted moves
    held_baseline_uplift_eur: float      # baseline € parked behind the gate
    accepted_book_p10_eur: float         # joint per-draw sum over accepted moves
    accepted_book_p50_eur: float
    accepted_book_p90_eur: float
    moves: list[MoveRobustness] = field(default_factory=list)
    draws_total_eur: list[float] = field(default_factory=list)   # == simulate's draws


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q, method="linear"))


def _frozen_move_uplift(p0: float, cost: float, q0: float, e: float,
                        price_frozen: float) -> float:
    """Monthly € of charging ``price_frozen`` instead of ``p0`` under the given
    (cost, demand, elasticity) conditions — the *executed* move's margin delta,
    using the same constant-elasticity demand curve as the pricing optimizer."""
    m_now = (p0 - cost) * q0
    m_frozen = (price_frozen - cost) * _demand(q0, p0, price_frozen, e)
    return m_frozen - m_now


# --------------------------------------------------------------------------- #
# the analysis                                                                 #
# --------------------------------------------------------------------------- #
def analyze(ctx: PredictContext,
            draws: int = DEFAULT_DRAWS,
            drivers: tuple[RiskDriver, ...] = DEFAULT_RISK_DRIVERS,
            seed: int = SEED) -> RobustnessResult:
    """Replay the risk band's joint draws with per-SKU detail and gate every
    baseline price move. Deterministic given ``seed``/``draws``; with the
    default arguments the draws are byte-for-byte the ones ``simulate`` reports
    (a test pins the two layers together)."""
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
    base_sol = solve_plan(ctx, base)
    category_of = {s.sku: s.category for s in ctx.skus}

    base_moves = [p for p in base_sol.prices
                  if abs(p.price_change_pct) >= MOVE_THRESHOLD_PCT]
    # deterministic identity order: biggest baseline € first, then SKU id
    base_moves.sort(key=lambda p: (-p.margin_uplift_eur, p.sku))
    m = len(base_moves)
    idx_of = {p.sku: k for k, p in enumerate(base_moves)}
    base_sign = np.array([1.0 if p.price_change_pct > 0 else -1.0
                          for p in base_moves])
    # the action being gated: the *published* (rounded-to-cents) price
    frozen_price = np.array([p.recommended_price for p in base_moves])

    # the frozen move's € at baseline conditions — computed with the identical
    # arithmetic used per draw, so degenerate bands collapse to it exactly.
    base_inputs = {s.sku: s for s in base_sol.price_inputs}
    base_frozen = np.zeros(m)
    for k, p in enumerate(base_moves):
        s0 = base_inputs[p.sku]
        base_frozen[k] = _frozen_move_uplift(
            s0.current_price_eur, s0.unit_cost_eur, s0.avg_monthly_demand,
            s0.elasticity, float(frozen_price[k])) * 12

    carried_ct = np.zeros(m)
    agree_ct = np.zeros(m)
    uplift = np.zeros((n, m))           # unconditional: stays 0 when delisted
    tot = np.empty(n)
    for i in range(n):
        kw = {fields[j]: float(cols[j][i]) for j in range(d)}
        sol = solve_plan(ctx, replace(base, name=f"draw{i}", **kw))
        tot[i] = sol.total_uplift_eur
        inputs = {s.sku: s for s in sol.price_inputs}
        for line in sol.prices:
            k = idx_of.get(line.sku)
            if k is None:
                continue
            carried_ct[k] += 1
            s_i = inputs[line.sku]
            uplift[i, k] = _frozen_move_uplift(
                s_i.current_price_eur, s_i.unit_cost_eur, s_i.avg_monthly_demand,
                s_i.elasticity, float(frozen_price[k])) * 12
            if (line.price_change_pct * base_sign[k] > 0
                    and abs(line.price_change_pct) >= MOVE_THRESHOLD_PCT):
                agree_ct[k] += 1

    moves: list[MoveRobustness] = []
    accepted = np.zeros(m, dtype=bool)
    for k, p in enumerate(base_moves):
        carry = carried_ct[k] / n
        agree = agree_ct[k] / n
        p10, p50, p90 = (_pct(uplift[:, k], q) for q in (10, 50, 90))
        verdict, reason = gate(carry, agree, p10)
        accepted[k] = verdict == VERDICT_ACCEPT
        moves.append(MoveRobustness(
            sku=p.sku, category=category_of.get(p.sku, ""),
            direction="raise" if p.price_change_pct > 0 else "cut",
            baseline_change_pct=round(p.price_change_pct, 2),
            baseline_uplift_annual_eur=round(float(base_frozen[k]), 2),
            carry_rate=round(carry, 4),
            direction_agreement=round(agree, 4),
            uplift_p10_eur=round(p10, 2), uplift_p50_eur=round(p50, 2),
            uplift_p90_eur=round(p90, 2),
            verdict=verdict, reason=reason))

    # the accepted book: a *joint* per-draw sum, so its P10 is a real portfolio
    # downside (percentiles are not additive across moves).
    book = uplift[:, accepted].sum(axis=1) if accepted.any() else np.zeros(n)
    acc_base = sum(mv.baseline_uplift_annual_eur for mv in moves
                   if mv.verdict == VERDICT_ACCEPT)
    held_base = sum(mv.baseline_uplift_annual_eur for mv in moves
                    if mv.verdict == VERDICT_HOLD)

    return RobustnessResult(
        n_draws=n, seed=seed,
        baseline_total_uplift_eur=round(base_sol.total_uplift_eur, 2),
        n_moves=m,
        n_accept=int(accepted.sum()), n_hold=m - int(accepted.sum()),
        accepted_baseline_uplift_eur=round(acc_base, 2),
        held_baseline_uplift_eur=round(held_base, 2),
        accepted_book_p10_eur=round(_pct(book, 10), 2),
        accepted_book_p50_eur=round(_pct(book, 50), 2),
        accepted_book_p90_eur=round(_pct(book, 90), 2),
        moves=moves,
        draws_total_eur=[round(float(x), 2) for x in tot],
    )


# --------------------------------------------------------------------------- #
# deterministic deliverables — per-move CSV (fact-shaped) + hand-drawn SVG     #
# --------------------------------------------------------------------------- #
_MOVE_COLS = ["sku", "category", "direction", "baseline_change_pct",
              "baseline_uplift_annual_eur", "carry_rate", "direction_agreement",
              "uplift_p10_eur", "uplift_p50_eur", "uplift_p90_eur",
              "verdict", "reason"]


def write_moves_csv(res: RobustnessResult, path: str | Path) -> Path:
    """Byte-stable per-move CSV (grain = SKU; joins ``dim_sku`` in Power BI)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_MOVE_COLS)
        for mv in res.moves:
            d = asdict(mv)
            w.writerow([d[c] for c in _MOVE_COLS])
    return p


# palette shared with report.py / scenario.py / simulate.py / the web app
_BLUE, _PINK, _GREEN, _INK, _GREY = (
    "#2f6bff", "#ea4b71", "#1d9e6f", "#1f2933", "#9aa5b1")


def _axis_bounds(vmin: float, vmax: float, step: float = 500.0) -> tuple[float, float]:
    """Deterministic padded axis: pad ~6% each side, snap out to a grid step."""
    import math
    span = max(1.0, vmax - vmin)
    lo = vmin - 0.06 * span
    hi = vmax + 0.06 * span
    return (math.floor(lo / step) * step, math.ceil(hi / step) * step)


def render_moves_svg(res: RobustnessResult, path: str | Path) -> Path:
    """Hand-drawn per-move dot-and-whisker chart: each baseline price move is a
    row, its whisker spans the P10..P90 of executing the published price across
    the joint draws (EUR 0 when delisted), the dot is the median and the tick
    the baseline €. Green = accepted, pink = held (reason printed).
    Deterministic."""
    rows = res.moves
    W = 760
    top, row_h, gap = 96, 15, 7
    n = len(rows)
    plot_h = max(1, n) * (row_h + gap)
    H = top + plot_h + 64
    ml, mr = 150, 118
    plot_w = W - ml - mr

    vals = [0.0]
    for r in rows:
        vals += [r.uplift_p10_eur, r.uplift_p90_eur, r.baseline_uplift_annual_eur]
    ax_lo, ax_hi = _axis_bounds(min(vals), max(vals))

    def x(v: float) -> float:
        return ml + (v - ax_lo) / (ax_hi - ax_lo) * plot_w

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{ml - 130}" y="30" font-size="16" font-weight="700" '
             f'fill="{_INK}">Price-move robustness gate ({res.n_draws} joint draws)</text>')
    s.append(f'<text x="{ml - 130}" y="48" font-size="11" fill="{_GREY}">Whisker '
             f'P10..P90 of executing the published price across the Monte-Carlo '
             f'draws (EUR 0 when delisted) &#183; dot = median &#183; tick = baseline</text>')
    s.append(f'<text x="{ml - 130}" y="66" font-size="11" fill="{_INK}">'
             f'<tspan fill="{_GREEN}" font-weight="700">accept {res.n_accept}</tspan>'
             f' (carry &#8805; {ACCEPT_MIN_CARRY * 100:.0f}%, same move &#8805; '
             f'{ACCEPT_MIN_AGREEMENT * 100:.0f}%, P10 &gt; 0) &#183; '
             f'<tspan fill="{_PINK}" font-weight="700">hold {res.n_hold}</tspan>'
             f' &#183; accepted book P10 EUR {res.accepted_book_p10_eur:,.0f}/yr</text>')

    # x gridlines + labels
    for i in range(5):
        v = ax_lo + (ax_hi - ax_lo) * i / 4
        gx = x(v)
        s.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
                 f'stroke="#eef1f5" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{top + plot_h + 18:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="{_GREY}">EUR {v / 1000:.1f}k</text>')

    # zero reference line
    zx = x(0.0)
    s.append(f'<line x1="{zx:.1f}" y1="{top - 4}" x2="{zx:.1f}" y2="{top + plot_h}" '
             f'stroke="{_INK}" stroke-width="1.1" stroke-dasharray="4 3" opacity="0.6"/>')

    for i, r in enumerate(rows):
        yc = top + i * (row_h + gap) + gap / 2 + row_h / 2
        color = _GREEN if r.verdict == VERDICT_ACCEPT else _PINK
        lo_x, hi_x = x(r.uplift_p10_eur), x(r.uplift_p90_eur)
        # whisker P10..P90
        s.append(f'<line x1="{lo_x:.1f}" y1="{yc:.1f}" x2="{hi_x:.1f}" y2="{yc:.1f}" '
                 f'stroke="{color}" stroke-width="4" opacity="0.45"/>')
        # baseline tick
        bx = x(r.baseline_uplift_annual_eur)
        s.append(f'<line x1="{bx:.1f}" y1="{yc - 5:.1f}" x2="{bx:.1f}" y2="{yc + 5:.1f}" '
                 f'stroke="{_INK}" stroke-width="1.4"/>')
        # median dot
        s.append(f'<circle cx="{x(r.uplift_p50_eur):.1f}" cy="{yc:.1f}" r="3.2" '
                 f'fill="{color}"/>')
        # left label: SKU + the move itself
        s.append(f'<text x="{ml - 8}" y="{yc + 3:.1f}" font-size="10" '
                 f'text-anchor="end" fill="{_INK}">{r.sku} {r.baseline_change_pct:+.1f}%</text>')
        # right label: verdict (accept -> agreement, hold -> reason)
        note = (f"{r.direction_agreement * 100:.0f}% same move"
                if r.verdict == VERDICT_ACCEPT else r.reason)
        s.append(f'<text x="{ml + plot_w + 8}" y="{yc + 3:.1f}" font-size="9.5" '
                 f'fill="{color}" font-weight="600">{note}</text>')

    s.append(f'<text x="{ml + plot_w / 2:.1f}" y="{H - 28}" font-size="10" '
             f'text-anchor="middle" fill="{_INK}">Annual EUR of executing the '
             f'published price (EUR 0 when the SKU is delisted)</text>')
    s.append(f'<text x="{ml - 130}" y="{H - 10}" font-size="10" fill="{_GREY}">'
             f'Modelled on synthetic data under illustrative planning ranges - '
             f'a screening discipline, not a guarantee.</text>')
    s.append('</svg>')

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s) + "\n", encoding="utf-8")
    return p


def write_robustness_deliverables(res: RobustnessResult,
                                  outdir: str | Path = "deliverables") -> dict:
    """Emit the two deterministic robustness deliverables and return their paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "price_move_robustness.csv": write_moves_csv(
            res, out / "price_move_robustness.csv"),
        "price_move_robustness.svg": render_moves_svg(
            res, out / "price_move_robustness.svg"),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _print_summary(res: RobustnessResult) -> None:
    line = "=" * 78
    print(line)
    print(" REVOPS — price-move robustness gate (accept/hold across the MC draws)")
    print(line)
    print(f" Draws {res.n_draws}   seed {res.seed}   baseline total "
          f"EUR {res.baseline_total_uplift_eur:,.0f}/yr")
    print(f" Gate: carry >= {ACCEPT_MIN_CARRY * 100:.0f}%  &  same move >= "
          f"{ACCEPT_MIN_AGREEMENT * 100:.0f}%  &  P10 > 0")
    print("-" * 78)
    print(f" {'SKU':<10}{'move':>8}{'base EUR/yr':>13}{'carry':>8}{'agree':>8}"
          f"{'P10':>10}{'P90':>10}  verdict")
    for mv in res.moves:
        tag = mv.verdict if mv.verdict == VERDICT_ACCEPT else f"hold ({mv.reason})"
        print(f" {mv.sku:<10}{mv.baseline_change_pct:>+7.1f}%"
              f"{mv.baseline_uplift_annual_eur:>13,.0f}"
              f"{mv.carry_rate * 100:>7.0f}%{mv.direction_agreement * 100:>7.0f}%"
              f"{mv.uplift_p10_eur:>10,.0f}{mv.uplift_p90_eur:>10,.0f}  {tag}")
    print("-" * 78)
    print(f" Moves {res.n_moves}: accept {res.n_accept} "
          f"(baseline EUR {res.accepted_baseline_uplift_eur:,.0f}/yr), "
          f"hold {res.n_hold} (EUR {res.held_baseline_uplift_eur:,.0f}/yr parked)")
    print(f" Accepted book, joint   P10 EUR {res.accepted_book_p10_eur:,.0f} / "
          f"P50 {res.accepted_book_p50_eur:,.0f} / "
          f"P90 {res.accepted_book_p90_eur:,.0f} per year")
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

    ap = argparse.ArgumentParser(prog="revops.robustness", description=__doc__)
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help=f"number of joint Monte-Carlo draws (default {DEFAULT_DRAWS}, "
                         f"matching revops.simulate so the two layers share draws)")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full robustness result to this JSON file")
    ap.add_argument("--outdir", default="deliverables",
                    help="where to write the CSV/SVG deliverables")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the forecaster's training printout")
    args = ap.parse_args(argv)

    ctx = predict_context(verbose=not args.quiet)

    # The HiGHS build leaks solver chatter to the OS-level stdout; silence it at
    # the file-descriptor level only around the draw loop, so the gate table
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

    print(f"Gating the baseline price moves across {args.draws} joint draws "
          f"(re-solving the MILP each draw)...", file=sys.stderr, flush=True)
    with _quiet_fd():
        res = analyze(ctx, draws=args.draws)

    paths = write_robustness_deliverables(res, outdir=args.outdir)
    _print_summary(res)
    for name, p in paths.items():
        print(f"  wrote {name:<28} {p}")

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(res), f, indent=2)
        print(f"\nFull robustness result written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
