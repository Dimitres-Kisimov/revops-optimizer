"""compare.py — plan A vs plan B: what differs, and where every euro went.

The scenario layer evaluates named what-ifs and the Monte-Carlo layer prices the
headline € as a band — but both answer *how big is the number*. A planner
holding two plans asks a different question: **what actually changes between
them, and why is B's total what it is?** This module answers it as a
deterministic A/B diff over the *existing* solve path (``scenario.solve_plan``
— the same MILP + elasticity pricing + concave promo the headline plan comes
from; nothing here re-derives an optimizer or a perturbation):

  * **Assortment churn** — which SKUs enter B's range and which leave A's.
  * **Price moves** — which recommended prices change, which stay put but change
    value because the conditions moved, and — when the gate is run — whether the
    accept/hold verdict from ``revops/robustness.py`` **flips** between the two
    plans (each plan gated on its own moves, over the same fixed-seed draws).
  * **Promo reallocation** — the per-category budget shift.
  * **A € bridge** from A's total to B's, in which **every euro is attributed to
    a named driver** and the identity closes *exactly* (it is summed in integer
    cents, so it reconciles by construction, not to a tolerance):

        total_A
          + Pricing - price action (same SKUs)
          + Pricing - changed conditions (frozen price)
          + Pricing - assortment mix (SKUs in/out)
          + Promo - budget change
          + Promo - response mix (carried set)
          + Assortment - MILP-vs-greedy gap
        = total_B

    Each driver is a *difference of published quantities*, so the bridge is
    auditable line by line:

      - **conditions** freezes plan A's published price and moves the world to
        B's cost/demand/elasticity — the same "frozen action, world moves"
        arithmetic the robustness gate uses (``_frozen_move_uplift``);
      - **price action** is what is left of a shared SKU's Δ once the frozen
        term is taken out — i.e. the effect of recommending a *different* price
        (being the remainder, it also carries the cent-rounding of the published
        price, ≤ €0.12/SKU/yr; the two always add up to the SKU's exact Δ);
      - **mix** is the price-line € carried in or out by the assortment churn;
      - **promo budget** re-solves the concave allocation on **A's** carried set
        at **B's** budget, so the split between "we spent more" and "the
        response curve moved" is a real counterfactual, not an apportionment;
      - **assortment** is the MILP-vs-greedy gap, moved as one term (it is a
        difference of two optimization objectives over different subsets, so it
        does not decompose per SKU — the optimized and benchmark margins ship
        beside it as context instead of being split into fake drivers).

Deterministic: no RNG outside the gate's fixed seed, no wall-clock — the CSV/SVG
deliverables are byte-identical across re-runs. Comparing a plan with itself
yields an all-zero bridge and not a single changed line.

Illustrative planning ranges on synthetic data — what the model computes under
each assumption, not a forecast.

    from revops.compare import compare_plans, predict_context, scenario_by_name
    ctx = predict_context(verbose=False)
    cmp = compare_plans(ctx, scenario_by_name("Baseline"),
                        scenario_by_name("Cost inflation +8%"))

Author: Dimitres Kisimov.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .optimize.promo import optimize_promo
from .robustness import _frozen_move_uplift, analyze
from .scenario import (
    DEFAULT_SCENARIOS,
    PlanSolution,
    PredictContext,
    Scenario,
    predict_context,
    solve_plan,
)
from .simulate import DEFAULT_DRAWS, DEFAULT_RISK_DRIVERS, SEED, RiskDriver

# the published compare: the headline plan against the input-cost what-if — the
# pair that moves every lever at once (prices *and* the range *and* the promo
# response), so the bridge shows all six drivers doing work.
DEFAULT_A_NAME = "Baseline"
DEFAULT_B_NAME = "Cost inflation +8%"

# the bridge drivers, named once and reused by the CSV, the SVG and the CLI.
DRIVER_PRICE_ACTION = "Pricing - price action (same SKUs)"
DRIVER_PRICE_CONDITIONS = "Pricing - changed conditions (frozen price)"
DRIVER_PRICE_MIX = "Pricing - assortment mix (SKUs in/out)"
DRIVER_PROMO_BUDGET = "Promo - budget change"
DRIVER_PROMO_MIX = "Promo - response mix (carried set)"
DRIVER_ASSORTMENT = "Assortment - MILP-vs-greedy gap"
BRIDGE_DRIVERS = (DRIVER_PRICE_ACTION, DRIVER_PRICE_CONDITIONS, DRIVER_PRICE_MIX,
                  DRIVER_PROMO_BUDGET, DRIVER_PROMO_MIX, DRIVER_ASSORTMENT)

KIND_ANCHOR = "anchor"
KIND_DRIVER = "driver"


def scenario_by_name(name: str,
                     scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS) -> Scenario:
    """Look a named scenario up in the library (the CLI's ``--a``/``--b``)."""
    for sc in scenarios:
        if sc.name == name:
            return sc
    known = ", ".join(repr(sc.name) for sc in scenarios)
    raise KeyError(f"unknown scenario {name!r}; known scenarios: {known}")


# --------------------------------------------------------------------------- #
# exact euro arithmetic — the bridge is summed in integer cents                 #
# --------------------------------------------------------------------------- #
def _cents(eur: float) -> int:
    """A published € figure as exact integer cents (every figure the optimizers
    publish is already rounded to the cent, so this is lossless)."""
    return int(round(eur * 100.0))


def _annual_cents(monthly_eur: float) -> int:
    """Annualize a monthly € the way every published price line does: round to
    the cent first, then take twelve of them."""
    return _cents(monthly_eur) * 12


def _eur(cents: int) -> float:
    return round(cents / 100.0, 2)


def _pricing_cents(sol: PlanSolution) -> int:
    return sum(_annual_cents(p.margin_uplift_eur) for p in sol.prices)


def _total_cents(sol: PlanSolution) -> int:
    """Exactly ``evaluate()``'s published total, in cents."""
    return (_pricing_cents(sol)
            + _cents(sol.promo.incremental_margin_eur)
            + _cents(sol.assortment.milp_uplift_vs_greedy_eur))


# --------------------------------------------------------------------------- #
# results                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class BridgeStep:
    """One row of the reconciliation: the two anchors carry the plan totals,
    every driver carries a signed € and the running total after it."""
    order: int
    driver: str
    kind: str                    # "anchor" | "driver"
    delta_eur: float
    running_eur: float


@dataclass
class LineDiff:
    """One thing that differs between the plans. ``metric`` names what ``a_eur``
    and ``b_eur`` measure, because the three kinds are not in the same unit:
    assortment/price rows carry the SKU's annual price-line uplift, promo rows
    carry the category's budget."""
    kind: str                    # "assortment" | "price" | "promo"
    key: str                     # SKU or category
    category: str
    metric: str
    change: str                  # enters/leaves | repriced/revalued/steady | up/down/steady
    a_label: str
    b_label: str
    a_eur: float
    b_eur: float
    delta_eur: float
    conditions_eur: float        # the row's share of the frozen-price driver
    action_eur: float            # ... and of the price-action driver (price rows)
    a_verdict: str               # "" when the SKU is not a gated move in A
    b_verdict: str
    verdict_flip: str            # "" | "accept->hold" | "hold->accept"


@dataclass
class PlanComparison:
    plan_a: str
    plan_b: str
    total_a_eur: float
    total_b_eur: float
    delta_eur: float
    residual_eur: float          # bridge closure error — 0.0 by construction
    pricing_a_eur: float
    pricing_b_eur: float
    promo_a_eur: float
    promo_b_eur: float
    assortment_a_eur: float
    assortment_b_eur: float
    milp_margin_a_eur: float     # context for the assortment driver: the two
    milp_margin_b_eur: float     # objectives whose difference the gap is
    greedy_margin_a_eur: float
    greedy_margin_b_eur: float
    n_carried_a: int
    n_carried_b: int
    capital_used_a_eur: float
    capital_used_b_eur: float
    promo_budget_a_eur: float
    promo_budget_b_eur: float
    entered: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    n_price_changed: int = 0
    n_verdict_flips: int = 0
    gate_draws: int = 0
    gate_seed: int = SEED
    bridge: list[BridgeStep] = field(default_factory=list)
    lines: list[LineDiff] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# the comparison                                                               #
# --------------------------------------------------------------------------- #
def compare_plans(ctx: PredictContext, plan_a: Scenario, plan_b: Scenario,
                  gate_draws: int = 0,
                  drivers: tuple[RiskDriver, ...] = DEFAULT_RISK_DRIVERS,
                  seed: int = SEED) -> PlanComparison:
    """Solve both plans through the existing core and diff them.

    ``gate_draws`` > 0 also replays the robustness gate on **each** plan's own
    moves (same sampler, same seed, each draw composed onto that plan) so a
    verdict flip is a like-for-like comparison; 0 skips it and leaves the
    verdict columns empty. Deterministic either way."""
    sol_a = solve_plan(ctx, plan_a)
    sol_b = solve_plan(ctx, plan_b)

    verdict_a: dict[str, str] = {}
    verdict_b: dict[str, str] = {}
    if gate_draws:
        res_a = analyze(ctx, draws=gate_draws, drivers=drivers, seed=seed,
                        base=plan_a)
        res_b = analyze(ctx, draws=gate_draws, drivers=drivers, seed=seed,
                        base=plan_b)
        verdict_a = {m.sku: m.verdict for m in res_a.moves}
        verdict_b = {m.sku: m.verdict for m in res_b.moves}

    line_a = {p.sku: p for p in sol_a.prices}
    line_b = {p.sku: p for p in sol_b.prices}
    cond_a = {s.sku: s for s in sol_a.price_inputs}
    cond_b = {s.sku: s for s in sol_b.price_inputs}
    category_of = {s.sku: s.category for s in ctx.skus}

    carried_a, carried_b = set(line_a), set(line_b)
    entered = sorted(carried_b - carried_a)
    left = sorted(carried_a - carried_b)
    shared = sorted(carried_a & carried_b)

    # per-SKU annual price-line € exactly as published (cent-rounded, x12)
    ua = {s: _annual_cents(line_a[s].margin_uplift_eur) for s in carried_a}
    ub = {s: _annual_cents(line_b[s].margin_uplift_eur) for s in carried_b}

    # ---- the three pricing drivers ---------------------------------------- #
    mix_c = sum(ub[s] for s in entered) - sum(ua[s] for s in left)
    cond_c = 0
    action_c = 0
    frozen: dict[str, tuple[int, int]] = {}
    for s in shared:
        p_frozen = line_a[s].recommended_price          # A's published action
        sa, sb = cond_a[s], cond_b[s]
        f_a = _annual_cents(_frozen_move_uplift(
            sa.current_price_eur, sa.unit_cost_eur, sa.avg_monthly_demand,
            sa.elasticity, p_frozen))
        f_b = _annual_cents(_frozen_move_uplift(
            sb.current_price_eur, sb.unit_cost_eur, sb.avg_monthly_demand,
            sb.elasticity, p_frozen))
        frozen[s] = (f_a, f_b)
        cond_c += f_b - f_a                              # frozen action, world moves
        action_c += (ub[s] - ua[s]) - (f_b - f_a)        # the rest: a new price

    # ---- the two promo drivers -------------------------------------------- #
    promo_a_c = _cents(sol_a.promo.incremental_margin_eur)
    promo_b_c = _cents(sol_b.promo.incremental_margin_eur)
    # counterfactual: A's carried set (as the promo optimizer saw it) at B's budget
    cross_c = _cents(optimize_promo(sol_a.promo_inputs,
                                    plan_b.promo_budget).incremental_margin_eur)
    promo_budget_c = cross_c - promo_a_c
    promo_mix_c = promo_b_c - cross_c

    # ---- the assortment driver -------------------------------------------- #
    assort_a_c = _cents(sol_a.assortment.milp_uplift_vs_greedy_eur)
    assort_b_c = _cents(sol_b.assortment.milp_uplift_vs_greedy_eur)
    assort_c = assort_b_c - assort_a_c

    # ---- the bridge -------------------------------------------------------- #
    total_a_c, total_b_c = _total_cents(sol_a), _total_cents(sol_b)
    deltas = [action_c, cond_c, mix_c, promo_budget_c, promo_mix_c, assort_c]
    residual_c = (total_b_c - total_a_c) - sum(deltas)

    # anchors carry no delta of their own, so the delta column sums to exactly
    # the plan-to-plan difference and the running column closes on plan B.
    bridge = [BridgeStep(0, plan_a.name, KIND_ANCHOR, 0.0, _eur(total_a_c))]
    running = total_a_c
    for i, (name, d) in enumerate(zip(BRIDGE_DRIVERS, deltas, strict=True), start=1):
        running += d
        bridge.append(BridgeStep(i, name, KIND_DRIVER, _eur(d), _eur(running)))
    bridge.append(BridgeStep(len(BRIDGE_DRIVERS) + 1, plan_b.name, KIND_ANCHOR,
                             0.0, _eur(total_b_c)))

    # ---- the line-level diff ----------------------------------------------- #
    lines: list[LineDiff] = []

    def _flip(sku: str) -> tuple[str, str, str]:
        va, vb = verdict_a.get(sku, ""), verdict_b.get(sku, "")
        return va, vb, (f"{va}->{vb}" if va and vb and va != vb else "")

    for sku in entered + left:
        enters = sku in carried_b
        va, vb, flip = _flip(sku)
        a_c, b_c = (0, ub[sku]) if enters else (ua[sku], 0)
        lines.append(LineDiff(
            kind="assortment", key=sku, category=category_of.get(sku, ""),
            metric="price-line uplift EUR/yr",
            change="enters" if enters else "leaves",
            a_label="dropped" if enters else f"{line_a[sku].price_change_pct:+.1f}%",
            b_label=f"{line_b[sku].price_change_pct:+.1f}%" if enters else "dropped",
            a_eur=_eur(a_c), b_eur=_eur(b_c), delta_eur=_eur(b_c - a_c),
            conditions_eur=0.0, action_eur=0.0,          # this row IS the mix driver
            a_verdict=va, b_verdict=vb, verdict_flip=flip))

    for sku in shared:
        pa, pb = line_a[sku].recommended_price, line_b[sku].recommended_price
        f_a, f_b = frozen[sku]
        va, vb, flip = _flip(sku)
        if pa == pb and ua[sku] == ub[sku] and f_a == f_b and not flip:
            continue                                     # nothing to report
        change = ("repriced" if pa != pb
                  else "revalued" if ua[sku] != ub[sku] else "steady")
        lines.append(LineDiff(
            kind="price", key=sku, category=category_of.get(sku, ""),
            metric="price-line uplift EUR/yr", change=change,
            a_label=f"{line_a[sku].price_change_pct:+.1f}%",
            b_label=f"{line_b[sku].price_change_pct:+.1f}%",
            a_eur=_eur(ua[sku]), b_eur=_eur(ub[sku]),
            delta_eur=_eur(ub[sku] - ua[sku]),
            conditions_eur=_eur(f_b - f_a),
            action_eur=_eur((ub[sku] - ua[sku]) - (f_b - f_a)),
            a_verdict=va, b_verdict=vb, verdict_flip=flip))

    alloc_a, alloc_b = sol_a.promo.allocation, sol_b.promo.allocation
    for cat in sorted(set(alloc_a) | set(alloc_b)):
        a_c, b_c = _cents(alloc_a.get(cat, 0.0)), _cents(alloc_b.get(cat, 0.0))
        lines.append(LineDiff(
            kind="promo", key=cat, category=cat, metric="promo budget EUR",
            change="up" if b_c > a_c else "down" if b_c < a_c else "steady",
            a_label="", b_label="",
            a_eur=_eur(a_c), b_eur=_eur(b_c), delta_eur=_eur(b_c - a_c),
            conditions_eur=0.0, action_eur=0.0,
            a_verdict="", b_verdict="", verdict_flip=""))

    # deterministic reading order: biggest absolute € first inside each kind
    order = {"assortment": 0, "price": 1, "promo": 2}
    lines.sort(key=lambda r: (order[r.kind], -abs(_cents(r.delta_eur)), r.key))

    return PlanComparison(
        plan_a=plan_a.name, plan_b=plan_b.name,
        total_a_eur=_eur(total_a_c), total_b_eur=_eur(total_b_c),
        delta_eur=_eur(total_b_c - total_a_c), residual_eur=_eur(residual_c),
        pricing_a_eur=_eur(_pricing_cents(sol_a)),
        pricing_b_eur=_eur(_pricing_cents(sol_b)),
        promo_a_eur=_eur(promo_a_c), promo_b_eur=_eur(promo_b_c),
        assortment_a_eur=_eur(assort_a_c), assortment_b_eur=_eur(assort_b_c),
        milp_margin_a_eur=sol_a.assortment.total_margin_eur,
        milp_margin_b_eur=sol_b.assortment.total_margin_eur,
        greedy_margin_a_eur=sol_a.assortment.greedy_margin_eur,
        greedy_margin_b_eur=sol_b.assortment.greedy_margin_eur,
        n_carried_a=sol_a.assortment.n_carried,
        n_carried_b=sol_b.assortment.n_carried,
        capital_used_a_eur=sol_a.assortment.capital_used_eur,
        capital_used_b_eur=sol_b.assortment.capital_used_eur,
        promo_budget_a_eur=plan_a.promo_budget,
        promo_budget_b_eur=plan_b.promo_budget,
        entered=entered, left=left,
        n_price_changed=sum(1 for r in lines if r.kind == "price"
                            and r.change in ("repriced", "revalued")),
        n_verdict_flips=sum(1 for r in lines if r.verdict_flip),
        gate_draws=int(gate_draws), gate_seed=seed,
        bridge=bridge, lines=lines)


# --------------------------------------------------------------------------- #
# deterministic deliverables — tall summary + bridge + line CSVs, bridge SVG   #
# --------------------------------------------------------------------------- #
_BRIDGE_COLS = ["order", "driver", "kind", "delta_eur", "running_eur"]
_LINE_COLS = ["kind", "key", "category", "metric", "change", "a_label", "b_label",
              "a_eur", "b_eur", "delta_eur", "conditions_eur", "action_eur",
              "a_verdict", "b_verdict", "verdict_flip"]


def summary_rows(cmp: PlanComparison) -> list[tuple[str, float]]:
    """The (metric, value) rows written to the summary CSV — the same tall,
    numeric-only shape as ``uplift_simulation.csv`` so Power BI can read a
    scalar with one filter. The plan *names* live on the bridge CSV's anchor
    rows, which keeps this table's value column numeric."""
    return [
        ("total_a_eur", cmp.total_a_eur), ("total_b_eur", cmp.total_b_eur),
        ("delta_eur", cmp.delta_eur), ("bridge_residual_eur", cmp.residual_eur),
        ("pricing_a_eur", cmp.pricing_a_eur), ("pricing_b_eur", cmp.pricing_b_eur),
        ("promo_a_eur", cmp.promo_a_eur), ("promo_b_eur", cmp.promo_b_eur),
        ("assortment_a_eur", cmp.assortment_a_eur),
        ("assortment_b_eur", cmp.assortment_b_eur),
        ("milp_margin_a_eur", cmp.milp_margin_a_eur),
        ("milp_margin_b_eur", cmp.milp_margin_b_eur),
        ("greedy_margin_a_eur", cmp.greedy_margin_a_eur),
        ("greedy_margin_b_eur", cmp.greedy_margin_b_eur),
        ("n_carried_a", cmp.n_carried_a), ("n_carried_b", cmp.n_carried_b),
        ("capital_used_a_eur", cmp.capital_used_a_eur),
        ("capital_used_b_eur", cmp.capital_used_b_eur),
        ("promo_budget_a_eur", cmp.promo_budget_a_eur),
        ("promo_budget_b_eur", cmp.promo_budget_b_eur),
        ("n_entered", len(cmp.entered)), ("n_left", len(cmp.left)),
        ("n_price_changed", cmp.n_price_changed),
        ("n_verdict_flips", cmp.n_verdict_flips),
        ("gate_draws", cmp.gate_draws), ("seed", cmp.gate_seed),
    ]


def write_summary_csv(cmp: PlanComparison, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for metric, value in summary_rows(cmp):
            w.writerow([metric, value])
    return p


def write_bridge_csv(cmp: PlanComparison, path: str | Path) -> Path:
    """The reconciliation, one row per step (anchors included, so the file
    itself proves the identity closes)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_BRIDGE_COLS)
        for step in cmp.bridge:
            d = asdict(step)
            w.writerow([d[c] for c in _BRIDGE_COLS])
    return p


def write_lines_csv(cmp: PlanComparison, path: str | Path) -> Path:
    """Byte-stable per-line diff (grain = SKU or category; joins ``dim_sku`` /
    ``dim_category`` in Power BI)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_LINE_COLS)
        for row in cmp.lines:
            d = asdict(row)
            w.writerow([d[c] for c in _LINE_COLS])
    return p


# "decision desk" tokens — the dataviz-validated palette shared with the other
# SVG deliverables and the web dashboard. A bridge's job is polarity, so it
# wears the diverging pair (blue adds, red subtracts) with a neutral gray for a
# zero step — never a hue at the midpoint. Both poles are validated against this
# surface (and their dark-mode steps against the dashboard's) and no step
# travels on color alone: every bar is direct-labelled with its signed €.
_SURFACE, _INK, _SEC, _MUTED, _GRID = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9")
_AXIS, _NEUTRAL = "#c3c2b7", "#c3c2b7"
_BLUE, _RED = "#2a78d6", "#e34948"
_FONT = "system-ui, Segoe UI, sans-serif"


def _esc(text: str) -> str:
    """XML-escape a label: scenario names, SKUs and categories are data, and the
    SVG has to stay well-formed whatever they contain."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _axis_bounds(vmin: float, vmax: float, step: float = 5_000.0) -> tuple[float, float]:
    """Deterministic padded axis: pad ~8% each side, snap out to a grid step."""
    import math
    span = max(1.0, vmax - vmin)
    lo = vmin - 0.08 * span
    hi = vmax + 0.08 * span
    return (math.floor(lo / step) * step, math.ceil(hi / step) * step)


def _step_path(x0: float, x1: float, y: float, h: float, r: float = 3.0) -> str:
    """A floating bridge segment: both ends are data ends, so both are rounded.
    Degenerates to a sliver rect when the step is too narrow to round."""
    if x1 < x0:
        x0, x1 = x1, x0
    w = x1 - x0
    r = min(r, w / 2, h / 2)
    if r < 0.75:
        return f'M{x0:.1f} {y:.1f}H{x1:.1f}V{y + h:.1f}H{x0:.1f}Z'
    return (f'M{x0 + r:.1f} {y:.1f}H{x1 - r:.1f}'
            f'A{r:.1f} {r:.1f} 0 0 1 {x1:.1f} {y + r:.1f}'
            f'V{y + h - r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x1 - r:.1f} {y + h:.1f}'
            f'H{x0 + r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x0:.1f} {y + h - r:.1f}'
            f'V{y + r:.1f}A{r:.1f} {r:.1f} 0 0 1 {x0 + r:.1f} {y:.1f}Z')


def render_bridge_svg(cmp: PlanComparison, path: str | Path) -> Path:
    """Hand-drawn horizontal € bridge: the two plan totals are dashed ink rules
    that frame the plot, and between them each named driver is a floating
    segment from the running total before it to the running total after it —
    blue when it adds, red when it subtracts, a neutral tick when it is exactly
    zero. Hairline connectors carry the running total from one step to the next,
    so the staircase closes on plan B by construction. Deterministic."""
    steps = cmp.bridge
    drivers = [s for s in steps if s.kind == KIND_DRIVER]
    W = 760
    top, row_h, gap = 128, 20, 16
    n = len(steps)
    plot_h = n * (row_h + gap)
    H = top + plot_h + 74
    ml, mr = 288, 104
    plot_w = W - ml - mr
    hx = 28                                       # header left edge

    levels = [s.running_eur for s in steps]
    ax_lo, ax_hi = _axis_bounds(min(levels), max(levels))

    def x(v: float) -> float:
        return ml + (v - ax_lo) / (ax_hi - ax_lo) * plot_w

    def row_y(i: int) -> float:
        return top + i * (row_h + gap) + gap / 2

    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="{_FONT}">')
    s.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" '
             f'fill="{_SURFACE}" stroke="{_GRID}"/>')
    s.append(f'<text x="{hx}" y="34" font-size="16" font-weight="700" '
             f'fill="{_INK}">Plan compare - where the EUR went, driver by driver</text>')
    s.append(f'<text x="{hx}" y="52" font-size="11" fill="{_SEC}">'
             f'{_esc(cmp.plan_a)} vs {_esc(cmp.plan_b)} &#183; both re-solved through '
             f'the same MILP + pricing + promo core</text>')
    s.append(f'<text x="{hx}" y="70" font-size="11" fill="{_INK}">EUR '
             f'{cmp.total_a_eur:,.0f} &#8594; {cmp.total_b_eur:,.0f}/yr = '
             f'<tspan font-weight="700">{cmp.delta_eur:+,.0f}</tspan>, reconciled '
             f'exactly by {len(drivers)} named drivers</text>')
    ne, nl, npc = len(cmp.entered), len(cmp.left), cmp.n_price_changed
    flips = (f' &#183; {cmp.n_verdict_flips} gate verdict'
             f'{"" if cmp.n_verdict_flips == 1 else "s"} flip'
             f'{"s" if cmp.n_verdict_flips == 1 else ""}'
             if cmp.gate_draws else "")
    s.append(f'<text x="{hx}" y="88" font-size="11" fill="{_SEC}">'
             f'{ne} SKU{"" if ne == 1 else "s"} enter{"s" if ne == 1 else ""} '
             f'&#183; {nl} leave{"s" if nl == 1 else ""} &#183; '
             f'{npc} price line{"" if npc == 1 else "s"} '
             f'change{"s" if npc == 1 else ""}{flips}</text>')
    # polarity key — the two poles, named in ink (color never travels alone)
    s.append(f'<rect x="{hx}" y="{100}" width="9" height="9" rx="2" fill="{_BLUE}"/>')
    s.append(f'<text x="{hx + 14}" y="{108.5}" font-size="10.5" fill="{_INK}">adds '
             f'to the total</text>')
    s.append(f'<rect x="{hx + 118}" y="{100}" width="9" height="9" rx="2" '
             f'fill="{_RED}"/>')
    s.append(f'<text x="{hx + 132}" y="{108.5}" font-size="10.5" fill="{_INK}">'
             f'subtracts</text>')

    # x gridlines + labels (solid hairlines, recessive)
    for i in range(5):
        v = ax_lo + (ax_hi - ax_lo) * i / 4
        gx = x(v)
        s.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" '
                 f'stroke="{_GRID}" stroke-width="1"/>')
        s.append(f'<text x="{gx:.1f}" y="{top + plot_h + 18:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="{_MUTED}">EUR {v / 1000:.0f}k</text>')

    # the two anchors as dashed ink rules — the frame the staircase moves between
    for step in steps:
        if step.kind != KIND_ANCHOR:
            continue
        axx = x(step.running_eur)
        s.append(f'<line x1="{axx:.1f}" y1="{top - 6}" x2="{axx:.1f}" '
                 f'y2="{top + plot_h}" stroke="{_INK}" stroke-width="1.3" '
                 f'stroke-dasharray="4 3" opacity="0.7"/>')

    prev_x = None
    for i, step in enumerate(steps):
        y0 = row_y(i)
        yc = y0 + row_h / 2
        anchor = step.kind == KIND_ANCHOR
        end_x = x(step.running_eur)
        start_x = end_x if anchor else x(step.running_eur - step.delta_eur)

        # connector: carry the running total down from the step above
        if prev_x is not None:
            s.append(f'<line x1="{prev_x:.1f}" y1="{y0 - gap:.1f}" '
                     f'x2="{prev_x:.1f}" y2="{y0:.1f}" stroke="{_AXIS}" '
                     f'stroke-width="1"/>')

        if anchor:
            s.append(f'<rect x="{end_x - 1.5:.1f}" y="{y0:.1f}" width="3" '
                     f'height="{row_h}" rx="1.5" fill="{_INK}"/>')
            label = f"EUR {step.running_eur:,.0f}"
        elif abs(step.delta_eur) < 0.005:
            s.append(f'<rect x="{end_x - 1:.1f}" y="{y0 + 6:.1f}" width="2" '
                     f'height="{row_h - 12}" fill="{_NEUTRAL}"/>')
            label = "0"
        else:
            color = _BLUE if step.delta_eur > 0 else _RED
            # a step too narrow to draw still gets a visible 1.2px sliver,
            # anchored at the running total it ends on
            x1 = end_x
            if abs(end_x - start_x) < 1.2:
                x1 = start_x + (1.2 if end_x >= start_x else -1.2)
            s.append(f'<path d="{_step_path(start_x, x1, y0, row_h)}" '
                     f'fill="{color}"/>')
            label = f"{step.delta_eur:+,.0f}"

        # driver name in the left gutter; running total in the right column
        weight = ' font-weight="700"' if anchor else ""
        s.append(f'<text x="{ml - 12}" y="{yc + 3:.1f}" font-size="11" '
                 f'text-anchor="end" fill="{_INK}"{weight}>{_esc(step.driver)}</text>')
        # the step's own € — outside the segment on its data end, so it can
        # never be clipped by a short bar
        if anchor:
            s.append(f'<text x="{end_x + 8:.1f}" y="{yc + 3:.1f}" font-size="10" '
                     f'fill="{_INK}" font-weight="700">{label}</text>')
        else:
            right = end_x >= start_x
            tx = (max(start_x, end_x) + 6) if right else (min(start_x, end_x) - 6)
            anch = "start" if right else "end"
            s.append(f'<text x="{tx:.1f}" y="{yc + 3:.1f}" font-size="10" '
                     f'text-anchor="{anch}" fill="{_INK}">{label}</text>')
            s.append(f'<text x="{ml + plot_w + 12}" y="{yc + 3:.1f}" font-size="9.5" '
                     f'fill="{_MUTED}">{step.running_eur:,.0f}</text>')
        prev_x = end_x

    s.append(f'<text x="{ml + plot_w / 2:.1f}" y="{H - 40}" font-size="10" '
             f'text-anchor="middle" fill="{_INK}">Total expected uplift '
             f'(EUR / year)</text>')
    s.append(f'<text x="{hx}" y="{H - 22}" font-size="10" fill="{_SEC}">Every euro '
             f'is attributed to a named driver: the bridge is summed in cents and '
             f'closes exactly (residual EUR {cmp.residual_eur:,.2f}).</text>')
    s.append(f'<text x="{hx}" y="{H - 8}" font-size="10" fill="{_MUTED}">'
             f'Illustrative planning ranges on synthetic data - what the model '
             f'computes under each assumption, not a forecast.</text>')
    s.append('</svg>')

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s) + "\n", encoding="utf-8")
    return p


def write_compare_deliverables(cmp: PlanComparison,
                               outdir: str | Path = "deliverables") -> dict:
    """Emit the four deterministic compare deliverables and return their paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "plan_compare_summary.csv": write_summary_csv(
            cmp, out / "plan_compare_summary.csv"),
        "plan_compare_bridge.csv": write_bridge_csv(
            cmp, out / "plan_compare_bridge.csv"),
        "plan_compare_lines.csv": write_lines_csv(
            cmp, out / "plan_compare_lines.csv"),
        "plan_compare.svg": render_bridge_svg(cmp, out / "plan_compare.svg"),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _print_tables(cmp: PlanComparison) -> None:
    line = "=" * 78
    print(line)
    print(" REVOPS — plan compare (A vs B over the same predict->optimize core)")
    print(line)
    print(f" A  {cmp.plan_a:<28} EUR {cmp.total_a_eur:>12,.0f}/yr  "
          f"{cmp.n_carried_a} SKUs")
    print(f" B  {cmp.plan_b:<28} EUR {cmp.total_b_eur:>12,.0f}/yr  "
          f"{cmp.n_carried_b} SKUs")
    print("-" * 78)
    print(" Bridge — every euro attributed to a named driver:")
    for step in cmp.bridge:
        if step.kind == KIND_ANCHOR:
            print(f"   {step.driver:<46}{'':>12}  EUR {step.running_eur:>12,.2f}")
        else:
            print(f"   {step.driver:<46}{step.delta_eur:>+12,.2f}"
                  f"  -> {step.running_eur:>12,.2f}")
    print(f"   {'residual (must be 0.00)':<46}{cmp.residual_eur:>+12,.2f}")
    print("-" * 78)
    print(f" Assortment: {len(cmp.entered)} enter, {len(cmp.left)} leave"
          f"   |   price lines changed: {cmp.n_price_changed}"
          f"   |   verdict flips: {cmp.n_verdict_flips}"
          f" ({cmp.gate_draws} draws)")
    top = [r for r in cmp.lines if r.kind != "promo"][:8]
    if top:
        print(f" {'kind':<11}{'key':<9}{'change':<10}{'A':>12}{'B':>12}"
              f"{'delta':>12}  verdict")
        for r in top:
            flip = r.verdict_flip or (r.a_verdict or r.b_verdict or "")
            print(f" {r.kind:<11}{r.key:<9}{r.change:<10}{r.a_eur:>12,.0f}"
                  f"{r.b_eur:>12,.0f}{r.delta_eur:>+12,.0f}  {flip}")
    promo = [r for r in cmp.lines if r.kind == "promo" and abs(r.delta_eur) >= 1.0]
    if promo:
        moved = ", ".join(f"{r.key} {r.delta_eur:+,.0f}" for r in promo[:5])
        print(f" Promo reallocation: {moved}")
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

    ap = argparse.ArgumentParser(prog="revops.compare", description=__doc__)
    # scenario names carry '%' and argparse %-formats its help strings
    ap.add_argument("--a", default=DEFAULT_A_NAME, metavar="NAME",
                    help=f"plan A, a named scenario "
                         f"(default {DEFAULT_A_NAME!r})".replace("%", "%%"))
    ap.add_argument("--b", default=DEFAULT_B_NAME, metavar="NAME",
                    help=f"plan B, a named scenario "
                         f"(default {DEFAULT_B_NAME!r})".replace("%", "%%"))
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                    help=f"joint draws used to gate BOTH plans' price moves "
                         f"(default {DEFAULT_DRAWS}, matching revops.simulate; "
                         f"0 skips the gate and leaves the verdict columns empty)")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the full comparison to this JSON file")
    ap.add_argument("--outdir", default="deliverables",
                    help="where to write the CSV/SVG deliverables")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the forecaster's training printout")
    args = ap.parse_args(argv)

    try:
        plan_a, plan_b = scenario_by_name(args.a), scenario_by_name(args.b)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 2

    ctx = predict_context(verbose=not args.quiet)

    # The HiGHS build leaks solver chatter to the OS-level stdout; silence it at
    # the file-descriptor level only around the solving, so the tables below stay
    # clean. Progress goes to stderr, which is left untouched.
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

    if args.draws:
        print(f"Comparing {args.a!r} vs {args.b!r} and gating both plans across "
              f"{args.draws} joint draws (re-solving the MILP each draw)...",
              file=sys.stderr, flush=True)
    with _quiet_fd():
        cmp = compare_plans(ctx, plan_a, plan_b, gate_draws=args.draws)

    paths = write_compare_deliverables(cmp, outdir=args.outdir)
    _print_tables(cmp)
    for name, p in paths.items():
        print(f"  wrote {name:<28} {p}")

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(cmp), f, indent=2)
        print(f"\nFull comparison written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
