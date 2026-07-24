# Business case — RevOps Optimizer

*An enterprise-framed read of the same pipeline the rest of this repo builds. The
distributor below is an illustrative scenario; every impact number is measured by
`prescribe()` on the seeded synthetic dataset and labelled where it's an estimate.*

## Situation

**Meridian Industrieteile GmbH** (fictional, but drawn to type) is a regional
industrial distributor: ~240 active SKUs across eight categories — fasteners,
power tools, chemicals, PPE, hand tools, abrasives, electrical, plumbing. It sells
B2B to trades and small manufacturers out of one warehouse. Working capital is
finite (roughly €60k of catalogue-level buying budget in the modelled slice) and
shelf space is finite (~10 m³ of the relevant racking). Every quarter the same
four commercial decisions come back: **what to carry, how much to hold, what to
charge, and where to spend the promo budget.**

Today those four decisions are made the way they are in most distributors this
size — in a stack of spreadsheets, by feel and by last year's ranking, each in its
own silo. The range review looks at GMROI, pricing is cost-plus, promo is split
evenly across categories "to be fair," and inventory is a blanket weeks-of-cover
rule. Nothing connects the demand forecast to any of it.

## Problem (quantified)

Three gaps, each measurable, plus the process cost of running it by hand.

**1. Assortment left on ranking heuristics.** With *both* the capital budget and
the shelf constraint binding, ranking the range by a single ratio (GMROI) is
provably sub-optimal — no single ratio orders the catalogue correctly when two
constraints compete. Measured gap between the greedy GMROI range and the
two-constraint MILP: **€106,162 / year**.

**2. Pricing set cost-plus, not to elasticity.** Some categories have room to move
on price; some don't. Moving each SKU to its Lerner-optimal markup (using the
*estimated* elasticity, clamped to ±15%) versus holding current prices recovers a
measured **€35,220 / year** in margin.

**3. Promo budget split evenly.** A fixed €40k promo budget spread flat across
categories ignores diminishing returns. Water-filling it by marginal return
instead of an even split is worth a measured **€18,584 / year**.

**Process cost (illustrative, assumptions stated).** Building the quarterly range
/ price / promo plan by hand: assume **2 analysts × 3 days per quarter** = 6
person-days ≈ 48 hours/quarter → **~192 hours/year**. At a fully-loaded **€70/hour**
that is **~€13,400/year** of analyst time — and because the manual process is slow,
in practice it runs closer to annually than quarterly, so prices and range drift
uncorrected for most of the year. (These are stated assumptions for scale, not
measured figures.)

Adding the three measured margin gaps: **€106,162 + €35,220 + €18,584 = €159,966 /
year** of margin that the current, disconnected process leaves on the table.

## Solution

`prescribe()` runs the predictive layer once — a next-month demand forecast per
SKU, a per-category price elasticity, and a decline probability — and feeds those
estimates (not raw historical averages) into four coupled optimizers:

- **Assortment** — a two-constraint knapsack **MILP** on forecast margins, with
  declining SKUs demand-haircut so they drop out first.
- **Inventory** — a **newsvendor** order-up-to level, reorder point and safety
  stock per carried SKU, from the forecast and the target service level.
- **Pricing** — the **Lerner** markup on estimated elasticity, clamped to ±15%,
  flagging inelastic SKUs rather than over-pricing them.
- **Promo** — **water-filling** the promo budget across categories by marginal
  return.

The output is one prescriptive plan: a single € uplift number, decomposed, plus a
named per-SKU action list — not a model to admire, a plan to execute.

## Impact / ROI (measured on the seeded dataset)

| Lever | Measured annual uplift |
|---|--:|
| Pricing (Lerner vs cost-plus) | €35,220 |
| Promo (water-fill vs even split) | €18,584 |
| Assortment (MILP vs greedy GMROI) | €106,162 |
| **Total expected uplift** | **€159,966 / year** |

Supporting model quality, also measured:

- **Assortment MILP** carries a tight **29 of 240** SKUs against the €60k capital
  and 10 m³ shelf budgets (capital used €56,152 / €60,000).
- **Forecast MASE 0.75** on the held-out horizon — it beats the seasonal-naive
  baseline (MASE 1.01), so the demand signal driving every optimizer is genuinely
  better than "same as last period."
- **Decline-risk classifier ROC-AUC 0.99** — the haircut that drops fragile SKUs
  first is well-separated.

Against ~€13,400/year of analyst time replaced (illustrative), a **~€160k/year**
measured margin uplift is the headline. The large majority of it — the €106k
assortment gap — is the concrete artefact of the predict→optimize handoff, and it
exists *because* a second (shelf) constraint binds; with capital alone the gap
collapses, and this repo says so plainly.

## Stakeholders & use case

Run at the **quarterly business review / S&OP** cycle:

1. **Data/BI analyst** regenerates the synthetic dataset (or, in production, loads
   the real ledger) and runs `python -m revops` and `python -m revops.report`.
2. **Category / commercial manager** reads the uplift waterfall and the assortment
   before/after, and signs off which SKUs to drop.
3. **Pricing manager** reviews the ±15%-clamped price moves and the inelastic-SKU
   flags before any list change goes live.
4. **Supply/inventory planner** takes `actions.csv` — per-SKU order-up-to, reorder
   point and safety stock — as the buy-list.
5. **Commercial director** approves the promo split and the headline € plan in the
   review meeting, working from `executive_review.pdf`.
6. **BI team** publishes the same plan through the Power BI star schema for ongoing
   tracking.

## Deliverable

Everything below is **already produced** by the pipeline — this business case
points to it, it does not recreate it:

- **`deliverables/executive_review.pdf`** / **`.pptx`** — the ~8-slide executive
  deck (uplift waterfall, assortment before/after, inventory frontier, price-move
  distribution, promo allocation, model-quality slide, recommended actions).
- **`deliverables/optimization_workbook.xlsx`** — Summary / Assortment / Inventory
  / Pricing / Promo sheets for the analyst.
- **`deliverables/actions.csv`** — the operational per-SKU reorder + reprice list.
- **`deliverables/prescription.json`** — the whole plan, machine-readable.
- **`powerbi/`** — the same plan as a star schema + DAX for the BI team.
- **`web/index.html`** — an offline dashboard with a live promo what-if slider.

## Honest notes

- **The data is synthetic** — generated with real structure (seasonality, a
  risk-linked trend, a genuine price↔quantity relationship per SKU) so the models
  have something real to recover, but it is not a real distributor's ledger.
- **Impact figures are measured on that seeded dataset** and regenerate on every
  run; the process-cost / analyst-time figures are **illustrative assumptions**,
  labelled as such above.
- **The MILP advantage depends on the shelf constraint binding.** Relax
  `--shelf-capacity-m3` and the €106k gap shrinks toward zero — the headline is
  real precisely because a second constraint competes, and this repo would rather
  state that than imply the MILP always wins.
