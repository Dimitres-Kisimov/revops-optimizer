# Use case — a quarterly RevOps decision

A regional industrial distributor carries ~240 SKUs across eight categories:
fasteners, power tools, chemicals, PPE, hand tools, abrasives, electrical,
plumbing. Working capital is finite, the warehouse shelf is finite, and every
quarter the same four questions come back to the commercial team. This is the
narrative the tool automates, end to end, into one prescriptive plan.

## The quarterly questions

1. **What should we carry?** The catalogue has grown a long tail of slow movers.
   Capital and shelf space are both tight — which SKUs earn their place?
2. **How much of each should we hold?** High enough to hit service, low enough
   not to drown in holding cost — per SKU, not one blanket rule.
3. **What should we charge?** Some ranges have room to move on price; some don't.
   Where is margin left on the table, within a sane guardrail?
4. **Where should the promo budget go?** A fixed budget, diminishing returns —
   which categories convert spend into margin best?

## The decision flow

```
   forecast demand ─┬─► ASSORTMENT MILP  ──► carry 29 / drop 211  (capital + shelf)
   estimate elast.  │        │ carried set
   score decline ───┘        ▼
                       INVENTORY newsvendor ──► order-up-to, ROP, safety stock
                       PRICING Lerner+band  ──► 29 price moves within ±15%
                       PROMO water-filling  ──► category budget split
                                 │
                                 ▼
                      PRESCRIPTIVE PLAN  ──►  € uplift + named actions
```

First the predictive layer runs: a next-month demand forecast per SKU, a
per-category price elasticity, and a decline probability (see `METHODOLOGY.md`).
Those estimates — not raw historical averages — become the inputs to the four
optimizers (see `OPTIMIZATION_MODELS.md`).

- **Assortment** solves the two-constraint knapsack MILP on forecast margins,
  with declining SKUs demand-haircut so they fall out first. Here it carries a
  tight **29 of 240** SKUs against the €60k capital and 10 m³ shelf budgets, and
  earns a **six-figure €/yr** margin advantage over the GMROI greedy heuristic —
  because with shelf *and* capital binding, no single ratio ranks the range
  correctly.
- **Inventory** sets a newsvendor order-up-to level, reorder point and safety
  stock for each carried SKU from the forecast and target service level.
- **Pricing** applies the Lerner markup using the *estimated* elasticity, clamps
  every move to ±15%, and flags inelastic SKUs rather than over-pricing them.
- **Promo** water-fills the €40k budget across categories by marginal return,
  beating an even split.

## The output the team actually receives

Everything downstream is generated from one `prescribe()` call:

- **`deliverables/executive_review.pdf` / `.pptx`** — an ~8-slide deck opening
  with the uplift waterfall (baseline → +pricing → +promo → +assortment →
  optimized) and ending with a recommended-actions slide.
- **`deliverables/optimization_workbook.xlsx`** — Summary, Assortment, Inventory,
  Pricing, Promo sheets for the analyst who wants the numbers.
- **`deliverables/actions.csv`** — the operational list: per carried SKU, the
  reorder quantity (order-up-to), reorder point, and the price change to make.
- **`powerbi/`** — the same plan as a Power BI star schema + DAX for the BI team.
- **`web/index.html`** — an offline dashboard with a live promo what-if slider.

## The bottom line

The plan quantifies a **total expected annual uplift** against the current
baseline (current prices, no promo optimization, a naive GMROI assortment),
decomposed into pricing + promo + assortment, and hands the commercial team a
named action list rather than a model. On this synthetic dataset that lands
around **€160k/yr**, of which the assortment MILP contributes the large majority
— the concrete artefact of the predict→optimize pipeline. Numbers are
regenerated on every run; the exact figures in the committed deliverables come
from the seeded synthetic data.
