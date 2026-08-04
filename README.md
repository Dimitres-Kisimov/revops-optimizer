# RevOps Optimizer

[![CI](https://github.com/Dimikissimov/revops-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Dimikissimov/revops-optimizer/actions/workflows/ci.yml)

This is a revenue-and-operations decision engine for a distributor: it forecasts
demand, estimates price elasticity and decline risk, and then feeds those
estimates into four optimizers — assortment, inventory, pricing and promotion —
to produce one prescriptive plan with a single € uplift number and a named action
list. I built it because I kept seeing "AI" projects that stop at a prediction
and "optimization" projects that assume their inputs fall from the sky. The
interesting engineering is the seam between them, so I made that seam the whole
project.

![RevOps Optimizer web dashboard — KPI tiles, uplift waterfall, assortment margin and promo allocation](docs/img/dashboard.png)

I wrote it as part of an internship application in data & AI analytics. It runs
end to end on synthetic-but-structured data with no API keys and no cloud.

**Business case:** on the seeded dataset the plan is worth a measured **~€160k/year**
of margin uplift for a mid-size industrial distributor — see
[`docs/BUSINESS_CASE.md`](docs/BUSINESS_CASE.md) for the situation, the quantified
problem and the ROI.

![Uplift waterfall — baseline to optimized](docs/img/uplift_waterfall.png)

## What it does, concretely

`prescribe()` runs the predictive layer, hands its outputs to the optimization
core, and assembles a plan. On the seeded dataset (240 SKUs, eight categories):

```
Expected uplift        EUR      159,966 / year
  - pricing            EUR       35,220
  - promo              EUR       18,584
  - assortment (MILP)  EUR      106,162
SKUs carried           29 of 240   (capital EUR 56,152 / 60,000)
MILP vs greedy         EUR      106,162 / year   (two constraints: capital + shelf)
Forecast MASE          0.75  (seasonal-naive 1.01; beats naive)
Decline-risk ROC-AUC   0.99
```

The predict→optimize handoff: the demand forecast replaces the raw average in the
newsvendor and the assortment MILP; the *estimated* elasticity (not the true one)
drives the pricing markup; the decline probability haircuts fragile SKUs so the
range decision drops them first. The maths is written out in
[`docs/OPTIMIZATION_MODELS.md`](docs/OPTIMIZATION_MODELS.md) and the handoff in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

### Forecast uncertainty → service level (fill rate)

A point forecast is not enough to set inventory — a planner sets a *service
level*. So the forecaster's own one-step residuals (how wrong it actually is)
become the demand-uncertainty: standardized, pooled, and swept from an 80%→99%
target to trade off safety stock, holding cost, expected fill rate and expected
stockout cost. On the seeded assortment the cost-minimising recommendation is a
**96% target ≈ 99.3% expected unit fill rate**. The honest catch: the residuals
are **fat-tailed** — the 95% quantile sits at ~2.15σ, not the textbook 1.64σ — so
a Gaussian safety stock would under-provision. Cost rates are labelled
illustrative, not a guarantee.

![Forecast-uncertainty service-level curve — illustrative cost vs target service level, and expected unit fill rate](deliverables/service_level_curve.svg)

## Run it

```bash
pip install numpy scipy matplotlib openpyxl python-pptx torch pytest ruff

python data/generate_skus.py && python data/generate_history.py   # synthetic data
python -m revops --quiet          # headline plan + decision cards
python -m revops.report           # deliverables/ (json, xlsx, pdf, pptx, csv)
python powerbi/build_star.py      # powerbi/data/ star-schema CSVs
python web/build_data.py          # web/data.js  → open web/index.html offline
pytest -q                         # 21 tests, ~8s
```

Knobs: `--budget`, `--shelf-capacity-m3`, `--service-level`, `--promo-budget`,
`--price-guardrail`, `--json out.json`.

## What comes out

- **`deliverables/`** — an executive PDF/PPTX deck (waterfall, assortment
  before/after, inventory frontier, the forecast-uncertainty service-level curve,
  price-move distribution, promo allocation, a model-quality slide, recommended
  actions), a styled Excel workbook (with a `ServiceLevel` sheet), `actions.csv`
  (per-SKU reorder + reprice), and the service-level curve as
  `service_level_curve.csv` + a hand-drawn `service_level_curve.svg`.
- **`powerbi/`** — a star schema (`fact_prescription` + `dim_sku/category/date` +
  a scalar KPI table) with the KPIs written as real DAX, and a build spec for the
  three report pages. No tenant needed to produce or review it — that's stated
  honestly in [`powerbi/README.md`](powerbi/README.md).
- **`web/index.html`** — a dependency-free dashboard (hand-drawn SVG charts,
  light/dark, a promo what-if slider that re-solves the concave allocation in the
  browser). Opens straight off disk.

The [`docs/USE_CASE.md`](docs/USE_CASE.md) walks the whole quarterly decision as a
narrative.

## Honest limitations

- **The data is synthetic.** It is generated with real structure — seasonality, a
  risk-linked trend, a genuine price↔quantity relationship from each SKU's true
  elasticity — so the models have something to learn and recover, but it is not a
  real distributor's ledger.
- **The MILP-vs-greedy gap depends on the shelf constraint.** With only the
  capital budget, ranking by GMROI is already optimal and the gap collapses to
  near-zero. The headline six-figure advantage is real *because* a second (shelf)
  constraint binds — relax `--shelf-capacity-m3` and you can watch it shrink. I'd
  rather say that plainly than imply the MILP always wins.
- **The models are the smallest credible version of each** — a small global MLP
  forecaster, a from-scratch ridge elasticity, a from-scratch logistic decline
  classifier. The goal was a clear, honest pipeline, not a leaderboard.
- **The service-level curve's cost rates are illustrative.** Overage is one
  month's holding at each SKU's own rate; underage is the lost unit margin (times
  an optional goodwill/expedite penalty, default 1×). It is a single-period
  monthly model on synthetic data — the *recommended* service level and fill rate
  are what that model computes, not a guaranteed field outcome. The empirical
  prediction intervals, though, are a genuine, deterministic read of the
  forecaster's own error.

## A note on fit

I put this together while applying for a Data & AI Analytics working-student role
at Würth. It lines up with what that team does day to day: BI and dashboarding
(the Power BI star schema + DAX, and the offline web dashboard), predictive
analytics feeding real decisions (forecast/elasticity/decline → optimization),
and the Python/SQL-shaped data work around it, with Excel as a first-class
deliverable. The distributor framing isn't decoration — it's the setting these
techniques actually get used in.

---

Built and maintained by Dimitres Kisimov · © 2026 Dimitres Kisimov — all rights reserved; published for portfolio review. See LICENSE. · synthetic data, no
keys, runs offline.
