# Power BI pack — RevOps Optimizer

This folder is a **Power BI Desktop showcase** built from the optimization
plan. It needs no Power BI tenant, licence, or gateway to *produce* — the CSVs
and the DAX are generated and written out here so the dimensional model can be
imported and reviewed. Honest framing: I don't ship a `.pbix` (that needs the
Desktop app to author), I ship the **star schema + DAX + a build spec** so the
modelling work is fully reproducible by anyone with Power BI Desktop.

## What's here

```
powerbi/
  build_star.py        generates the CSVs below from deliverables/prescription.json
  data/
    fact_prescription.csv   grain = SKU (the measures aggregate this)
    dim_sku.csv             SKU attributes
    dim_category.csv        category attributes + promo allocation
    dim_date.csv            a small calendar (24 history months + forecast month)
    kpi_headline.csv        disconnected 1-row table of plan-level scalars
    kpi_uplift_risk.csv     disconnected risk-band table (only after the
                            simulation is run — see below)
  DAX_measures.md      the KPIs as real, paste-ready DAX
  README.md            this file
```

`kpi_uplift_risk.csv` is written **only when** the Monte-Carlo deliverable
exists — run `python -m revops.simulate` before `build_star.py` to get it. It
holds the P10/P50/P90 band on the headline €, the downside VaR/CVaR and the
clear-probability; §10 of `DAX_measures.md` reads it. Treat it as a second
disconnected table (no relationship), like `kpi_headline`.

Regenerate any time after the plan changes:

```bash
python -m revops.report        # writes deliverables/prescription.json
python powerbi/build_star.py   # writes powerbi/data/*.csv
```

## The model (star schema)

```
                +----------------+
                |   dim_date     |
                | date_key (PK)  |
                +--------+-------+
                         | 1
                         | *
+-------------+   *   +--v-------------------+   *   +---------------+
|  dim_sku    |-------|  fact_prescription   |-------|  dim_category |
| sku (PK)    | 1   * |  sku (FK)            | *   1 | category (PK) |
+-------------+       |  category (FK)       |       +---------------+
                      |  date_key (FK)       |
                      |  forecast_demand ... |
                      +----------------------+

  kpi_headline  (disconnected — plan-level scalars, read by metric name)
```

## Import steps (Power BI Desktop)

1. **Get Data → Text/CSV** and load all five files from `powerbi/data/`.
   Accept the auto-detected types; set `date_key` to Whole Number and
   `dim_date[date]` to Date.
2. **Model view → create relationships** (all single-direction, one-to-many
   from the dim to the fact):
   - `dim_sku[sku]` 1 — * `fact_prescription[sku]`
   - `dim_category[category]` 1 — * `fact_prescription[category]`
   - `dim_date[date_key]` 1 — * `fact_prescription[date_key]`
   - leave `kpi_headline` disconnected (no relationship).
   - Mark `dim_date` as a date table (Table tools → Mark as date table →
     `dim_date[date]`) so time intelligence works.
3. **Add the measures** from `DAX_measures.md` (create an empty `_Measures`
   table to home them, then paste each measure).

## Report pages to build

### Page 1 — Executive Overview
- KPI cards: **Total Expected Uplift**, **SKUs Carried**, **Capital
  Utilization %**, **Forecast MASE**, **GMROI**.
- **Uplift waterfall** (Waterfall visual): category axis = a small
  "lever" field (Baseline / Pricing / Promo / Assortment), value = the matching
  measure; or use the `kpi_headline` rows directly.
- **Assortment donut**: `SKUs Carried` vs total by `fact_prescription[carried]`.
- **Uplift risk band** (if the simulation was run): an error-bar / range card
  showing **Uplift Median (P50)** with **Uplift P10 → P90** whiskers, plus
  cards for **Uplift VaR (10%)**, **Uplift CVaR (10%)** and
  **P(Uplift ≥ Headline)** — the honest "how much do I trust the number" panel.

### Page 2 — Assortment & Inventory
- **Matrix**: rows `dim_category[category]` → `dim_sku[sku]`, values
  `GMROI`, `annual_margin_eur`, `capital_at_cost_eur`, `SKUs Carried`.
- **Scatter**: x = `capital_at_cost_eur`, y = `annual_margin_eur`, legend =
  `carried` — shows the MILP picking the high-margin-per-cube corner.
- **Inventory table**: `sku`, `order_up_to`, `safety_stock`, `reorder_point`,
  `Service Fill Rate %`, sliced by category.

### Page 3 — Pricing & Promo
- **Histogram** (or column by binned `price_change_pct`): the price-move
  distribution within the guardrail band; card = `Avg Price Change %`.
- **Bar**: `promo_allocation_eur` by `dim_category[category]` — the concave
  allocation.
- **Card**: `Promo Incremental Margin` and its uplift vs an even split.

## Why this demonstrates Power BI ability without a tenant

Everything a reviewer needs to judge dimensional-modelling and DAX skill is
here in text: a normalized star (fact + conformed dimensions + a disconnected
scalar table), single-direction relationships, a marked date table, and
measures that use `CALCULATE`, `DIVIDE`, `SUMX`/`AVERAGEX`, `RELATED`, and a
`DATEADD` time-intelligence pattern. Loading the five CSVs and pasting the DAX
reproduces the whole model in a few minutes — no licence required.

Author: Dimitres Kisimov.
