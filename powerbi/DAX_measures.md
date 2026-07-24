# DAX measures — RevOps Optimizer

Paste these into Power BI Desktop (right-click the `fact_prescription` table →
**New measure**) after importing the star schema from `powerbi/data/`. They are
written against the model in `README.md`: a `fact_prescription` grain-of-SKU
fact related to `dim_sku`, `dim_category`, `dim_date`, plus a disconnected
one-row `kpi_headline` table for plan-level scalars.

I group them into a dedicated `_Measures` table (Home → Enter Data → empty table
named `_Measures`) so they are easy to find, but any home table works.

---

## Helper: pull a scalar from `kpi_headline`

`kpi_headline` is a tall (metric, value) table. This helper returns one metric,
so every headline measure below is a one-liner.

```DAX
KPI Value =
VAR _m = SELECTEDVALUE ( _Measures[__unused], "" )  -- placeholder; see note
RETURN _m
```

In practice I read a named metric with `CALCULATE` + a filter on the text key:

```DAX
_KPI ( _metric ) :=  -- conceptual; DAX has no user funcs, so inline per measure
CALCULATE ( SUM ( kpi_headline[value] ), kpi_headline[metric] = _metric )
```

Each headline measure below inlines that pattern.

---

## 1. Total Expected Uplift (EUR / year)

The plan's headline: pricing + promo + assortment MILP-vs-greedy.

```DAX
Total Expected Uplift =
CALCULATE (
    SUM ( kpi_headline[value] ),
    kpi_headline[metric] = "total_expected_uplift_eur"
)
```

Or reconstructed additively (proves the components tie out):

```DAX
Total Expected Uplift (built) =
[Pricing Uplift] + [Promo Incremental Margin] + [Assortment MILP vs Greedy]
```

```DAX
Pricing Uplift =
SUM ( fact_prescription[margin_uplift_annual_eur] )
```

```DAX
Promo Incremental Margin =
CALCULATE (
    SUM ( kpi_headline[value] ),
    kpi_headline[metric] = "promo_incremental_eur"
)
```

```DAX
Assortment MILP vs Greedy =
CALCULATE (
    SUM ( kpi_headline[value] ),
    kpi_headline[metric] = "assortment_milp_vs_greedy_eur"
)
```

## 2. Gross Margin %

Unit margin over price, demand-weighted across the carried range.

```DAX
Gross Margin % =
VAR _rev =
    SUMX (
        FILTER ( fact_prescription, fact_prescription[carried] = 1 ),
        fact_prescription[forecast_demand] * fact_prescription[current_price]
    )
VAR _cogs =
    SUMX (
        FILTER ( fact_prescription, fact_prescription[carried] = 1 ),
        fact_prescription[forecast_demand]
            * RELATED ( dim_sku[unit_cost_eur] )
    )
RETURN
    DIVIDE ( _rev - _cogs, _rev )
```

## 3. GMROI (Gross-Margin Return On Inventory Investment)

Annual gross margin divided by average inventory at cost.

```DAX
GMROI =
DIVIDE (
    SUM ( fact_prescription[annual_margin_eur] ),
    SUM ( fact_prescription[capital_at_cost_eur] )
)
```

## 4. Forecast MASE

Held-out mean absolute scaled error vs the seasonal-naive baseline (< 1 = the
model beats naive).

```DAX
Forecast MASE =
CALCULATE (
    SUM ( kpi_headline[value] ),
    kpi_headline[metric] = "forecast_mase"
)
```

```DAX
Forecast Beats Naive =
VAR _mase = [Forecast MASE]
VAR _naive =
    CALCULATE ( SUM ( kpi_headline[value] ),
        kpi_headline[metric] = "seasonal_naive_mase" )
RETURN
    IF ( _mase < _naive, "Yes", "No" )
```

## 5. SKUs Carried

```DAX
SKUs Carried =
CALCULATE (
    COUNTROWS ( fact_prescription ),
    fact_prescription[carried] = 1
)
```

```DAX
Carry Rate % =
DIVIDE ( [SKUs Carried], COUNTROWS ( fact_prescription ) )
```

## 6. Capital Utilization %

Working capital consumed by the carried range against the assortment budget.

```DAX
Capital Utilization % =
VAR _budget =
    CALCULATE ( SUM ( kpi_headline[value] ),
        kpi_headline[metric] = "capital_budget_eur" )
VAR _used =
    CALCULATE (
        SUM ( fact_prescription[capital_at_cost_eur] ),
        fact_prescription[carried] = 1
    )
RETURN
    DIVIDE ( _used, _budget )
```

## 7. Avg Price Change %

Mean recommended price move across repriced (carried) SKUs.

```DAX
Avg Price Change % =
AVERAGEX (
    FILTER ( fact_prescription, fact_prescription[carried] = 1 ),
    fact_prescription[price_change_pct]
)
```

## 8. Service Level (OTIF-style fill rate)

Expected fill rate = 1 − expected shortage / forecast demand, weighted by
demand. Approximates On-Time-In-Full off the newsvendor expected shortage.

```DAX
Service Fill Rate % =
VAR _dem =
    CALCULATE ( SUM ( fact_prescription[forecast_demand] ),
        fact_prescription[carried] = 1 )
VAR _short =
    CALCULATE ( SUM ( fact_prescription[expected_shortage_units] ),
        fact_prescription[carried] = 1 )
RETURN
    1 - DIVIDE ( _short, _dem )
```

## 9. YoY placeholder

The plan is a single as-of snapshot, so there is no prior-year actual to divide
by yet. This is the shape the measure will take once a second period lands in
`dim_date` — wired to `dim_date` via a standard time-intelligence pattern.

```DAX
Uplift YoY % =
VAR _cur = [Total Expected Uplift]
VAR _prior =
    CALCULATE ( [Total Expected Uplift],
        DATEADD ( dim_date[date], -1, YEAR ) )
RETURN
    DIVIDE ( _cur - _prior, _prior )   -- BLANK until a prior year exists
```

---

### Notes on correctness

- `DIVIDE` is used everywhere instead of `/` to get safe BLANK handling on a
  zero denominator.
- `RELATED` reaches from the fact into `dim_sku` for `unit_cost_eur`; this
  requires the single-direction `dim_sku[sku] 1 --- * fact_prescription[sku]`
  relationship from the README.
- Filtering `fact_prescription[carried] = 1` keeps margin/service measures on
  the *decided* range; drop the filter to see the whole catalogue.
- `kpi_headline` is intentionally disconnected (no relationship) — it holds
  plan-level scalars that have no SKU grain, read by an explicit
  `metric = "..."` filter.
