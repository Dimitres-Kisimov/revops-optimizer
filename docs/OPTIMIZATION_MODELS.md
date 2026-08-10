# Optimization models

Four decisions, four optimization formulations. Each is a real program solved
with SciPy (HiGHS for the MILP, SLSQP for the promo), not a heuristic dressed up
as one. This document writes out the maths and, at the end, the honest cases
where a greedy baseline ties the optimizer.

---

## 1. Assortment — a 0/1 knapsack MILP (two constraints)

**Decision.** Which SKUs to carry, to maximize annual gross margin under a
working-capital budget *and* a shelf-space budget, keeping a minimum breadth per
category.

Let `x_i ∈ {0,1}` be "carry SKU *i*". With margin `m_i` (annual €), capital
`w_i` (average inventory value at cost), shelf cube `s_i`, categories `C`:

```
maximize    Σ_i  m_i · x_i
subject to  Σ_i  w_i · x_i  ≤  B        (working-capital budget)
            Σ_i  s_i · x_i  ≤  S        (shelf-capacity budget, m³)
            Σ_{i∈c} x_i     ≥  k_c   ∀c (minimum breadth per category)
            x_i ∈ {0,1}
```

This is a **multidimensional 0/1 knapsack** — NP-hard in general, solved to
optimality here by branch-and-bound in HiGHS via `scipy.optimize.milp`. HiGHS
minimizes, so we pass `c = -m`. The category-breadth rows make it more than a
plain knapsack: you cannot empty a category to free capital.

`w_i` and `s_i` both come from the same physical stock estimate — a cycle plus a
1.65σ safety buffer:

```
q_i = avg_demand_i · max(0.5, lead_time_i)/2  +  1.65 · demand_std_i
w_i = q_i · unit_cost_i           s_i = q_i · shelf_m3_per_unit_i
```

**Why two constraints matter.** With a single capital constraint, ranking by
GMROI (`margin / capital`) and filling the budget is *optimal* — the greedy tie
below. Add the shelf constraint and the problem becomes genuinely
two-dimensional: a SKU cheap in capital can be bulky on the shelf, so no single
ratio orders the items correctly. That is exactly where the MILP earns its keep
(a five-/six-figure €/yr gap over greedy in this dataset).

---

## 2. Inventory — the newsvendor critical fractile + safety stock

**Decision.** For each carried SKU, the order-up-to level `S` that balances the
cost of a stockout against the cost of overstock.

One period, demand `D ~ Normal(μ, σ)`. Underage cost `c_u` = lost unit margin;
overage cost `c_o` = one period's holding cost on an unsold unit. Expected cost
`C(S) = c_o·E[(S−D)⁺] + c_u·E[(D−S)⁺]`. Differentiating and setting `C'(S)=0`:

```
P(D ≤ S*) = c_u / (c_u + c_o)   ≡   the critical fractile  CF
S*        = μ + z_CF · σ,        z_CF = Φ⁻¹(CF)
```

Intuition: stock up to the demand quantile where the marginal expected cost of
one more unit flips sign. High-margin, cheap-to-hold SKUs get `CF → 1` (hold
plenty); thin-margin, expensive-to-hold SKUs get `CF → 0.5` and lean stock.

On top of the single-period optimum we set a **service-level safety stock**
covering demand *and* lead-time variability:

```
ss  = z_SL · sqrt( L·σ²  +  μ²·σ_L² )        z_SL = Φ⁻¹(service_level)
ROP = μ·L + ss
```

and report expected units short per period,
`E[(D−S)⁺] = σ·(φ(k) − k·(1−Φ(k)))` with `k = (S−μ)/σ`, which drives the
OTIF-style fill-rate KPI. A service-vs-cost frontier sweeps the service level so
the planner sees the trade-off, not just one point.

### 2a. Forecast uncertainty → the service-level (fill-rate) curve

The safety-stock formula above needs a demand-variability `σ`. The honest input
is not the raw historical spread — it is how wrong the *forecast* tends to be. So
`revops/optimize/service_level.py` takes the forecaster's own one-step residuals
`e = actual − forecast`, standardizes them per SKU (`r = e/σ̂_sku`) and pools them
into one empirical error distribution. Its quantiles replace `Φ⁻¹`:

```
z_emp(SL) = empirical SL-quantile of the pooled standardized residuals
ss_sku(SL) = z_emp(SL) · σ̂_sku                    (no normality assumed)
E[units short]_sku(SL) = σ̂_sku · mean_i( max(0, r_i − z_emp(SL)) )   (empirical)
fill_rate(SL) = 1 − Σ_sku E[units short]_sku / Σ_sku μ̂_sku
```

Sweeping `SL` from 80% → 99% gives, at each level, the required safety stock, its
holding cost, the expected unit fill rate and the expected stockout cost, and we
recommend the level that **minimizes total illustrative cost**:

```
overage  (holding)  = ss_sku · unit_cost · (annual_holding_rate / 12)     per month
underage (stockout) = E[units short]_sku · unit_margin · penalty_mult      per month
recommended SL = argmin_SL ( Σ holding + Σ stockout )
```

**Cost rates are illustrative, labelled, not a guarantee** — a single-period
(monthly) framing on synthetic data. `penalty_mult` (default 1.0 = pure lost
margin) is an optional goodwill/expedite multiplier; raising it moves the
recommendation up. On the seeded carried assortment the curve recommends a **96%
target**, which buys a **~99.3% expected unit fill rate** — a cycle service level
of 96% still meets ~99% of *demand* because most stockout cycles miss by only a
little, a distinction a single "service level" number hides.

The honest headline: the pooled residuals are **fat-tailed** (95% quantile at
~2.15σ vs the Gaussian 1.64σ), so a normal-curve safety stock would silently
under-provision at high service levels. Deliverables:
`deliverables/service_level_curve.csv` and a hand-drawn
`deliverables/service_level_curve.svg`.

---

## 3. Pricing — constant-elasticity profit max (Lerner) with a guardrail

**Decision.** A recommended price per SKU that maximizes unit-margin × volume,
clamped to a commercial band.

Constant-elasticity demand `Q(P) = Q₀·(P/P₀)^e`, `e < 0`. Profit
`π(P) = (P − c)·Q(P)`. Setting `dπ/dP = 0` gives the **Lerner** optimum:

```
(P* − c)/P* = −1/e      ⇒      P* = c · |e| / (|e| − 1)     (finite only if |e| > 1)
```

The markup is `|e|/(|e|−1)`: the *less* elastic the SKU (|e| near 1), the fatter
the optimal markup; the more elastic, the closer price sits to cost. Two guards
make it deployable:

- **Inelastic SKUs (|e| ≤ 1)** have no interior optimum (profit rises without
  bound in the model), so they are flagged and pushed to the top of the
  guardrail band rather than "priced to infinity".
- **The guardrail** clamps every recommendation to `[P₀(1−g), P₀(1+g)]`
  (default `g = 15%`), so the model never proposes a commercially absurd jump.

The elasticity `e` fed here is the *estimated* one from the predictive layer,
not the true generating value — the whole point of predict→optimize.

---

## 4. Promotion — concave budget allocation (KKT / water-filling)

**Decision.** Split a promo budget across categories to maximize incremental
margin under diminishing returns.

Each category's incremental margin from spend `x_c` is concave and saturating:

```
u_c(x_c) = a_c · (1 − e^{−b_c x_c})           a_c, b_c > 0
```

`a_c` (saturation) scales with the category's margin headroom; `b_c`
(responsiveness) with its average elasticity magnitude. The program:

```
maximize   Σ_c a_c (1 − e^{−b_c x_c})
subject to Σ_c x_c ≤ B,   x_c ≥ 0
```

is smooth and concave, so the KKT conditions are necessary *and* sufficient.
Stationarity gives, for every category funded at the interior:

```
a_c b_c e^{−b_c x_c} = λ    (a common shadow price λ on the budget)
⇒  x_c = max( 0,  (1/b_c) · ln(a_c b_c / λ) )
```

This is **water-filling**: pour budget where the marginal return `a_c b_c
e^{−b_c x_c}` is highest; as a category fills, its marginal return falls until it
meets the common level `λ`; the budget binds when `Σ x_c = B`, which pins `λ`.
The solver of record is `scipy.optimize.minimize(method="SLSQP")` with the
analytic gradient; the water-filling identity is the cross-check (and the exact
formula the web dashboard's what-if slider re-solves live via a bisection on
`λ`).

---

## Honest notes — when greedy ties the MILP

- **Assortment with only the capital constraint.** A single linear budget makes
  the LP relaxation integral at all but one "fractional" item, so GMROI-ranked
  greedy is optimal up to that last item. The MILP's advantage appears *only*
  once the second (shelf) constraint binds — this is why the default ships with
  a shelf budget, and why the reported MILP-vs-greedy gap collapses to near-zero
  if you relax it (try `--shelf-capacity-m3 500`).
- **Pricing** has a closed form; there is no "optimizer vs heuristic" story —
  the value is the elasticity *estimate* and the guardrail discipline.
- **Promo** is concave, so gradient ascent and water-filling agree to solver
  tolerance; "beats an even split" is the honest, modest claim (a few % of the
  budget's incremental margin), not a headline miracle.

The models are the **smallest credible version** of each — real formulations,
solved exactly, on synthetic-but-structured data. See `METHODOLOGY.md` for how
the AI models feed them and `USE_CASE.md` for the end-to-end narrative.

---

## Robustness — scenarios and a one-way sensitivity tornado

A prescriptive plan collapses a lot of uncertainty into a single €. `revops/scenario.py`
puts that number under stress without touching the optimizers: the predictive
layer is trained **once**, and each *scenario* perturbs its outputs before
re-solving the same MILP / pricing / promo programs (reusing `prescribe()`'s own
`_adjusted_skus` adapter, so the *Baseline* scenario reproduces the headline plan
exactly).

A scenario is four prediction multipliers `(demand, cost, |elasticity|, decline
risk)` and the four decision knobs `(capital budget, shelf capacity, promo budget,
price guardrail)`. Two views are produced:

* **Scenario library** — named business cases (demand downturn, cost inflation,
  tight capital, promo push, a more-elastic market), each compared to baseline on
  total uplift and its three components.
* **One-way sensitivity (tornado)** — each driver is swept across a plausible
  planning band `[a, b]` while everything else is held at baseline, and the swing
  in total uplift, `Δ = u(high) − u(low)`, is ranked widest-first. The widest bar
  is the assumption the € is most exposed to. Because the sweep is one-at-a-time it
  reads *local* sensitivity, not interactions; the named scenarios cover the
  combined moves.

The bands are **illustrative planning ranges, not a forecast** (absolute knobs
±25%; demand ±15%, cost ±8%, elasticity ±20%, risk ±30%), and each figure is what
the model computes under that assumption — not a guarantee. The whole thing is
deterministic: no RNG, no wall-clock, byte-identical CSV/SVG across re-runs.

### Joint Monte-Carlo — the interactions the tornado cannot see

The tornado is one-at-a-time, so it reads *local* sensitivity and **not
interactions**. `revops/simulate.py` closes exactly that gap: the four predictive
uncertainties `(demand, unit cost, |elasticity|, decline risk)` are drawn
**together** from three-point (triangular, mode = 1.0) planning distributions —
the same bands the tornado uses — and each joint draw is pushed back through the
*identical* `scenario.evaluate` core (re-solving the MILP, pricing and promo).
The decision *levers* are held at baseline, because they are choices, not
uncertainties; every draw is re-optimized, so the output is the range of
*achievable* uplift across plausible conditions.

The headline € then becomes a distribution rather than a point:

```
P10 .. P90 confidence band          the middle 80% of achievable uplift
median (P50)                        the central outcome
VaR(10%)  = P10                     a 90%-confident downside floor
CVaR(10%) = mean of the worst 10%   expected shortfall (a tail-risk read)
P(uplift >= headline point est.)    chance the outcome clears the point estimate
```

Sampling is a **Latin-Hypercube** design under a **fixed seed**, so the marginals
are smooth at a few hundred draws and the CSV/SVG are byte-identical across
re-runs — deterministic despite being a simulation. On the seeded 240-SKU set the
headline €159,966 point estimate sits *above* the modelled median (~€158.5k) with
only a ~43% chance of being cleared once the four uncertainties move jointly — a
mildly optimistic, slightly left-skewed profile. That is the honest, interaction-
aware read a single point estimate (or a one-way tornado) hides. Deliverables:
`deliverables/uplift_simulation.csv` (a tall, DAX-ready summary) and a hand-drawn
`deliverables/uplift_distribution.svg`. Figures are **modelled on synthetic data
— illustrative planning ranges, not a forecast or a guarantee.**

### From a risk band to a decision gate — per-move robustness

The Monte-Carlo band prices the *total*; a planner executes the plan one price
move at a time. `revops/robustness.py` turns the band into an **accept/hold
gate** on each recommended price move by replaying the *same* fixed-seed
Latin-Hypercube draws (same sampler, same seed, same triangular bands — a test
pins the per-draw totals to `simulate`'s, so the gate decomposes the *published*
band, not a parallel model) and re-solving the identical `scenario.solve_plan`
core per draw. Per baseline move (a recommendation with |Δprice| ≥ 1%), across
the draws:

```
carry_rate            share of draws the assortment MILP still carries the SKU
direction_agreement   share of draws the pricing optimizer still recommends the
                      same move (same direction, still ≥ the 1% threshold)
uplift P10/P50/P90    annual € of executing the PUBLISHED price under each
                      draw's (cost, demand, elasticity) — €0 when delisted
```

The uplift is deliberately the **frozen action's** €, not the re-optimized one:
a re-optimized move is non-negative whenever the SKU is carried, which would
make any downside test vacuous. Freezing the published price and letting the
world move is what actually happens when a price list ships. The gate itself is
stated in full and tested:

```
ACCEPT  iff  carry_rate ≥ 90%  and  direction_agreement ≥ 80%  and  P10 > 0
HOLD    otherwise, with one named reason, checked in order:
        delist-risk → direction-flips → downside
```

The accepted moves are also summed **per draw** into an "accepted book", whose
P10/P50/P90 is a true joint portfolio band (per-move P10s are not additive).
With degenerate (zero-width) bands every move collapses to accept with a point
distribution at its baseline € — the collapse-to-base-case gate. Deliverables:
`deliverables/price_move_robustness.csv` (SKU grain — also lifted into the
Power BI star as `fact_price_robustness`, §11 of the DAX pack) and a hand-drawn
`deliverables/price_move_robustness.svg`. Verdicts are **modelled on synthetic
data under illustrative planning ranges — a screening discipline, not a
guarantee** that an accepted move will earn its € (or that a held one would not
have).
