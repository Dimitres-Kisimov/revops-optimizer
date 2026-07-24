# Methodology — the predict → optimize handoff

The point of this project is not the models on their own and not the optimizers
on their own — it is the **handoff**. Three predictive models produce estimates;
those estimates replace the "known" inputs the optimizers would otherwise take on
faith. This document covers each model, its honest metric, and — the part that
matters — exactly which optimizer input it feeds.

```
   PREDICT (learned)                       OPTIMIZE (unchanged core)
   ─────────────────                       ─────────────────────────
   demand forecaster   ──forecast μ──►     newsvendor S*, assortment margin m_i
   elasticity ridge    ──ê per SKU───►     pricing Lerner markup P*
   decline classifier  ──risk r_i───►      demand haircut → assortment drops
```

---

## 1. Demand forecaster (global PyTorch MLP)

**What.** One global model across all SKUs (the M5/DeepAR lesson: a single model
on lag+calendar features beats fitting a tiny model per short series). Per
`(sku, month)` it engineers lags 1/2/3/12, a rolling mean(3), month Fourier terms
(k=1,2) and a per-SKU scale — everything scale-free (divided by the SKU's own
level) so one 10→32→16→1 ReLU network generalizes across a €0.9 fastener and a
€115 power tool. Trained ~40 epochs, Adam, MSE on `log1p` of scaled demand.

**Metric — MASE, rolling-origin.** Mean absolute scaled error vs an in-sample
seasonal-naive (`units[t−12]`) baseline: `MASE = model_MAE / naive_MAE`.
Below 1 means the model beats seasonal-naive. On this data it lands ~0.75 (vs a
naive ~1.01) — a real but modest edge, reported honestly, not cherry-picked.

**Feeds.** The next-month forecast `μ̂_sku` **replaces the raw historical
average** in a copy of the SKU records handed to (a) the newsvendor, so
order-up-to and safety stock track where demand is *going*, and (b) the
assortment MILP, so carry/drop margins are computed on forecast volume.

---

## 2. Elasticity estimator (from-scratch numpy ridge)

**What.** Constant-elasticity demand implies a log-log law
`ln(units) = a + b·ln(price) + controls`, where `b` is the price elasticity. SKUs
are pooled within a category (more price variation, one shared slope) and fit by
closed-form ridge `w = (XᵀX + λP)⁻¹Xᵀy`, coded directly in numpy. Controls: a
per-SKU fixed effect (so the slope is identified from *within*-SKU promo
variation, not cross-SKU level differences) and month Fourier terms for
seasonality. Only `ln(price)`'s coefficient is the elasticity.

**Metric — recovery error.** The synthetic history is generated with a known true
elasticity per SKU, so we can measure recovery: MAE of estimated-vs-true
elasticity, per category (~0.37) and per SKU. Signs come out negative
(down-sloping demand), which is the sanity gate.

**Feeds.** The estimated `ê` **replaces the true elasticity** in the pricing
optimizer's Lerner markup `P* = c·|ê|/(|ê|−1)`. This is deliberate: in production
you never know the true elasticity, so the price recommendation must stand on the
*estimate*. The guardrail band absorbs estimation error.

---

## 3. Decline-risk classifier (from-scratch numpy logistic regression)

**What.** A SKU is labelled "declining" when its 24-month demand slope,
normalized by level, is materially negative. An L2-penalized, class-weighted
logistic regression (sigmoid, binary cross-entropy, analytic gradient,
full-batch GD — all in numpy) predicts that label from four cheap features:
recent-vs-older demand ratio, normalized trend slope, coefficient of variation,
and margin %.

**Metric — ROC-AUC & PR-AUC, held-out.** From-scratch AUC (Mann-Whitney rank
identity) and average precision on a stratified 70/30 split — an honest
generalization estimate, not the training fit. ROC-AUC ~0.99 here because the
labels are strongly signalled by design; PR-AUC is reported alongside because the
positive class is the minority (prevalence ~0.3).

**Feeds.** The decline probability `r_i` applies a **demand haircut**
`μ_eff = μ̂·(1 − 0.5·r_i)` before the assortment MILP sees the SKU. Fragile ranges
therefore look less profitable to carry and are dropped first — the classifier
shapes the *range decision*, and the decision cards report how many dropped SKUs
were flagged declining.

---

## Why the handoff is the deliverable

Each optimizer is only as good as the numbers it is fed. A newsvendor with a
stale average over-stocks a fading SKU; a Lerner markup with a guessed elasticity
misprices; an assortment optimized on last year's volume carries dead ranges.
Wiring learned estimates into the constraints and objectives — and then reporting
**one € uplift number** the models are jointly responsible for — is the analytics
capability this project is meant to show. The metrics above are all honest and
held-out precisely so the uplift is credible rather than a fit to noise.
