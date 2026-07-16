# Nairobi Real Estate Valuation Model 🏙️

Ever wondered how much that apartment in Kilimani is _actually_ worth?

This project builds a machine learning model that predicts Nairobi apartment
prices based on location, size, and other listing features — using real data
scraped from BuyRentKenya.

## What This Does

Scrapes listings → Cleans data → Analyzes patterns → Engineers features →
Models & evaluates → Serves predictions via a web app

## Tech Stack

Python · Pandas · Scikit-learn · Statsmodels · Playwright · Streamlit

## Progress

- [x] Phase 1: Scraped 349 real listings from BuyRentKenya
- [x] Phase 2: Cleaned to 332 usable Nairobi listings
- [x] Phase 3: Exploratory Data Analysis
- [x] Phase 4: Feature Engineering
- [x] Phase 5: Modeling & Evaluation
- [ ] Phase 6: Model serialization
- [ ] Phase 7: Streamlit app — predict any apartment's price

## Data Snapshot

332 apartments across Kilimani, Westlands, Kileleshwa, Lavington and more.
Price range: KSh 3M – 43M. Median: KSh 8M.

## Reason For The Project

Nairobi's property market is opaque. Buyers overpay. Sellers underprice.
This model brings data into a market that runs on gut feel.

---

## Phase 4: Feature Engineering

**Size signal.** `bedrooms` is kept as the primary size proxy (r=0.857 with
`log_price`). `bathrooms` is dropped — it's redundant with `bedrooms`
(r=0.904, a multicollinearity problem) and carries slightly weaker standalone
correlation with price (r=0.817 vs 0.857). A derived feature,
`bed_bath_ratio = bedrooms / bathrooms`, captures unit layout (bathroom
provision relative to bedroom count) without reintroducing the collinearity.

**Neighborhood tier.** Locations are grouped into `Mid` / `Premium` tiers,
one-hot encoded with `Mid` as the reference category. An earlier version used
ordinal encoding (Affordable=1, Mid=2, Premium=3), which was rejected: it
forces the model to assume equal price intervals between tiers, an assumption
the data doesn't support (Westlands median ~11M vs. the Mid cluster's ~7-9M
is not the same gap as Mid vs. the original Affordable median of ~4.3M).
One-hot encoding lets tier effects be estimated empirically instead of
assumed.

**Small-sample locations.** South B (n=2), Kitisuru (n=1), Parklands (n=4),
and Syokimau (n=5) are all folded into the `Mid` tier rather than treated as
standalone categories. These sample sizes are too small to support a
reliable, separately-estimated price effect; South B in particular produced
a degenerate 1-train/1-test split when initially kept as its own
"Affordable" tier. This consolidation is a disclosed limitation, not a hidden
one: price behavior specific to these four locations is not separately
modeled, and any prediction for a listing in one of them relies on the `Mid`
tier's average behavior rather than a location-specific estimate.

## Phase 5: Modeling & Evaluation

**Setup.** Target is `log_price` (raw price is right-skewed per Phase 3
EDA). Predictors: `bedrooms`, `bed_bath_ratio`, `tier_Premium`. Data split
80/20, stratified by tier, for the sklearn model comparison. The OLS baseline
is fit on the full dataset for interpretability.

### OLS Baseline

| Feature | Coefficient | p-value | 95% CI |
|---|---|---|---|
| const | 14.93 | <0.001 | [14.80, 15.05] |
| bedrooms | +0.449 | <0.001 | [0.421, 0.477] |
| bed_bath_ratio | −0.147 | 0.004 | [−0.245, −0.049] |
| tier_Premium | +0.232 | <0.001 | [0.159, 0.304] |

R² = 0.770, Adj. R² = 0.768. All variance inflation factors ≤ 1.03 —
multicollinearity is not a concern in the final feature set.

**Plain-English interpretation** (coefficients are on log-price, so each one
is approximately a percentage change in price):

- **Bedrooms**: each additional bedroom is associated with roughly a 45%
  increase in price, holding layout and tier constant. This is, by a wide
  margin, the dominant driver of price in this dataset.
- **Bed-bath ratio**: a higher ratio (more bedrooms per bathroom — i.e.
  fewer bathrooms relative to bedrooms) is associated with a ~15% price
  *decrease*. Units with proportionally more bathrooms per bedroom command
  a premium, holding bedroom count fixed.
- **Tier (Premium vs. Mid)**: Westlands listings carry roughly a 23% price
  premium over Mid-tier locations, controlling for size. This effect held up
  even after removing the small-sample locations from separate consideration
  (see below).

### The tier-significance reversal

An earlier version of this model, run before South B was folded into Mid,
showed **no significant tier effect** (`tier_Mid` p=0.915, `tier_Premium`
p=0.315 in a three-category encoding). This produced an incorrect early
conclusion that neighborhood tier was a confound of bedroom count rather
than an independent price driver.

That conclusion did not survive scrutiny. The three-category version
included South B (n=2), which produced a 1-train/1-test split — a signal
that the encoding itself was unstable. Folding South B, Kitisuru, Parklands,
and Syokimau (combined n=12 across sparse locations) into `Mid` removed that
instability, and `tier_Premium` became highly significant (p<0.001,
coefficient +0.23). **The original null result was an artifact of noise in
fragmented, small-n categories — not evidence that tier lacks a genuine
price effect.** This is treated as the most important methodological finding
of the modeling phase: category fragmentation can hide a real effect just as
easily as it can manufacture a false one, and the fix in both directions is
the same — check sample size before trusting a coefficient's significance.

### Model Comparison (held-out test set, n=67)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest | 0.779 | 0.261 | 0.194 |
| Lasso | 0.778 | 0.261 | 0.197 |
| Ridge | 0.778 | 0.262 | 0.200 |

All three converge to essentially the same performance once the small-n
tier categories are properly consolidated. Notably, an earlier run (before
consolidation) showed Random Forest outperforming the linear models by a
wider margin (R²=0.819 vs. ~0.778) — that gap closed once South B was folded
in, indicating RF's earlier edge was overfitting to noise in the sparse
categories rather than capturing genuine signal. Random Forest's feature
importances confirm the OLS story: `bedrooms` accounts for ~89% of
importance, with `bed_bath_ratio` and `tier_Premium` contributing modestly
and roughly equally (~5-6% each).

### Limitations

- **Residual non-normality.** The OLS residuals fail the Jarque-Bera
  normality test (p≈0), with elevated skew (1.87) and kurtosis (7.86). This
  indicates the model systematically misses a subset of listings — likely
  higher-end properties with quality/finish characteristics not captured by
  bedroom count or tier alone. Addressing this would require a feature the
  current dataset lacks (e.g., property age, finish quality, or amenity
  count), not a different modeling technique. This is the primary target for
  future feature engineering.
- **Mild residual autocorrelation** (Durbin-Watson = 1.33) is consistent
  with omitted grouping structure — listings within the same building or
  micro-location likely share unexplained variance the model doesn't
  capture.
- **Small-n locations are not separately modeled.** South B, Kitisuru,
  Parklands, and Syokimau (combined n=12) are folded into the `Mid` tier.
  Predictions for listings in these specific locations rely on the Mid-tier
  average, not a location-specific estimate.
