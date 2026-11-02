# Nairobi Real Estate Valuation Model 🏙️

What does a Nairobi apartment typically _list_ for, given its location and size?

This project builds a machine learning model that estimates Nairobi apartment
asking prices from location, bedrooms, and bathroom layout, using listings
scraped from BuyRentKenya.

> **Read this first:** the model is trained on **asking prices from one listing
> portal**, not on sale prices. It estimates what comparable apartments are
> listed for, not what they sell for. It cannot tell you whether a listing is
> over- or under-priced relative to the transaction market.

**Live app:** https://nairobi-apartment-prices.streamlit.app/
(hosted on a free tier, so the first load after a period of inactivity can
take about 30 seconds while the app wakes up)

## What This Does

Scrapes listings → Cleans data → Analyzes patterns → Engineers features →
Models & evaluates → Serves predictions via a web app

## Tech Stack

Python · Pandas · Scikit-learn · Statsmodels · Playwright · Streamlit · Joblib

## Progress

- [x] Phase 1: Scraping (3,344 raw listings from a fixed, full-yield scraper)
- [x] Phase 2: Cleaned to 3,283 usable Nairobi listings
- [x] Phase 3: Exploratory Data Analysis
- [x] Phase 4: Feature Engineering
- [x] Phase 5: Modeling & Evaluation (cross-validated, model selected by a pre-committed rule)
- [x] Phase 6: Model serialization
- [x] Phase 7: Streamlit app (deployed, see link above)

## Data Snapshot

3,283 apartments across 21 Nairobi locations, dominated by Kilimani (820),
Kileleshwa (684), Lavington (618), and Westlands (1,108), with 17 additional
locations at smaller sample sizes. Price range (1st–99th percentile bounds
used for outlier trimming): roughly KSh 3M – 44.7M.

This is a substantial revision from an earlier 332-row version of this
dataset. The increase is not just "more scraping": it reflects a fixed
extraction pipeline (see Phase 1 notes below) that was silently discarding
~73% of listings on every page in every prior run of this project.

## Reason For The Project

Nairobi's property market has little public price data. This project turns
listing data into a transparent, testable estimate of typical asking prices,
and documents where that estimate can and cannot be trusted.

---

## Phase 1: Scraping: A Real Bug, Found Late

Every scrape run prior to this dataset, including the original 332-row
version this project shipped Phases 1–5 against, was silently dropping the
large majority of listings on every page. Logs consistently showed "Found
22–25 listing cards" but "Successfully extracted 6–7." The cause: the
scraper queried each card's data one at a time, with an `await` between
each, and this site (BuyRentKenya) is built on Livewire, which reactively
re-renders parts of the DOM. Element handles grabbed at the start of a page
load were going stale mid-loop as Livewire replaced nodes, silently failing
in the code's own error handling.

**Fix:** replaced the per-card Python query loop with a single atomic
`page.evaluate()` call that extracts every card's data inside the browser,
in one synchronous JavaScript pass, with no gap for the DOM to shift mid-
extraction. Yield went from ~27% to 100% (25/25 cards per page) on
verification.

The scraper was also switched from a Kenya-wide URL under the site's default
"Featured" (pay-for-placement) sort order to a Nairobi-scoped URL under
`sort=latest` (chronological), since Featured systematically overrepresents
a handful of paying agencies concentrated in a few neighborhoods.

**The site's own location tags are not reliable.** Syokimau (Machakos County)
appeared under the site's "Nairobi" tag and is excluded in cleaning after
manual verification; Kiambu Road was checked the same way and confirmed to be
inside Nairobi.

**Sparse locations are mostly a real feature of the data.** Part of the
earlier sparsity (Parklands, Kitisuru, South B) was a side effect of losing
three of every four listings. But several locations (Karen at n=1, for
example) are still tiny at 100% extraction yield, which suggests apartments
for sale in those areas are genuinely rare on this portal, not just
under-scraped.

## Phase 4: Feature Engineering

**Size signal.** `bedrooms` is kept as the primary size proxy. `bathrooms`
is dropped: it's redundant with `bedrooms` (r=0.91, a multicollinearity
problem) and carries slightly weaker standalone correlation with price
(r=0.78 vs 0.81 for log_price, in the full 3,283-row dataset). A derived
feature, `bed_bath_ratio = bedrooms / bathrooms`, captures unit layout
without reintroducing the collinearity.

**Neighborhood tier.** Locations are grouped into `Mid` / `Premium` tiers,
one-hot encoded with `Mid` as the reference category. Ordinal encoding was
tried and rejected early in this project: it forces the model to assume
equal price intervals between tiers, an assumption the data doesn't support.

**Tier assignment methodology.** With 21 total locations and highly uneven
sample sizes (from n=1,108 for Westlands down to n=1 for several outlying
neighborhoods), tier assignment used a tiered approach to evidence:

- **n≥4: assigned by real median price**, checked against the two reliable
  anchors Westlands (median 8.5M, n=1,108) and Kilimani (median 7.36M,
  n=820). Muthaiga (39.0M, n=5), Parklands (25.0M, n=9), General Mathenge
  (23.95M, n=4), Brookside (23.0M, n=9), Kiambu Road (14.45M, n=4), and
  Gigiri (13.5M, n=4) all sit well above both anchors and are assigned
  Premium.
- **Parklands and Brookside were corrected from an earlier Mid assignment.**
  In Phase 4's first pass (n=4 for Parklands at the time), the sample was
  too thin to trust a 25.2M median, so Parklands was folded into Mid as a
  conservative default. With n=9 now confirming the same magnitude, this
  was revisited and corrected to Premium. Small-n domain overrides are
  provisional in this project and get revisited once more data arrives (the
  same thing happened earlier with South B, and with the tier-significance
  finding below).
- **n≤3: defaulted to Mid, explicitly as a placeholder, not a finding.**
  Several of these single data points actively contradict common
  assumptions about the area, most notably Karen (median 6.5M, n=1), which
  sits _below_ both Westlands and Kilimani despite Karen's reputation as one
  of Nairobi's most expensive residential areas. The likely explanation is
  that Karen's housing stock is dominated by large standalone houses, so the
  one apartment listing found there isn't representative of "Karen" as a
  market. Mombasa Road (36.5M, n=2) and Eastleigh (28.5M, n=1) show the
  opposite pattern, almost certainly one or two atypical listings. None of
  these n≤3 locations should be read as data-backed tier assignments.

## Phase 5: Modeling & Evaluation

**Setup.** Target is `log_price` (raw price is right-skewed per Phase 3
EDA). Predictors: `bedrooms`, `bed_bath_ratio`, `tier_Premium`.

### Linear baseline (OLS, fit on all 3,283 rows)

This model is fit on the full dataset for interpretation. Its R² is
**in-sample**; the out-of-sample comparison is in the cross-validation
section below.

| Feature        | Coefficient | p-value | 95% CI           |
| -------------- | ----------- | ------- | ---------------- |
| const          | 14.960      | <0.001  | [14.917, 15.004] |
| bedrooms       | +0.441      | <0.001  | [0.431, 0.450]   |
| bed_bath_ratio | −0.165      | <0.001  | [−0.196, −0.135] |
| tier_Premium   | +0.326      | <0.001  | [0.304, 0.347]   |

R² = 0.734, Adj. R² = 0.733. All variance inflation factors ≤ 1.02, so
multicollinearity is not a concern.

**Interpretation.** The target is log-price, so a coefficient _b_ means a
price change of `exp(b) − 1`. The "coefficient ≈ percent" shortcut only
works for small coefficients, and it is off here for the larger ones.

- **Bedrooms** (+0.441): each additional bedroom is associated with about a
  **55%** higher price (`exp(0.441) − 1`), holding layout and tier constant.
  This is an average slope; the real per-bedroom jump shrinks as bedrooms
  increase (see below).
- **Bed-bath ratio** (−0.165): a higher ratio (proportionally fewer
  bathrooms per bedroom) is associated with about a **15%** lower price.
- **Tier (Premium vs. Mid)** (+0.326): Premium listings carry about a **39%**
  price premium over Mid-tier locations, controlling for size. This is higher
  than on the original 332-row dataset, consistent with the
  Parklands/Brookside correction moving high-priced listings into Premium.

### The tier-significance reversal (historical finding, still holds)

An early version of this model (3-category tier encoding, n=332) showed no
significant tier effect at all (p=0.92, p=0.32). That null result was later
shown to be an artifact of a single unstable small-n category (South B,
n=2) rather than evidence tier doesn't matter. Once consolidated, tier
became highly significant and has remained so, strengthening further, in
every subsequent, larger version of this dataset. The general lesson,
that category fragmentation can hide a real effect as easily as it can
manufacture a false one, held up across an order-of-magnitude increase in
sample size (332 → 3,283 rows).

### Model comparison: 5-fold cross-validation

An earlier single 80/20 split put Random Forest at R² 0.755 against ~0.705 for
Ridge/Lasso, but with the rarest tier categories at n=4–9, a few rows moving
between train and test could swing test R² by several points. That
comparison was **replaced by cross-validation**:

- 5-fold `StratifiedKFold` (stratified on `tier_Premium`, shuffled, seed 42)
- A scikit-learn `Pipeline` with a `ColumnTransformer` scales only
  `bedrooms` and `bed_bath_ratio` (not the binary tier dummy). The scaler is
  refit inside every fold on that fold's training portion only, so no
  validation-fold information leaks into the scaling.
- Ridge and Lasso use `RidgeCV` / `LassoCV` over alpha ∈ [0.0001, 100],
  selected on each fold's training data only.

| Model            | R² (mean ± std) | RMSE   | MAE    |
| ---------------- | --------------- | ------ | ------ |
| Random Forest    | 0.7751 ± 0.0224 | 0.2726 | 0.1901 |
| OLS              | 0.7325 ± 0.0196 | 0.2974 | 0.2247 |
| Ridge (CV-tuned) | 0.7325 ± 0.0196 | 0.2974 | 0.2247 |
| Lasso (CV-tuned) | 0.7325 ± 0.0196 | 0.2974 | 0.2247 |

Random Forest's RMSE is about 8% lower than the linear models'
(0.2726 vs 0.2974).

**Ridge and Lasso match OLS to four decimals, and that is a real result.**
With three features and variance inflation factors near 1.0, there is no
collinearity for Ridge to shrink and no redundant feature for Lasso to drop.
A first pass with hand-picked alphas showed the same equivalence, so the
alphas were tuned over a wide grid to rule out a weak-regularization
artifact. Regularization provides no benefit on this feature set.

### Why Random Forest beats the linear model: bedroom non-linearity

A linear model forces one constant bedroom slope. The data does not follow
one. Raw mean log-price jumps between consecutive bedroom counts (not
controlling for bed_bath_ratio):

| Tier    | 1→2 bedrooms | 2→3 bedrooms | 3→4 bedrooms |
| ------- | ------------ | ------------ | ------------ |
| Mid     | 0.63 (+87%)  | 0.39 (+48%)  | 0.34 (+41%)  |
| Premium | 0.49 (+63%)  | 0.46 (+59%)  | 0.22 (+24%)  |

The jump shrinks steadily in Mid. In Premium it is nearly flat from 1→2 to
2→3 and only drops at 3→4. (The five-bedroom cells have n=3 per tier and are
excluded as noise.)

Two checks against the alternative explanation that the effect is a
`bedrooms × tier` interaction:

- Adding a linear `bedrooms × tier_Premium` term to OLS raised in-sample R²
  only from 0.7337 to 0.7347. The term is statistically significant
  (coef −0.037, p=0.0003) but explains almost none of the gap to Random
  Forest.
- Cross-validated, a categorical OLS (defined in the next section) gains
  about +0.030 R² over the linear model (0.7628 vs 0.7325; the two figures
  come from different CV protocols, so treat the difference as approximate).
  That is roughly 70% of Random Forest's ~0.043 edge. Random Forest adds a
  further +0.013 over the categorical OLS. What it captures beyond bedroom
  non-linearity was not tested; non-linearity in `bed_bath_ratio` is the
  obvious candidate.

### Model selection: pre-committed decision rule

The strongest linear competitor is a "categorical OLS": bedrooms one-hot
encoded (capped at 4+), `bed_bath_ratio`, `tier_Premium`, and a bedroom × tier
interaction. Random Forest was compared against it with repeated
cross-validation (5 folds × 5 repeats = 25 splits, same splits for both
models). The decision rule was written down **before** running the
comparison:

> Select Random Forest only if the mean paired R² gain is at least 0.010
> **and** it beats categorical OLS on at least 20 of 25 splits. Otherwise
> select categorical OLS.

| Metric                             |        Result |
| ---------------------------------- | ------------: |
| Categorical OLS mean R²            |        0.7628 |
| Random Forest mean R²              |        0.7754 |
| Mean paired R² gain (RF − OLS)     |       +0.0126 |
| Std of paired differences          |        0.0062 |
| RF wins                            |         25/25 |
| Required mean gain / required wins |    0.010 / 20 |
| Selected model                     | Random Forest |

Both criteria were met, so **Random Forest** is the final model.

How to read this: the gain over the categorical OLS is consistent but small.
In an earlier single CV run against a categorical OLS without the interaction
term, RMSE was 0.2726 for Random Forest vs 0.2821 (about 3% lower). The 25
splits share training data, so "25/25" is a descriptive consistency check,
not a statistical significance test.

### Final model and prediction interval

The final Random Forest (300 trees, default depth, seed 42) is fit on all
3,283 rows. Out-of-fold R² is 0.7753, matching the cross-validated estimate.

Random Forest has no built-in prediction interval, so the app reports a
**90% range** from the 5th and 95th percentiles of out-of-fold residuals on
the log scale (residual = actual − predicted): −0.331 to +0.558, or roughly
−28% to +75% around the point estimate. The band is asymmetric: the model's
underpredictions are larger in magnitude than its overpredictions. It is
also **the same width for every input**.

### Limitations

- **Asking prices, one portal.** See the note at the top. Results do not
  transfer to sale prices or other sources.
- **Residual diagnostics describe the OLS model, not the shipped Random
  Forest.** OLS residuals are non-normal, though improved with more data and
  the tier correction: skew is 1.34 and kurtosis 5.13 on the full dataset
  (down from 1.87 and 7.86 on the original 332 rows), and the Jarque-Bera
  test still rejects normality (p≈0). This is attributed to a missing
  property-quality or finish feature the scraped fields cannot capture.
- **Mild OLS residual autocorrelation** (Durbin-Watson ≈ 1.14) is consistent
  with omitted grouping structure: listings in the same building or
  micro-location likely share unexplained variance.
- **Constant-width prediction interval.** Uncertainty is not modeled as a
  function of the inputs, so the band can be too narrow for listings the
  model sees less often.
- **Small-n tier placeholders.** Predictions for locations with n≤3 rely on
  the Mid-tier default, not a verified location-specific estimate. The app
  warns when a location has fewer than 10 training listings.
- **Sparse high-bedroom data.** There are six five-bedroom listings in total
  (three per tier). Random Forest does not extrapolate: any bedroom count
  above what it saw gets the same prediction as the largest it saw.
- **Cross-validation dependence.** CV splits share training observations,
  so paired comparisons are not independent experiments.

## Phase 6–7: Serialized Model and App

The fitted model, the tier map, per-location listing counts, and the residual
quantiles are saved together in `models/rf_model.joblib`, so the app cannot
drift from the training configuration.

The app takes a location, a bedroom count, and a bathroom count. It maps the
location to a tier, predicts, and reports a point estimate with the 90% range.

Input handling:

- **Bedrooms** are limited to 1, 2, 3, and 4+. "4+" is passed to the model as
  4, because larger units had too few listings to model.
- **Locations with fewer than 10 training listings** trigger a warning that
  the result is indicative only.
- A short note on the page states that results are asking-price estimates.
