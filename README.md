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

- [x] Phase 1: Scraping (3,344 raw listings from a fixed, full-yield scraper)
- [x] Phase 2: Cleaned to 3,283 usable Nairobi listings
- [x] Phase 3: Exploratory Data Analysis
- [x] Phase 4: Feature Engineering
- [x] Phase 5: Modeling & Evaluation
- [ ] Phase 6: Model serialization
- [ ] Phase 7: Streamlit app — predict any apartment's price

## Data Snapshot

3,283 apartments across 21 Nairobi locations, dominated by Kilimani (820),
Kileleshwa (684), Lavington (618), and Westlands (1,108), with 17 additional
locations at smaller sample sizes. Price range (1st–99th percentile bounds
used for outlier trimming): roughly KSh 3M – 44.7M.

This is a substantial revision from an earlier 332-row version of this
dataset. The increase is not just "more scraping" — it reflects a fixed
extraction pipeline (see Phase 1 notes below) that was silently discarding
~73% of listings on every page in every prior run of this project.

## Reason For The Project

Nairobi's property market is opaque. Buyers overpay. Sellers underprice.
This model brings data into a market that runs on gut feel.

---

## Phase 1: Scraping — A Real Bug, Found Late

Every scrape run prior to this dataset — including the original 332-row
version this project shipped Phases 1–5 against — was silently dropping the
large majority of listings on every page. Logs consistently showed "Found
22–25 listing cards" but "Successfully extracted 6–7." The cause: the
scraper queried each card's data one at a time, with an `await` between
each — and this site (BuyRentKenya) is built on Livewire, which reactively
re-renders parts of the DOM. Element handles grabbed at the start of a page
load were going stale mid-loop as Livewire replaced nodes, silently failing
in the code's own error handling.

**Fix:** replaced the per-card Python query loop with a single atomic
`page.evaluate()` call that extracts every card's data inside the browser,
in one synchronous JavaScript pass, with no gap for the DOM to shift mid-
extraction. Yield went from ~27% to 100% (25/25 cards per page) on
verification.

This also incidentally fixed part of the earlier "sparse location" problem
that Phases 3–4 spent significant effort working around (folding Parklands,
Kitisuru, South B, Syokimau into a Mid tier due to tiny sample sizes): a
meaningful fraction of that sparsity was an artifact of losing 3 out of
every 4 listings, not a true reflection of how many listings those areas
actually have on the site.

The scraper was also switched from a Kenya-wide URL under the site's default
"Featured" (pay-for-placement) sort order to a Nairobi-scoped URL under
`sort=latest` (chronological), since Featured systematically overrepresents
a handful of paying agencies concentrated in a few neighborhoods.

## Phase 4: Feature Engineering

**Size signal.** `bedrooms` is kept as the primary size proxy. `bathrooms`
is dropped — it's redundant with `bedrooms` (r=0.91, a multicollinearity
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
- **Parklands and Brookside are corrected from an earlier Mid assignment.**
  In Phase 4's first pass (n=4 for Parklands at the time), the sample was
  too thin to trust a 25.2M median, so Parklands was folded into Mid as a
  conservative default. With n=9 now confirming the same magnitude, this
  was revisited and corrected to Premium. This is treated as a deliberate,
  disclosed methodological pattern in this project, not a one-off fix:
  small-n domain overrides are provisional, and get revisited once more
  data arrives (the same thing happened earlier with South B, and with the
  tier-significance finding below).
- **n≤3: defaulted to Mid, explicitly as a placeholder, not a finding.**
  Several of these single data points actively contradict common
  assumptions about the area — most notably Karen (median 6.5M, n=1),
  which sits *below* both Westlands and Kilimani despite Karen's general
  reputation as one of Nairobi's most expensive residential areas. The
  likely explanation is that Karen's housing stock is dominated by large
  standalone houses on big plots, not apartments, so the one apartment
  listing found there isn't representative of "Karen" as a market. Mombasa
  Road (36.5M, n=2) and Eastleigh (28.5M, n=1) show the opposite pattern —
  medians far above what their commercial/lower-income reputation would
  suggest, almost certainly one or two atypical listings rather than a real
  market signal. None of these n≤3 locations should be read as data-backed
  tier assignments; they are the conservative default in the absence of
  enough data to say anything with confidence, and this is a disclosed
  limitation, not a hidden one.

## Phase 5: Modeling & Evaluation

**Setup.** Target is `log_price` (raw price is right-skewed per Phase 3
EDA). Predictors: `bedrooms`, `bed_bath_ratio`, `tier_Premium`. Data split
80/20, stratified by tier, for the sklearn model comparison. The OLS
baseline is fit on the full 3,283-row dataset for interpretability.

### OLS Baseline

| Feature | Coefficient | p-value | 95% CI |
|---|---|---|---|
| const | 14.960 | <0.001 | [14.917, 15.004] |
| bedrooms | +0.441 | <0.001 | [0.431, 0.450] |
| bed_bath_ratio | −0.165 | <0.001 | [−0.196, −0.135] |
| tier_Premium | +0.326 | <0.001 | [0.304, 0.347] |

R² = 0.734, Adj. R² = 0.733. All variance inflation factors ≤ 1.02 —
multicollinearity is not a concern.

**Plain-English interpretation** (coefficients are on log-price, so each is
approximately a percentage change in price):

- **Bedrooms**: each additional bedroom is associated with roughly a 44%
  increase in price, holding layout and tier constant — the dominant price
  driver by a wide margin, consistent across every version of this dataset.
- **Bed-bath ratio**: a higher ratio (proportionally fewer bathrooms per
  bedroom) is associated with a ~15–17% price decrease.
- **Tier (Premium vs. Mid)**: Premium-tier listings carry roughly a 33–39%
  price premium over Mid-tier locations, controlling for size. This is
  higher than earlier versions of this model estimated (23% on the original
  332-row dataset), consistent with the Parklands/Brookside correction
  moving genuinely high-priced listings into the Premium category.

### The tier-significance reversal (historical finding, still holds)

An early version of this model (3-category tier encoding, n=332) showed no
significant tier effect at all (p=0.92, p=0.32). That null result was later
shown to be an artifact of a single unstable small-n category (South B,
n=2) rather than evidence tier doesn't matter — once consolidated, tier
became highly significant and has remained so, strengthening further, in
every subsequent, larger version of this dataset. The general lesson —
category fragmentation can hide a real effect as easily as it can
manufacture a false one — held up across an order-of-magnitude increase in
sample size (332 → 3,283 rows) and is the most durable methodological
finding of this project so far.

### Model Comparison (held-out test set, n=657)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest | 0.755 | 0.277 | 0.191 |
| Lasso | 0.706 | 0.303 | 0.226 |
| Ridge | 0.705 | 0.303 | 0.229 |

**Caveat on comparing this table across dataset versions:** the train/test
split is stratified by `tier`, and correcting Parklands/Brookside's tier
labels changes which rows land in train vs. test even with a fixed random
seed. With the rarest categories at n=4–9, a handful of rows shifting
position between train and test can move a 657-row test R² by several
points on its own. The full-data OLS numbers above are the more stable
comparison point across dataset versions; the sklearn table should be
treated as somewhat noisy at this sample size for the smaller tier
categories, not as strong evidence that Random Forest's relative advantage
changed between dataset versions.

### Limitations

- **Residual non-normality**, though improved with more data and the tier
  correction. Skew is 1.34 and kurtosis 5.13 on the full 3,283-row dataset
  (down from 1.87 and 7.86 on the original 332-row dataset), and the
  Jarque-Bera test still rejects normality (p≈0). This is attributed to a
  missing property-quality/finish feature the current dataset has no way to
  capture — not a modeling technique problem.
- **Mild residual autocorrelation** (Durbin-Watson ≈ 1.14) is consistent
  with omitted grouping structure — listings within the same building or
  micro-location likely share unexplained variance.
- **Several small-n tier assignments (n≤3) are placeholders, not findings**
  — see the Phase 4 methodology note above. Predictions for listings in
  these specific locations rely on the Mid-tier average rather than a
  verified location-specific estimate.
