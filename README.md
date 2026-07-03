# Nairobi Real Estate Valuation Model 🏙️

Ever wondered how much that apartment in Kilimani is _actually_ worth?

This project builds a machine learning model that predicts Nairobi apartment
prices based on location, size, and other listing features — using real data
scraped from BuyRentKenya.

## What This Does

Scrapes listings → Cleans data → Analyzes patterns → Predicts prices →
Serves predictions via a web app

## Tech Stack

Python · Pandas · Scikit-learn · Statsmodels · Playwright · Streamlit

## Progress

- [x] Phase 1: Scraped 349 real listings from BuyRentKenya
- [x] Phase 2: Cleaned to 332 usable Nairobi listings
- [x] Phase 3: Exploratory Data Analysis
- [ ] Phase 4: Feature Engineering
- [ ] Phase 5: Modeling (OLS + Random Forest)
- [ ] Phase 6: Streamlit app — predict any apartment's price

## Data Snapshot

332 apartments across Kilimani, Westlands, Kileleshwa, Lavington and more.
Price range: KSh 3M – 43M. Median: KSh 8M.

## Reason For The Project

Nairobi's property market is opaque. Buyers overpay. Sellers underprice.
This model brings data into a market that runs on gut feel.
