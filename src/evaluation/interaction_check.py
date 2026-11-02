import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/processed/listings_features.csv")

# Model A: what you have now (no interaction)
base = smf.ols("log_price ~ bedrooms + bed_bath_ratio + tier_Premium", data=df).fit()

# Model B: same, plus the interaction
inter = smf.ols(
    "log_price ~ bedrooms + bed_bath_ratio + tier_Premium + bedrooms:tier_Premium",
    data=df,
).fit()

print(inter.summary())
print(f"\nR2 without interaction: {base.rsquared:.4f}")
print(f"R2 with interaction:    {inter.rsquared:.4f}")

# Does the interaction add real explanatory power?
f_stat, p_val, df_diff = inter.compare_f_test(base)
print(f"\nF-test (interaction vs base): F={f_stat:.2f}, p={p_val:.4g}")

# Step 2: raw slope of bedrooms within each tier
print("\nBedroom slope per tier (log_price ~ bedrooms):")
for tier_val, g in df.groupby("tier_Premium"):
    m = smf.ols("log_price ~ bedrooms", data=g).fit()
    label = "Premium" if tier_val == 1 else "Mid"
    print(f"  {label}: {m.params['bedrooms']:+.3f} (n={len(g)})")

# Model C: bedrooms as categories + interaction (a "lookup table" OLS)
cell = smf.ols(
    "log_price ~ C(bedrooms) * tier_Premium + bed_bath_ratio", data=df
).fit()
print(f"\nR2 base (linear):               {base.rsquared:.4f}")
print(f"R2 + interaction:               {inter.rsquared:.4f}")
print(f"R2 categorical bedrooms x tier: {cell.rsquared:.4f}")
print("RF CV R2 for reference:          0.7751")

print("\nMean log_price by bedrooms x tier:")
print(df.groupby(["tier_Premium", "bedrooms"])["log_price"].agg(["mean", "count"]))
