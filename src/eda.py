import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/listings_clean.csv")

# 1. Price distribution — raw vs log
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["price"].plot(kind="hist", bins=30, ax=axes[0], title="Price (raw)")
df["log_price"].plot(kind="hist", bins=30, ax=axes[1], title="Price (log)")
plt.tight_layout()
plt.savefig("data/processed/price_distribution.png")
plt.close()

# 2. Median price by location
df.groupby("location")["price"].median().sort_values().plot(
    kind="barh", figsize=(8, 5), title="Median Price by Location"
)
plt.tight_layout()
plt.savefig("data/processed/median_price_by_location.png")
plt.close()

# 3. Price vs bedrooms boxplot
df.boxplot(column="price", by="bedrooms", figsize=(8, 5))
plt.title("Price by Bedroom Count")
plt.suptitle("")
plt.tight_layout()
plt.savefig("data/processed/price_by_bedrooms.png")
plt.close()

# 4. Correlation matrix
print(df[["price", "log_price", "bedrooms", "bathrooms"]].corr())
