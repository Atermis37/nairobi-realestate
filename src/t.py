import pandas as pd
df = pd.read_csv("data/processed/listings_clean.csv")
print(df["location"].value_counts())
