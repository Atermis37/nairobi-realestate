import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path("data/processed/listings_clean.csv")
FEATURES_PATH = Path("data/processed/listings_features.csv")

TIER_MAP = {
    "Westlands": "Premium",
    "Lavington": "Mid",
    "Kileleshwa": "Mid",
    "Kilimani": "Mid",
    "Syokimau": "Mid",
    "Parklands": "Mid",
    "South B": "Affordable",
    "Kitisuru": "Mid",
    "Other": "Mid",
}

def engineer_features(clean_path=CLEAN_PATH, out_path=FEATURES_PATH):
    df = pd.read_csv(clean_path)

    # Bed-bath ratio
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, 1)

    # Neighborhood tier
    df["tier"] = df["location"].map(TIER_MAP)

    # Encode tier as ordinal
    tier_order = {"Affordable": 1, "Mid": 2, "Premium": 3}
    df["tier_encoded"] = df["tier"].map(tier_order)

    df.to_csv(out_path, index=False)
    print(f"Features saved: {out_path}")
    print(df[["location", "tier", "tier_encoded", "bed_bath_ratio"]].head(10))

if __name__ == "__main__":
    engineer_features()
