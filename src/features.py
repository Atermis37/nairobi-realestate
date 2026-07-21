import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path("data/processed/listings_clean.csv")
FEATURES_PATH = Path("data/processed/listings_features.csv")

# Tier assignment: data-driven where sample size allows, domain judgment only
# as a last resort, explicitly flagged wherever n is too small to trust.
#
# Reliable anchors: Westlands (median 8.5M, n=1108), Kilimani (median 7.36M,
# n=820).
#
# PREMIUM -- median well above both anchors, n large enough to trust (n>=4):
#   Westlands        8.5M   n=1108  (original anchor)
#   Muthaiga         39.0M  n=5
#   Parklands        25.0M  n=9     -- CORRECTED from Mid (Phase 4 override
#                     was made at n=4 with a then-unreliable 25.2M median;
#                     n=9 now confirms the same magnitude -- this is a real,
#                     repeatable signal, not noise. Revisiting a small-n
#                     override once better data arrives is the same pattern
#                     already applied to South B and the tier-significance
#                     finding; it is not a reversal to be embarrassed about.
#   General Mathenge  23.95M n=4
#   Brookside         23.0M  n=9     -- also corrected from an earlier
#                     Mid-by-fallback default; n=9 confirms well above anchor.
#   Kiambu Road       14.45M n=4     -- confirms the earlier tentative
#                     Premium assignment made with n=1.
#   Gigiri            13.5M  n=4
#
# MID -- everything else, including several locations where the single
# available data point actively CONTRADICTS common assumptions about the
# area (e.g. Karen showing BELOW both anchors at n=1 -- almost certainly
# because Karen's housing stock is dominated by standalone houses on large
# plots, not apartments, so one atypical apartment listing is not
# representative of "Karen" as a market). These are defaulted to Mid as the
# conservative, least-assumption-laden choice, NOT because the data supports
# a Mid classification -- there simply isn't enough data (n=1-3) to support
# ANY classification. This is disclosed explicitly here and must be repeated
# in the report: these are placeholders, not findings.
#   Karen           6.5M   n=1   -- contradicts reputation as a premium area;
#                   apartment submarket likely unrepresentative of the area
#   Lower Kabete    4.1M   n=1
#   Loresho         22.0M  n=1
#   Valley Arcade   17.5M  n=2
#   Spring Valley   39.5M  n=3   -- borderline; n still too thin to trust
#                   despite the large median, kept at Mid pending more data
#   Mombasa Road    36.5M  n=2   -- contradicts commercial/lower-income
#                   reputation; likely one atypical luxury listing
#   Eastleigh       28.5M  n=1   -- same caveat as Mombasa Road
#   Embakasi        8.9M   n=1
#   Ngong Road      10.3M  n=1
#   Langata         10.0M  n=1
#
# Small-n locations folded into Mid from earlier phases (South B, Kitisuru,
# Syokimau-excluded) remain Mid per the established precedent.
TIER_MAP = {
    "Westlands": "Premium",
    "Muthaiga": "Premium",
    "Parklands": "Premium",
    "General Mathenge": "Premium",
    "Brookside": "Premium",
    "Kiambu Road": "Premium",
    "Gigiri": "Premium",

    "Lavington": "Mid",
    "Kileleshwa": "Mid",
    "Kilimani": "Mid",
    "South B": "Mid",
    "Kitisuru": "Mid",
    "Other": "Mid",
    "Karen": "Mid",
    "Lower Kabete": "Mid",
    "Loresho": "Mid",
    "Valley Arcade": "Mid",
    "Spring Valley": "Mid",
    "Mombasa Road": "Mid",
    "Eastleigh": "Mid",
    "Embakasi": "Mid",
    "Ngong Road": "Mid",
    "Langata": "Mid",
}


def engineer_features(clean_path=CLEAN_PATH, out_path=FEATURES_PATH):
    df = pd.read_csv(clean_path)

    # --- Size signal ---
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"].replace(0, 1)
    df = df.drop(columns=["bathrooms"])

    # --- Neighborhood tier ---
    df["tier"] = df["location"].map(TIER_MAP)

    unmapped = sorted(df.loc[df["tier"].isna(), "location"].unique())
    if unmapped:
        print(f"\n⚠ WARNING: {len(unmapped)} location(s) not in TIER_MAP, "
              f"defaulted to 'Mid': {unmapped}")
        print("  Review these: (1) confirm each is actually within Nairobi "
              "(see the Syokimau lesson), (2) add an explicit, justified "
              "tier assignment to TIER_MAP instead of relying on this default.\n")
    df["tier"] = df["tier"].fillna("Mid")

    tier_dummies = pd.get_dummies(df["tier"], prefix="tier", drop_first=True)
    assert "tier_Affordable" not in tier_dummies.columns, "Reference category not dropped as expected"
    df = pd.concat([df, tier_dummies.astype(int)], axis=1)

    df.to_csv(out_path, index=False)
    print(f"Features saved: {out_path}")
    print(df[["location", "tier"] + list(tier_dummies.columns) + ["bedrooms", "bed_bath_ratio"]].head(10))
    print("\nTier counts (flag n<10 as unreliable for coefficient interpretation):")
    print(df["tier"].value_counts())
    print("\nLocation counts within tiers:")
    print(df.groupby("tier")["location"].value_counts())


if __name__ == "__main__":
    engineer_features()
