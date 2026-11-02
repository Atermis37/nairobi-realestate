import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("data/processed/listings_features.csv")
df["bedrooms_c"] = df["bedrooms"].clip(upper=4)
# Restore the bedroom x tier interaction as explicit columns
for b in [2, 3, 4]:
    df[f"bed{b}_x_prem"] = ((df["bedrooms_c"] == b) & (df["tier_Premium"] == 1)).astype(int)

lin_cols = ["bedrooms_c", "bed_bath_ratio", "tier_Premium"] + [f"bed{b}_x_prem" for b in [2, 3, 4]]
rf_cols = ["bedrooms", "bed_bath_ratio", "tier_Premium"]
y = df["log_price"]

lin = Pipeline([
    ("pre", ColumnTransformer([
        ("bed", OneHotEncoder(drop="first"), ["bedrooms_c"]),
        ("scale", StandardScaler(), ["bed_bath_ratio"]),
        ("pass", "passthrough", ["tier_Premium"] + [f"bed{b}_x_prem" for b in [2, 3, 4]]),
    ])),
    ("m", LinearRegression()),
])
rf = RandomForestRegressor(n_estimators=300, random_state=42)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
diffs, lin_r2, rf_r2 = [], [], []
for tr, te in rskf.split(df, df["tier_Premium"]):
    lin.fit(df.iloc[tr][lin_cols], y.iloc[tr])
    rf.fit(df.iloc[tr][rf_cols], y.iloc[tr])
    a = r2_score(y.iloc[te], lin.predict(df.iloc[te][lin_cols]))
    b = r2_score(y.iloc[te], rf.predict(df.iloc[te][rf_cols]))
    lin_r2.append(a); rf_r2.append(b); diffs.append(b - a)

diffs = np.array(diffs)
print(f"Linear (cat bedrooms x tier) R2: {np.mean(lin_r2):.4f}")
print(f"RF R2:                           {np.mean(rf_r2):.4f}")
print(f"Mean paired diff (RF - linear):  {diffs.mean():+.4f}")
print(f"Std of paired diffs:             {diffs.std(ddof=1):.4f}")
print(f"RF wins {np.sum(diffs > 0)} of {len(diffs)} splits")
