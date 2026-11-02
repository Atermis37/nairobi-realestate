import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score

from features import TIER_MAP

FEATURES_PATH = Path("data/processed/listings_features.csv")
MODEL_PATH = Path("models/rf_model.joblib")
FEATURE_COLS = ["bedrooms", "bed_bath_ratio", "tier_Premium"]
TARGET = "log_price"

df = pd.read_csv(FEATURES_PATH)
X, y = df[FEATURE_COLS], df[TARGET]

# Same spec that won the paired comparison (no max_depth)
rf = RandomForestRegressor(n_estimators=300, random_state=42)

# Out-of-fold predictions -> honest residuals for the prediction interval
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(rf, X, y, cv=cv.split(X, df["tier_Premium"]))
resid = y - oof
lo, hi = np.quantile(resid, [0.05, 0.95])
print(f"OOF R2: {r2_score(y, oof):.4f} | 90% residual band (log scale): [{lo:+.3f}, {hi:+.3f}]")

# Final fit on all data
rf.fit(X, y)

artifact = {
    "model": rf,
    "feature_cols": FEATURE_COLS,
    "tier_map": TIER_MAP,
    "location_counts": df["location"].value_counts().to_dict(),
    "resid_lo": float(lo),
    "resid_hi": float(hi),
    "n_train": len(df),
    "cv_r2": 0.7754,
}
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(artifact, MODEL_PATH)
print(f"Saved -> {MODEL_PATH}")
