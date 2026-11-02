import joblib
import numpy as np
import pandas as pd

a = joblib.load("models/rf_model.joblib")

def predict(bedrooms, bathrooms, location):
    tier = a["tier_map"].get(location, "Mid")
    row = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bed_bath_ratio": bedrooms / max(bathrooms, 1),
        "tier_Premium": 1 if tier == "Premium" else 0,
    }])[a["feature_cols"]]
    log_p = a["model"].predict(row)[0]
    return np.exp(log_p), np.exp(log_p + a["resid_lo"]), np.exp(log_p + a["resid_hi"])

for args in [(2, 2, "Westlands"), (3, 3, "Kilimani"), (1, 1, "Karen")]:
    p, lo, hi = predict(*args)
    n = a["location_counts"].get(args[2], 0)
    print(f"{args}: KES {p:,.0f}  (90%: {lo:,.0f} - {hi:,.0f})  [location n={n}]")
