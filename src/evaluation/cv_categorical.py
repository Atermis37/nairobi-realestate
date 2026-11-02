import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/processed/listings_features.csv")
df["bedrooms_c"] = df["bedrooms"].clip(upper=4)  # merge the n=3 five-bed cells into 4+

X = df[["bedrooms_c", "bed_bath_ratio", "tier_Premium"]]
y = df["log_price"]

pre = ColumnTransformer([
    ("bed", OneHotEncoder(drop="first"), ["bedrooms_c"]),
    ("scale", StandardScaler(), ["bed_bath_ratio"]),
    ("pass", "passthrough", ["tier_Premium"]),
])
pipe = Pipeline([("pre", pre), ("model", LinearRegression())])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(
    pipe, X, y, cv=cv.split(X, df["tier_Premium"]),
    scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
)
print(f"Categorical-bedroom OLS CV R2: {res['test_r2'].mean():.4f} +/- {res['test_r2'].std():.4f}")
print(f"RMSE: {-res['test_rmse'].mean():.4f}")
print("RF reference: R2 0.7751 +/- 0.0224, RMSE 0.2726")
