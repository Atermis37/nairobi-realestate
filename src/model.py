import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

FEATURES_PATH = Path("data/processed/listings_features.csv")

FEATURE_COLS = ["bedrooms", "bed_bath_ratio", "tier_Premium"]
CONTINUOUS_COLS = ["bedrooms", "bed_bath_ratio"]  # standardized for Ridge/Lasso only
TARGET_COL = "log_price"


def run_ols(df):
    print("=" * 70)
    print("OLS BASELINE (statsmodels) -- full dataset, academic interpretability")
    print("=" * 70)

    X = df[FEATURE_COLS].astype(float)
    X = sm.add_constant(X)
    y = df[TARGET_COL]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    print("\nVariance Inflation Factors:")
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data.to_string(index=False))

    return model


def stratified_split(df):
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["tier"]
    )
    print(f"\nTrain: {len(train_df)} rows | Test: {len(test_df)} rows")
    print("Train tier distribution:\n", train_df["tier"].value_counts())
    print("Test tier distribution:\n", test_df["tier"].value_counts())
    return train_df, test_df


def prepare_xy(train_df, test_df):
    X_train = train_df[FEATURE_COLS].astype(float).copy()
    X_test = test_df[FEATURE_COLS].astype(float).copy()
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test_scaled[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate(name, y_test, y_pred, results):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append({"model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
    print(f"{name:20s} R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")


def run_comparison(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    print("\n" + "=" * 70)
    print("TEST SET COMPARISON (log_price scale)")
    print("=" * 70)

    results = []

    # Ridge/Lasso: standardized continuous features + raw dummies
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    evaluate("Ridge", y_test, ridge.predict(X_test_scaled), results)

    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    evaluate("Lasso", y_test, lasso.predict(X_test_scaled), results)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", y_test, rf.predict(X_test), results)

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\nRanked comparison:")
    print(results_df.to_string(index=False))

    print("\nRidge coefficients (standardized continuous + raw dummies):")
    for feat, coef in zip(FEATURE_COLS, ridge.coef_):
        print(f"  {feat:15s} {coef:+.4f}")

    print("\nLasso coefficients (0 = feature dropped by regularization):")
    for feat, coef in zip(FEATURE_COLS, lasso.coef_):
        print(f"  {feat:15s} {coef:+.4f}")

    print("\nRandom Forest feature importances:")
    for feat, imp in zip(FEATURE_COLS, rf.feature_importances_):
        print(f"  {feat:15s} {imp:.4f}")

    return results_df


def main():
    df = pd.read_csv(FEATURES_PATH)

    ols_model = run_ols(df)
    train_df, test_df = stratified_split(df)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = prepare_xy(train_df, test_df)
    results_df = run_comparison(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    return ols_model, results_df


if __name__ == "__main__":
    main()
