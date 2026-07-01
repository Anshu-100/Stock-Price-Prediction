"""
Stock Market Trend Forecasting
==============================
Predicts a stock's closing price using the PREVIOUS day's OHLV metrics
(Open, High, Low, Volume) via regression.

Run this on your own machine (needs internet access to Yahoo Finance):
    pip install yfinance scikit-learn pandas numpy matplotlib seaborn
    python stock_forecast.py

Change TICKER below to any valid stock symbol (e.g. "TCS.NS" for NSE-listed
Indian stocks, "AAPL" for Apple, "RELIANCE.NS" for Reliance).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------
# CONFIG — change these to experiment
# ----------------------------------------------------------------------
TICKER = "AAPL"          # stock symbol to forecast
PERIOD = "2y"            # how much history to pull ("1y", "2y", "5y", "max")
TEST_SIZE = 0.2          # fraction of most-recent data held out for testing

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ----------------------------------------------------------------------
# 1. FETCH DATA
# ----------------------------------------------------------------------
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    print(f"Fetching {period} of data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. Check the ticker symbol "
            f"and your internet connection."
        )

    # yfinance sometimes returns MultiIndex columns for a single ticker —
    # flatten them so the rest of the script can assume plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"Retrieved {len(df)} trading days, from {df.index[0].date()} "
          f"to {df.index[-1].date()}.")
    return df


# ----------------------------------------------------------------------
# 2. BUILD FEATURES
#    Predict TODAY's close using YESTERDAY's Open/High/Low/Volume.
#    This is what makes it a real forecast rather than leakage: on any
#    given morning, yesterday's OHLV is already known, today's close
#    is not.
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["Prev_Open"] = df["Open"].shift(1)
    feat["Prev_High"] = df["High"].shift(1)
    feat["Prev_Low"] = df["Low"].shift(1)
    feat["Prev_Volume"] = df["Volume"].shift(1)
    feat["Target_Close"] = df["Close"]  # today's close = what we predict
    feat = feat.dropna()
    return feat


# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT — chronological, NOT random.
#    Random shuffling would put future days in the training set and let
#    the model implicitly learn from data it shouldn't have access to
#    yet ("lookahead bias"). The test set must be the most recent slice.
# ----------------------------------------------------------------------
def chronological_split(feat: pd.DataFrame, test_size: float):
    split_idx = int(len(feat) * (1 - test_size))
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]
    print(f"\nTrain set: {len(train)} days "
          f"({train.index[0].date()} to {train.index[-1].date()})")
    print(f"Test set:  {len(test)} days "
          f"({test.index[0].date()} to {test.index[-1].date()})")
    return train, test


# ----------------------------------------------------------------------
# 4. TRAIN MODELS + EVALUATE
# ----------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  RMSE : ${rmse:.2f}  (typical error size, in dollars)")
    print(f"  MAE  : ${mae:.2f}  (average absolute error, in dollars)")
    print(f"  R^2  : {r2:.4f}  (1.0 = perfect, 0.0 = no better than the mean)")
    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "preds": y_pred}


def main():
    df = fetch_data(TICKER, PERIOD)
    feat = build_features(df)

    feature_cols = ["Prev_Open", "Prev_High", "Prev_Low", "Prev_Volume"]
    X = feat[feature_cols]
    y = feat["Target_Close"]

    train, test = chronological_split(feat, TEST_SIZE)
    X_train, y_train = train[feature_cols], train["Target_Close"]
    X_test, y_test = test[feature_cols], test["Target_Close"]

    # Scale features — Volume is on a totally different numeric scale
    # (millions) than price (tens/hundreds), which hurts Linear Regression
    # if left unscaled. Fit the scaler on TRAIN ONLY, then apply to test —
    # fitting on the full dataset would leak test-set statistics into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # --- Model 1: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results.append(evaluate("Linear Regression", y_test, lr_pred))

    print("\n  Learned coefficients (after scaling):")
    for feat_name, coef in zip(feature_cols, lr.coef_):
        print(f"    {feat_name:<12}: {coef:+.3f}")

    # --- Model 2: Random Forest ---
    # Tree-based models don't need scaling, so we feed it raw X_train/X_test.
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred))

    print("\n  Feature importances:")
    for feat_name, imp in zip(feature_cols, rf.feature_importances_):
        print(f"    {feat_name:<12}: {imp:.3f}")

    # --- Naive baseline: "today's close = yesterday's close" ---
    # Any real model MUST beat this, or it isn't adding value. Stock
    # prices are highly autocorrelated day-to-day, so this baseline is
    # surprisingly strong and is the honest bar to clear.
    naive_pred = test["Prev_Open"].values * 0 + train["Target_Close"].iloc[-1]
    naive_pred = np.concatenate([[train["Target_Close"].iloc[-1]],
                                  test["Target_Close"].values[:-1]])
    results.append(evaluate("Naive Baseline (yesterday's close)",
                             y_test, naive_pred))

    # ------------------------------------------------------------------
    # 5. PLOTS
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax = axes[0]
    ax.plot(y_test.index, y_test.values, label="Actual Close",
            color="black", linewidth=2)
    ax.plot(y_test.index, lr_pred, label="Linear Regression",
            linestyle="--", alpha=0.8)
    ax.plot(y_test.index, rf_pred, label="Random Forest",
            linestyle="--", alpha=0.8)
    ax.set_title(f"{TICKER}: Actual vs Predicted Closing Price (Test Set)")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1]
    residuals = y_test.values - rf_pred
    ax.scatter(y_test.index, residuals, alpha=0.6, color="steelblue")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Random Forest Residuals (Actual − Predicted)")
    ax.set_ylabel("Error ($)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_path = "outputs/forecast_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # ------------------------------------------------------------------
    # 6. SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    summary = pd.DataFrame(
        [{"Model": r["name"], "RMSE ($)": round(r["rmse"], 2),
          "MAE ($)": round(r["mae"], 2), "R^2": round(r["r2"], 4)}
         for r in results]
    )
    print(summary.to_string(index=False))
    summary.to_csv("outputs/model_comparison.csv", index=False)
    print("\nSaved comparison table to outputs/model_comparison.csv")


if __name__ == "__main__":
    main()
