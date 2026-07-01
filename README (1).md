# Stock Market Trend Forecasting

Predicts a stock's closing price using the **previous day's** Open, High, Low, and Volume (OHLV) via regression. Uses live data from Yahoo Finance through the `yfinance` library.

## Setup

pip install \-r requirements.txt

python stock\_forecast.py

Requires internet access (pulls live data from Yahoo Finance).

## Configuration

Edit the top of `stock_forecast.py`:

TICKER \= "AAPL"      \# any valid ticker — try "RELIANCE.NS" or "TCS.NS" for NSE stocks

PERIOD \= "2y"         \# "1y", "2y", "5y", "max"

TEST\_SIZE \= 0.2       \# fraction of most recent data held out for testing

## What it does

1. Downloads OHLCV data for the chosen ticker.  
2. Builds features: `Prev_Open`, `Prev_High`, `Prev_Low`, `Prev_Volume` — each shifted one day back — to predict today's `Close`.  
3. Splits chronologically (not randomly) into train/test, since test data must come *after* training data in time. Random shuffling would leak future information into training.  
4. Trains two models — Linear Regression and Random Forest — and compares both against a naive baseline (predict "today's close \= yesterday's close").  
5. Saves a comparison plot and metrics table to `outputs/`.

## Why compare against a naive baseline?

Daily stock closes are strongly autocorrelated — today's price is usually close to yesterday's. Because of this, "just guess yesterday's close" is a surprisingly hard baseline to beat, and it's common for trained models to land close to it. Seeing the trained models roughly match (rather than crush) the naive baseline isn't a bug — it's the honest, expected result and a useful thing to understand and be able to explain about this kind of short-horizon financial prediction.

## Output

- `outputs/forecast_results.png` — actual vs. predicted close on the test set, plus a residual plot.  
- `outputs/model_comparison.csv` — RMSE / MAE / R² for all three models.

## Metrics explained

| Metric | Meaning |
| :---- | :---- |
| RMSE | Root Mean Squared Error — typical prediction error, in dollars. Penalizes large errors more than small ones. |
| MAE | Mean Absolute Error — average absolute error, in dollars. More intuitive, less sensitive to outliers than RMSE. |
| R² | Fraction of variance explained. 1.0 \= perfect fit, 0.0 \= no better than predicting the mean every time. |

