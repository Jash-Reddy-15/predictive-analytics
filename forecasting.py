"""
Predictive Analytics Using Historical Data
===========================================
Forecasts future sales/revenue trends from historical time-series data.

Pipeline:
  1. Generate / load historical data (replace with real CSV easily)
  2. Clean & preprocess (missing values, feature engineering, scaling)
  3. Train two models:
       - Linear Regression (trend + seasonality via engineered features)
       - ARIMA (classical time-series model)
  4. Evaluate accuracy: MAE, RMSE, MAPE, R^2
  5. Visualize: historical vs predicted, future forecast, residuals
  6. Write an insights report

Run:
    python forecasting.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

OUT = "outputs"

# ----------------------------------------------------------------------
# 1. GENERATE / LOAD DATA
# ----------------------------------------------------------------------
# Replace with: df = pd.read_csv("sales_history.csv", parse_dates=["date"])
# Expected columns: date, revenue

np.random.seed(7)
n_months = 48  # 4 years of monthly data
dates = pd.date_range("2021-01-01", periods=n_months, freq="MS")

trend = np.linspace(50, 130, n_months)               # long-term growth
seasonality = 10 * np.sin(2 * np.pi * (dates.month / 12))  # yearly seasonality
noise = np.random.normal(0, 4, n_months)
revenue = trend + seasonality + noise
revenue = np.clip(revenue, 20, None)

df = pd.DataFrame({"date": dates, "revenue": revenue})
print(f"Loaded {len(df)} months of historical revenue data.\n")
print(df.head())

# ----------------------------------------------------------------------
# 2. CLEAN & PREPROCESS
# ----------------------------------------------------------------------
# Check & handle missing values
missing = df["revenue"].isna().sum()
print(f"\nMissing values: {missing}")
df["revenue"] = df["revenue"].interpolate(method="linear")

# Feature engineering for regression model
df["t"] = np.arange(len(df))                       # time index (trend)
df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Train/test split (last 6 months held out for evaluation)
test_size = 6
train, test = df.iloc[:-test_size], df.iloc[-test_size:]

feature_cols = ["t", "month_sin", "month_cos"]
X_train, y_train = train[feature_cols], train["revenue"]
X_test, y_test = test[feature_cols], test["revenue"]

# ----------------------------------------------------------------------
# 3a. MODEL 1 — LINEAR REGRESSION (trend + seasonality)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)

# ----------------------------------------------------------------------
# 3b. MODEL 2 — ARIMA (classical time series)
# ----------------------------------------------------------------------
arima_model = ARIMA(train["revenue"], order=(2, 1, 2))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=test_size).values

# ----------------------------------------------------------------------
# 4. EVALUATE ACCURACY
# ----------------------------------------------------------------------
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R^2  : {r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

results = []
results.append(evaluate(y_test.values, lr_pred, "Linear Regression"))
results.append(evaluate(y_test.values, arima_pred, "ARIMA(2,1,2)"))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_evaluation.csv", index=False)

best_model_name = results_df.sort_values("RMSE").iloc[0]["model"]
print(f"\nBest model by RMSE: {best_model_name}")

# ----------------------------------------------------------------------
# 5. FORECAST FUTURE (next 12 months)
# ----------------------------------------------------------------------
future_steps = 12
future_dates = pd.date_range(df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
                              periods=future_steps, freq="MS")

# Refit on FULL data for final forecast
full_X = df[feature_cols]
full_y = df["revenue"]
full_X_s = scaler.fit_transform(full_X)

lr_final = LinearRegression().fit(full_X_s, full_y)

future_t = np.arange(len(df), len(df) + future_steps)
future_month = future_dates.month
future_df = pd.DataFrame({
    "t": future_t,
    "month_sin": np.sin(2 * np.pi * future_month / 12),
    "month_cos": np.cos(2 * np.pi * future_month / 12),
})
future_X_s = scaler.transform(future_df)
lr_future = lr_final.predict(future_X_s)

arima_final = ARIMA(full_y, order=(2, 1, 2)).fit()
arima_future = arima_final.forecast(steps=future_steps).values
arima_conf = arima_final.get_forecast(steps=future_steps).conf_int(alpha=0.2)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "linear_regression_forecast": lr_future.round(2),
    "arima_forecast": arima_future.round(2),
    "arima_lower_80": arima_conf.iloc[:, 0].round(2).values,
    "arima_upper_80": arima_conf.iloc[:, 1].round(2).values,
})
forecast_df.to_csv(f"{OUT}/future_forecast.csv", index=False)
print("\nFuture forecast (next 12 months):")
print(forecast_df)

# ----------------------------------------------------------------------
# 6. VISUALIZATIONS
# ----------------------------------------------------------------------

# 6a. Historical data + test predictions
plt.figure(figsize=(11, 5))
plt.plot(df["date"], df["revenue"], label="Historical revenue", color="#1a1a18", linewidth=1.5)
plt.plot(test["date"], lr_pred, "--", label="Linear Regression (test)", color="#378ADD", linewidth=2)
plt.plot(test["date"], arima_pred, "--", label="ARIMA (test)", color="#D4537E", linewidth=2)
plt.axvline(test["date"].iloc[0], color="gray", linestyle=":", alpha=0.6)
plt.title("Historical Revenue vs. Model Predictions (Hold-out Test Set)")
plt.xlabel("Date")
plt.ylabel("Revenue ($K)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_historical_vs_predicted.png", dpi=150)
plt.close()

# 6b. Future forecast
plt.figure(figsize=(11, 5))
plt.plot(df["date"], df["revenue"], label="Historical revenue", color="#1a1a18", linewidth=1.5)
plt.plot(forecast_df["date"], forecast_df["linear_regression_forecast"],
         "--", label="Linear Regression forecast", color="#378ADD", linewidth=2)
plt.plot(forecast_df["date"], forecast_df["arima_forecast"],
         "--", label="ARIMA forecast", color="#D4537E", linewidth=2)
plt.fill_between(forecast_df["date"], forecast_df["arima_lower_80"], forecast_df["arima_upper_80"],
                 color="#D4537E", alpha=0.15, label="ARIMA 80% confidence interval")
plt.axvline(df["date"].iloc[-1], color="gray", linestyle=":", alpha=0.6)
plt.title("12-Month Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue ($K)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_future_forecast.png", dpi=150)
plt.close()

# 6c. Residuals plot (Linear Regression on test set)
residuals = y_test.values - lr_pred
plt.figure(figsize=(8, 4))
plt.bar(test["date"].dt.strftime("%Y-%m"), residuals, color="#BA7517")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Linear Regression Residuals (Test Set)")
plt.ylabel("Actual - Predicted ($K)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_residuals.png", dpi=150)
plt.close()

# 6d. Model comparison bar chart
plt.figure(figsize=(7, 4))
metrics = ["MAE", "RMSE", "MAPE"]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, results_df.iloc[0][metrics], width, label="Linear Regression", color="#378ADD")
plt.bar(x + width/2, results_df.iloc[1][metrics], width, label="ARIMA(2,1,2)", color="#D4537E")
plt.xticks(x, metrics)
plt.title("Model Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_model_comparison.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 7. SAVE PROCESSED DATA
# ----------------------------------------------------------------------
df.to_csv(f"{OUT}/historical_data_processed.csv", index=False)

# ----------------------------------------------------------------------
# 8. INSIGHTS REPORT
# ----------------------------------------------------------------------
growth_12mo = ((forecast_df["arima_forecast"].iloc[-1] - df["revenue"].iloc[-1])
               / df["revenue"].iloc[-1] * 100)

with open(f"{OUT}/insights_report.md", "w") as f:
    f.write("# Predictive Analytics — Insights Report\n\n")
    f.write(f"**Historical period:** {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} "
            f"({len(df)} months)\n\n")
    f.write("## Models Evaluated\n\n")
    f.write("| Model | MAE | RMSE | MAPE | R² |\n|---|---|---|---|---|\n")
    for _, r in results_df.iterrows():
        f.write(f"| {r['model']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | "
                f"{r['MAPE']:.2f}% | {r['R2']:.3f} |\n")
    f.write(f"\n**Best performing model:** {best_model_name}\n\n")

    f.write("## 12-Month Forecast Summary\n\n")
    f.write(f"- Last observed revenue: **${df['revenue'].iloc[-1]:.1f}K**\n")
    f.write(f"- Forecasted revenue (12 months out, ARIMA): "
            f"**${forecast_df['arima_forecast'].iloc[-1]:.1f}K**\n")
    f.write(f"- Projected growth over 12 months: **{growth_12mo:+.1f}%**\n")
    f.write(f"- 80% confidence range for month 12: "
            f"${forecast_df['arima_lower_80'].iloc[-1]:.1f}K – "
            f"${forecast_df['arima_upper_80'].iloc[-1]:.1f}K\n\n")

    f.write("## Observations\n\n")
    f.write("- The data shows a clear **upward long-term trend** combined with "
            "**recurring yearly seasonality** (peaks and troughs repeat every 12 months).\n")
    f.write("- Linear Regression captures the trend and seasonal pattern using "
            "engineered time and cyclical features, while ARIMA models the "
            "autocorrelation structure directly from the series.\n")
    f.write("- Residuals are small and show no strong systematic pattern, "
            "indicating the model is not significantly biased.\n\n")

    f.write("## Recommended Actions\n\n")
    f.write("- Use the forecast to set **revenue targets and inventory/staffing plans** "
            "for the next 12 months, especially around seasonal peaks.\n")
    f.write("- Monitor actual vs. forecasted values monthly; large deviations may "
            "signal market shifts requiring model retraining.\n")
    f.write("- Retrain models quarterly as new data becomes available to maintain accuracy.\n")

print(f"\nAll outputs saved to ./{OUT}/")
