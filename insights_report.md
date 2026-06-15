# Predictive Analytics — Insights Report

**Historical period:** 2021-01-01 to 2024-12-01 (48 months)

## Models Evaluated

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | 4.89 | 5.58 | 4.09% | 0.724 |
| ARIMA(2,1,2) | 7.52 | 12.39 | 5.72% | -0.358 |

**Best performing model:** Linear Regression

## 12-Month Forecast Summary

- Last observed revenue: **$139.0K**
- Forecasted revenue (12 months out, ARIMA): **$153.6K**
- Projected growth over 12 months: **+10.5%**
- 80% confidence range for month 12: $111.9K – $195.4K

## Observations

- The data shows a clear **upward long-term trend** combined with **recurring yearly seasonality** (peaks and troughs repeat every 12 months).
- Linear Regression captures the trend and seasonal pattern using engineered time and cyclical features, while ARIMA models the autocorrelation structure directly from the series.
- Residuals are small and show no strong systematic pattern, indicating the model is not significantly biased.

## Recommended Actions

- Use the forecast to set **revenue targets and inventory/staffing plans** for the next 12 months, especially around seasonal peaks.
- Monitor actual vs. forecasted values monthly; large deviations may signal market shifts requiring model retraining.
- Retrain models quarterly as new data becomes available to maintain accuracy.
