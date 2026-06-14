# Predictive Analytics Using Historical Data

Forecasts future revenue/sales trends from historical time-series data using **regression** and **ARIMA** time-series models, with full evaluation, visualization, and a written insights report.

## What This Project Does

1. **Loads historical data** — monthly revenue over 4 years (swap in your own CSV easily)
2. **Cleans & preprocesses** — handles missing values, engineers trend (`t`) and seasonal (`month_sin`/`month_cos`) features
3. **Trains two models**:
   - **Linear Regression** — trend + cyclical seasonality via engineered features
   - **ARIMA(2,1,2)** — classical autoregressive time-series model
4. **Evaluates accuracy** — MAE, RMSE, MAPE, R² on a 6-month hold-out test set
5. **Forecasts 12 months ahead** — including ARIMA's 80% confidence interval
6. **Visualizes everything** — historical vs predicted, future forecast, residuals, model comparison
7. **Writes an insights report** — growth projection + recommended business actions

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas / numpy | Data handling |
| scikit-learn | Linear Regression, scaling, evaluation metrics |
| statsmodels | ARIMA time-series model |
| matplotlib | Visualizations |

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
python forecasting.py
```

All outputs are saved to `outputs/`.

## Output Files

| File | Description |
|---|---|
| `historical_data_processed.csv` | Cleaned data with engineered features |
| `model_evaluation.csv` | MAE / RMSE / MAPE / R² for each model |
| `future_forecast.csv` | 12-month forecast (both models + ARIMA confidence interval) |
| `01_historical_vs_predicted.png` | Historical data with test-set predictions overlaid |
| `02_future_forecast.png` | 12-month forecast with confidence band |
| `03_residuals.png` | Residuals plot for error analysis |
| `04_model_comparison.png` | Side-by-side accuracy comparison |
| `insights_report.md` | Written summary + business recommendations |

## Example Results

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Linear Regression | ~4.9 | ~5.6 | ~4.1% | ~0.72 |
| ARIMA(2,1,2) | ~7.5 | ~12.4 | ~5.7% | negative |

Linear Regression performed best here because the synthetic data has a clean trend + repeating yearly seasonality, which the engineered cyclical features capture directly.

## Using Your Own Data

Replace the synthetic data block with:

```python
df = pd.read_csv("sales_history.csv", parse_dates=["date"])
```

Required columns: `date` (monthly, sorted) and `revenue` (or rename the target column throughout the script).

## Learning Outcomes

- Time-series preprocessing (handling missing values, feature engineering for trend/seasonality)
- Regression vs. classical time-series (ARIMA) forecasting
- Model evaluation with MAE, RMSE, MAPE, R²
- Forecast visualization with confidence intervals
- Translating forecasts into business decisions (targets, inventory, retraining cadence)

## License

MIT — free to use, modify, and distribute.
# predictive-analytics
