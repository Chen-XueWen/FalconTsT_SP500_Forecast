# S&P 500 Forecasting with Falcon-TST and Baselines

This script downloads S&P 500 daily data, fits a few simple baselines, and runs multi-horizon forecasts with the Falcon Time Series Transformer. It evaluates rolling blocks of 1, 5, 10, and 20 days into the 2025 test period, then plots the results.

## What it does
- Pulls multi-year S&P 500 OHLCV data from Yahoo Finance.
- Splits data into pre-2025 (training) and 2025-to-date (testing).
- Standardizes features and runs chunked multi-step forecasts:
  - Falcon-TST (Hugging Face `ant-intl/Falcon-TST_Large` by default).
  - Moving Average, ARIMA, and GARCH baselines.
- Evaluates metrics for each horizon and saves a 4-panel plot at `sp500_falcon_forecast.png` (2024 history + 2025 forecasts).

## Requirements
- Python 3.8+
- See `requirements.txt` for Python packages.
- Optional GPU (`--device cuda`) for faster Falcon-TST inference.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python sp500_falcon_forecast.py \
  --device cpu \               # or cuda/mps/auto
  --falcon_model ant-intl/Falcon-TST_Large \
  --ma_window 20 \             # moving average window
  --arima_order 5 1 0 \        # ARIMA (p d q)
  --baseline_window_days 365   # recent history for baselines
```

### Key arguments
- `--device` (`auto`|`cpu`|`cuda`|`mps`): Torch device for Falcon-TST.
- `--falcon_model`: HF model id (defaults to `ant-intl/Falcon-TST_Large` with `trust_remote_code=True`).
- `--ma_window`: Moving average window length.
- `--arima_order`: ARIMA order (p, d, q).
- `--baseline_window_days`: How many days of recent training closes to use for baselines (keeps them responsive).

## Outputs
- Console metrics for each horizon (1/5/10/20-day blocks) and each model.
- Plot saved to `sp500_falcon_forecast.png` showing actual closes (2024+2025) and aligned forecasts for all horizons.

## Error metrics (per horizon)
- MAE: \( \text{MAE} = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t| \)
- MSE: \( \text{MSE} = \frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2 \)
- RMSE: \( \text{RMSE} = \sqrt{\text{MSE}} \)
- MAPE: \( \text{MAPE} = \frac{100}{N} \sum_{t=1}^N \left|\frac{y_t - \hat{y}_t}{y_t}\right| \)

## Baseline formulations
- Moving Average (window \(w\)): \(\hat{y}_{t+h} = \frac{1}{w} \sum_{i=1}^{w} y_{t+1-i}\). In chunked mode, the same mean is used for the next block, then history is updated with revealed actuals.
- ARIMA \((p,d,q)\): fitted on the recent close history with trend; forecasts the next block \(\hat{y}_{t+1}, \dots, \hat{y}_{t+h}\) from the ARIMA mean equation, then history is rolled forward with actuals.
- GARCH(1,1) on returns: model returns \(r_t\); mean return forecast \(\hat{r}_{t+j}\) is compounded into price: \(\hat{P}_{t+j} = \hat{P}_{t+j-1} \times (1 + \hat{r}_{t+j})\) for the block, then history is rolled forward with actuals.
- Falcon-TST: Transformer that forecasts the next block (1/5/10/20 days) from the recent standardized context; after each block, the true block is appended and the next block is predicted.

## Notes
- Multi-horizon evaluation is “chunked”: forecast a block (e.g., 10 days), reveal those actuals, then continue. This is harder than single-step teacher forcing and shows how models drift over several days.
- GARCH fitting can be slow on CPU; reduce `baseline_window_days` if needed.
- Yahoo Finance data pulls require network access. Ensure the machine can reach `query2.finance.yahoo.com`.
