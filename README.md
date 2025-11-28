# S&P 500 Forecasting with Falcon-TST and Baselines

This script downloads S&P 500 daily data, fits a few simple baselines, and runs multi-horizon forecasts with the Falcon Time Series Transformer. It evaluates rolling blocks of 1, 5, 10, and 20 days into the 2025 test period, then plots the results.

## What it does
- Pulls multi-year S&P 500 OHLCV data from Yahoo Finance.
- Splits data into pre-2025 (training) and 2025-to-date (testing).
- Standardizes features and runs chunked multi-step forecasts:
  - Falcon-TST (Hugging Face `ant-intl/Falcon-TST_Large` by default).
  - Moving Average, ARIMA, and GARCH baselines.
- Evaluates metrics for each horizon and saves a 4-panel plot at `sp500_falcon_forecast.png` (2024 history + 2025 forecasts).

## Experimental Results


### Horizon 1 day

| Model          | MAE      | MSE         | RMSE     | MAPE%  |
|----------------|----------|-------------|----------|--------|
| Falcon-TST     | 490.1231 | 367,317.7885| 606.0675 | 8.0643 |
| Moving Average | 493.2862 | 372,597.5053| 610.4077 | 8.1000 |
| ARIMA          | 497.2487 | 379,735.4557| 616.2268 | 8.1733 |
| GARCH          | 497.4448 | 379,941.8820| 616.3943 | 8.1782 |
```

---

### Horizon 5 days

| Model          | MAE      | MSE         | RMSE     | MAPE%  |
|----------------|----------|-------------|----------|--------|
| Falcon-TST     | 508.9289 | 395,586.2408| 628.9565 | 8.3629 |
| Moving Average | 491.6399 | 369,765.5994| 608.0835 | 8.0680 |
| ARIMA          | 501.2288 | 386,329.9360| 621.5545 | 8.2385 |
| GARCH          | 501.7832 | 387,397.9526| 622.4130 | 8.2535 |
```

---

### Horizon 10 days

| Model          | MAE      | MSE         | RMSE     | MAPE%  |
|----------------|----------|-------------|----------|--------|
| Falcon-TST     | 498.0493 | 376,021.1484| 613.2056 | 8.1852 |
| Moving Average | 490.9558 | 367,846.4516| 606.5035 | 8.0454 |
| ARIMA          | 488.1131 | 363,371.4831| 602.8030 | 8.0288 |
| GARCH          | 488.4643 | 363,693.2718| 603.0699 | 8.0459 |
```

---

### Horizon 20 days

| Model          | MAE      | MSE         | RMSE     | MAPE%  |
|----------------|----------|-------------|----------|--------|
| Falcon-TST     | 495.0069 | 370,717.5058| 608.8658 | 8.1082 |
| Moving Average | 466.8213 | 332,509.4500| 576.6363 | 7.6444 |
| ARIMA          | 477.8699 | 348,227.8984| 590.1084 | 7.8594 |
| GARCH          | 482.9414 | 355,615.4531| 596.3350 | 7.9575 |
```



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
```math
\text{MAE}  = \frac{1}{N} \sum_{t=1}^N |y_t - \hat{y}_t|
```
```math
\text{MSE}  = \frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2 \\
```
```math
\text{RMSE} = \sqrt{\text{MSE}}
```
```math
\text{MAPE} = \frac{100}{N} \sum_{t=1}^N \left|\frac{y_t - \hat{y}_t}{y_t}\right|
```

## Baseline formulations
- Moving Average (window $$w$$):
```math
\hat{y}_{t+h} = \frac{1}{w} \sum_{i=1}^{w} y_{t+1-i}
```
  (Chunked: reuse this mean for the next block, then refresh history with revealed actuals.)
- ARIMA $$(p,d,q)$$:
```math
\hat{y}_{t+1}, \dots, \hat{y}_{t+h}
```
  come from the ARIMA mean equation fitted on recent closes (with trend), then history is rolled forward with actuals.
- GARCH(1,1) on returns:
```math
\hat{P}_{t+j} = \hat{P}_{t+j-1} \times (1 + \hat{r}_{t+j})
```
  where $$\hat{r}_{t+j}$$ is the forecast mean return; after each block, history is refreshed with actuals.
- Falcon-TST: Transformer that forecasts the next block (1/5/10/20 days) from the recent standardized context; after each block, the true block is appended and the next block is predicted.

## Notes
- Multi-horizon evaluation is “chunked”: forecast a block (e.g., 10 days), reveal those actuals, then continue. This is harder than single-step teacher forcing and shows how models drift over several days.
- GARCH fitting can be slow on CPU; reduce `baseline_window_days` if needed.
- Yahoo Finance data pulls require network access. Ensure the machine can reach `query2.finance.yahoo.com`.
