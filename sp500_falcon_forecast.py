#!/usr/bin/env python3
"""
Run Falcon-TST S&P 500 Forecast with Baselines

This script downloads multi-year S&P 500 data (history up to 2024 for
training, 2025 for evaluation), generates predictions with the Falcon Time
Series Transformer, and compares them against moving average, ARIMA, and
GARCH baselines. It prints error metrics and saves a plot comparing the
forecasts to the actual closes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from transformers import AutoModel


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily_Return"]
# Falcon-TST is pretrained with a ~2880-step lookback. Pull enough history to
# avoid padding almost the entire context.
DEFAULT_LOOKBACK_DAYS = 4000  # ~11 years of daily bars
BASELINE_MAX_POINTS = 800  # cap rolling baseline history length to keep them fast
PLOT_PATH = "sp500_falcon_forecast.png"
EVAL_HORIZONS = [1, 5, 10, 20]


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def get_device(preferred: str = "auto") -> torch.device:
    """
    Select a torch device based on user preference (auto/cpu/cuda/mps)
    and runtime availability with graceful fallbacks.
    """
    preferred = (preferred or "auto").lower()
    available = []
    if torch.cuda.is_available():
        available.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        available.append("mps")
    available.append("cpu")

    if preferred == "auto":
        chosen = available[0]
    elif preferred in available:
        chosen = preferred
    else:
        print(f"Requested device '{preferred}' is unavailable. Falling back to CPU.")
        chosen = "cpu"
    return torch.device(chosen)


def fetch_sp500_data(end_date: date, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Download S&P 500 OHLCV data from Yahoo Finance. We grab ~11 years of
    history so the Falcon lookback window (2880 timesteps) is filled with
    real data instead of padding.
    """
    history_start = datetime.combine(end_date - timedelta(days=lookback_days), datetime.min.time())
    download_end = end_date + timedelta(days=1)
    df = yf.download("^GSPC", start=history_start, end=download_end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("No S&P 500 data downloaded. Check your network connection.")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)
    df = df.ffill()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def train_test_split(df: pd.DataFrame, forecast_end: date) -> DatasetSplit:
    """Split dataframe into pre-2025 training data and 2025-to-date test data."""
    cutoff = pd.Timestamp("2025-01-01")
    train_mask = df.index < cutoff
    test_mask = (df.index >= cutoff) & (df.index <= pd.Timestamp(forecast_end))
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    if len(train_df) < 365:
        raise ValueError("Training window returned too little data; expected at least 1 year of history.")
    if test_df.empty:
        raise ValueError("Test window (2025 to current date) returned no data.")
    return DatasetSplit(train=train_df, test=test_df)


def standardize_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Fit a StandardScaler on training features and transform both splits."""
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_scaled = scaler.transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)
    return scaler, train_scaled, test_scaled


def _clean_close_series(close_series) -> np.ndarray:
    """
    Convert a close price container to a 1D float numpy array, dropping NaNs
    and flattening any accidental extra dimensions.
    """
    arr = np.asarray(close_series)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        raise ValueError("Close series contained no valid numeric values.")
    return arr


def _chunked_forecast(
    test_close: pd.Series,
    train_close: pd.Series,
    horizon: int,
    forecaster,
) -> np.ndarray:
    """
    Generic chunked multi-step forecast: predict up to `horizon` days ahead,
    then reveal those actuals and continue until the test set is exhausted.
    """
    history = _clean_close_series(train_close).tolist()
    preds = []
    test_vals = _clean_close_series(test_close)
    i = 0
    n = len(test_vals)
    while i < n:
        step = min(horizon, n - i)
        hist_slice = history[-BASELINE_MAX_POINTS:]
        forecast = forecaster(hist_slice, step)
        preds.extend(forecast)
        # reveal next block of actuals
        history.extend(test_vals[i : i + step].tolist())
        i += step
    return np.array(preds, dtype=float)


def chunked_ma_forecast(train_close: pd.Series, test_close: pd.Series, horizon: int, window: int = 5) -> np.ndarray:
    """Multi-step MA: predict horizon days from current history, then roll the window."""
    window = max(1, window)

    def forecaster(hist, step):
        w = min(window, len(hist))
        val = float(np.asarray(hist[-w:], dtype=float).mean())
        return np.full(step, val, dtype=float)

    return _chunked_forecast(test_close, train_close, horizon, forecaster)


def chunked_arima_forecast(train_close: pd.Series, test_close: pd.Series, horizon: int, order=(5, 1, 0)) -> np.ndarray:
    """Multi-step ARIMA: fit on capped history, forecast horizon steps, roll forward."""

    def forecaster(hist, step):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(hist, order=order, trend="t", enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
        return np.asarray(fitted.forecast(steps=step), dtype=float)

    return _chunked_forecast(test_close, train_close, horizon, forecaster)


def chunked_garch_forecast(train_close: pd.Series, test_close: pd.Series, horizon: int) -> np.ndarray:
    """Multi-step GARCH: fit on returns, forecast mean returns horizon steps, roll forward."""

    def forecaster(hist, step):
        returns = pd.Series(hist).pct_change().dropna() * 100.0
        if len(returns) < 30:
            return np.full(step, hist[-1], dtype=float)
        am = arch_model(returns, vol="Garch", p=1, o=0, q=1, mean="AR", lags=1, dist="normal")
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=step, reindex=False)
        mean_rets = forecast.mean.values[-1] / 100.0
        price = float(hist[-1])
        preds = []
        for r in mean_rets:
            price *= (1.0 + float(r))
            preds.append(price)
        return np.array(preds, dtype=float)

    return _chunked_forecast(test_close, train_close, horizon, forecaster)


def run_falcon_forecast(
    model: AutoModel,
    context_np: np.ndarray,
    scaler: StandardScaler,
    forecast_horizon: int,
    device: torch.device,
) -> np.ndarray:
    """Generate Falcon-TST predictions and return the Close price component."""
    # The model is pretrained with a 2880-token lookback; trim to that to avoid
    # excessive padding when we have more history.
    if hasattr(model, "config") and getattr(model.config, "seq_length", None):
        seq_len = int(model.config.seq_length)
        context_np = context_np[-seq_len:]
    context_tensor = torch.tensor(context_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        preds = model.predict(context_tensor, forecast_horizon=forecast_horizon)
    preds_np = preds.detach().cpu().squeeze(0).numpy()
    preds_original = scaler.inverse_transform(preds_np)
    close_idx = FEATURE_COLUMNS.index("Close")
    return preds_original[:, close_idx]


def chunked_falcon_forecast(
    model: AutoModel,
    scaler: StandardScaler,
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    """
    Multi-step chunked forecast with Falcon-TST:
    predict up to `horizon` days, then reveal actuals, and continue.
    """
    history = np.array(train_scaled, dtype=np.float32)
    preds = []
    seq_len = getattr(getattr(model, "config", None), "seq_length", None)
    i = 0
    n = len(test_scaled)
    while i < n:
        step = min(horizon, n - i)
        context_np = history[-seq_len:] if seq_len else history
        pred_block = run_falcon_forecast(
            model=model,
            context_np=context_np,
            scaler=scaler,
            forecast_horizon=step,
            device=device,
        )
        preds.extend(pred_block.tolist())
        history = np.vstack([history, test_scaled[i : i + step]])
        i += step
    return np.array(preds, dtype=float)


def rolling_falcon_forecast(
    model: AutoModel,
    scaler: StandardScaler,
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """One-step-ahead rolling forecast with Falcon-TST; updates context with each observed test point."""
    history = np.array(train_scaled, dtype=np.float32)
    preds = []
    seq_len = getattr(getattr(model, "config", None), "seq_length", None)
    for i in range(len(test_scaled)):
        context_np = history[-seq_len:] if seq_len else history
        pred_close = run_falcon_forecast(
            model=model,
            context_np=context_np,
            scaler=scaler,
            forecast_horizon=1,
            device=device,
        )[0]
        preds.append(pred_close)
        history = np.vstack([history, test_scaled[i : i + 1]])
    return np.array(preds, dtype=float)


def moving_average_baseline(close_series: pd.Series, horizon: int, window: int = 5) -> np.ndarray:
    """
    Rolling moving-average forecast: iteratively append the predicted value so
    later steps can drift instead of staying flat.
    """
    history = _clean_close_series(close_series).tolist()
    window = max(1, min(window, len(history)))
    preds = []
    for _ in range(horizon):
        window_values = np.asarray(history[-window:], dtype=float)
        next_val = float(window_values.mean())
        preds.append(next_val)
        history.append(next_val)
    return np.array(preds, dtype=float)


def arima_baseline(close_series: pd.Series, horizon: int, order=(5, 1, 0)) -> np.ndarray:
    """
    Rolling ARIMA forecast: refit one-step-ahead each iteration so the mean
    path can evolve instead of collapsing to a flat line.
    """
    values = _clean_close_series(close_series)
    if len(values) < sum(order) + 1:
        raise ValueError("ARIMA baseline received too few valid Close values after cleaning.")
    history = values.tolist()
    preds = []
    for _ in range(horizon):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(history, order=order, trend="t", enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
        next_val = float(fitted.forecast(steps=1)[0])
        preds.append(next_val)
        history.append(next_val)
    return np.array(preds, dtype=float)


def garch_baseline(close_series: pd.Series, horizon: int) -> np.ndarray:
    """
    Fit a GARCH(1,1) model on returns and roll forecasts forward, updating the
    mean path each step.
    """
    clean_close = pd.Series(_clean_close_series(close_series))
    returns = clean_close.pct_change().dropna()
    if len(returns) < 30:
        return np.full(horizon, clean_close.iloc[-1], dtype=float)
    scaled_returns = (returns * 100.0).reset_index(drop=True)
    preds = []
    current_price = float(clean_close.iloc[-1])
    history = scaled_returns
    for _ in range(horizon):
        am = arch_model(history, vol="Garch", p=1, o=0, q=1, mean="AR", lags=1, dist="normal")
        res = am.fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        mean_ret = float(forecast.mean.values[-1, 0]) / 100.0
        current_price *= (1.0 + mean_ret)
        preds.append(current_price)
        history = pd.concat([history, pd.Series([mean_ret * 100.0])], ignore_index=True)
    return np.array(preds, dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, MSE, RMSE, and MAPE for the given arrays."""
    if len(y_true) != len(y_pred):
        horizon = min(len(y_true), len(y_pred))
        y_true = y_true[:horizon]
        y_pred = y_pred[:horizon]
    eps = 1e-8
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE%": mape}


def plot_forecasts(
    full_dates: pd.DatetimeIndex,
    full_actual: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    horizon_forecasts: dict[int, dict[str, np.ndarray]],
    plot_start: pd.Timestamp | None = None,
) -> None:
    """
    Plot multi-panel forecasts: actual (2024+2025) vs forecasts for each horizon.
    Each horizon's forecasts are aligned to forecast_dates.
    """
    if plot_start is not None:
        mask = full_dates >= plot_start
        full_dates = full_dates[mask]
        full_actual = full_actual[mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for ax, horizon in zip(axes, EVAL_HORIZONS):
        ax.plot(full_dates, full_actual, label="Actual Close", linewidth=1.8, color="black")
        forecasts = horizon_forecasts[horizon]
        for name, pred in forecasts.items():
            pred_series = pd.Series(pred, index=forecast_dates)
            aligned = pred_series.reindex(full_dates)
            ax.plot(full_dates, aligned, label=name, alpha=0.9)
        ax.set_title(f"Forecast horizon: next {horizon} day(s)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_ylabel("Close Price")
    axes[-2].set_xlabel("Date")
    axes[-1].set_xlabel("Date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle("S&P 500 multi-horizon forecasts into 2025 (with 2024 history)", y=0.975)
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forecast S&P500 with Falcon-TST and baseline models.")
    parser.add_argument(
        "--falcon_model",
        type=str,
        default="ant-intl/Falcon-TST_Large",
        help="Hugging Face model identifier for Falcon-TST.",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=5,
        help="Window size for the moving average baseline.",
    )
    parser.add_argument(
        "--arima_order",
        type=int,
        nargs=3,
        default=(5, 1, 0),
        metavar=("P", "D", "Q"),
        help="ARIMA order for the ARIMA baseline.",
    )
    parser.add_argument(
        "--baseline_window_days",
        type=int,
        default=365,
        help="Use only the last N days of training closes for baselines (keeps them responsive to recent trends).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device to run Falcon-TST on. 'auto' picks the best available.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    today = date.today()
    forecast_end = min(today, date(2025, 12, 31))
    if forecast_end < date(2025, 1, 1):
        raise ValueError("There is no available 2025 data yet to evaluate.")

    df = fetch_sp500_data(forecast_end)
    split = train_test_split(df[FEATURE_COLUMNS], forecast_end)
    scaler, train_scaled, test_scaled = standardize_features(split.train, split.test)
    forecast_horizon = len(split.test)

    print(f"Loaded {len(split.train)} training points (pre-2025) and {forecast_horizon} test points (2025).")
    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading Falcon-TST model...")
    falcon_model = AutoModel.from_pretrained(
        args.falcon_model,
        trust_remote_code=True,
    ).to(device)

    # Baselines: restrict to most recent window so they reflect last-year trends
    forecast_start = split.test.index.min()
    baseline_cutoff = forecast_start - pd.Timedelta(days=args.baseline_window_days)
    baseline_train_close = split.train.loc[split.train.index >= baseline_cutoff, "Close"]
    if baseline_train_close.empty:
        baseline_train_close = split.train["Close"]

    actual_close = split.test["Close"].to_numpy()
    horizon_forecasts: dict[int, dict[str, np.ndarray]] = {}

    print("\nRunning multi-horizon chunked forecasts...")
    for horizon in EVAL_HORIZONS:
        print(f"- horizon {horizon} day(s)")
        falcon_preds = chunked_falcon_forecast(
            falcon_model, scaler, train_scaled, test_scaled, horizon, device
        )
        ma_preds = chunked_ma_forecast(baseline_train_close, split.test["Close"], horizon, window=args.ma_window)
        arima_preds = chunked_arima_forecast(baseline_train_close, split.test["Close"], horizon, order=tuple(args.arima_order))
        garch_preds = chunked_garch_forecast(baseline_train_close, split.test["Close"], horizon)

        horizon_forecasts[horizon] = {
            "Falcon-TST": falcon_preds,
            "Moving Average": ma_preds,
            "ARIMA": arima_preds,
            "GARCH": garch_preds,
        }

        print("  Error metrics vs actual 2025 closes:")
        for name, pred in horizon_forecasts[horizon].items():
            metrics = compute_metrics(actual_close, pred)
            metric_str = ", ".join(f"{k}: {v:,.4f}" for k, v in metrics.items())
            print(f"    {name:<15} {metric_str}")

    full_actual = pd.concat([split.train["Close"], split.test["Close"]])
    plot_forecasts(
        full_actual.index,
        full_actual.to_numpy(),
        split.test.index,
        horizon_forecasts,
        plot_start=pd.Timestamp("2024-01-01"),
    )
    print(f"\nSaved forecast comparison plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
