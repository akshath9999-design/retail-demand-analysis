"""
backend/baselines.py
Simple, leakage-free baseline forecasters (Naive and Seasonal Naive) for rigorous
benchmarking against machine learning regression pipelines.
"""

from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Calculates standard regression evaluation metrics (MAE, MSE, RMSE, R2)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p)) if len(y_t) >= 2 and np.var(y_t) > 0 else 0.0
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def recursive_naive_forecast(demand_history: Sequence[float], forecast_days: int) -> List[float]:
    """
    Multi-step Naive baseline:
      Predicted demand at T+h equals previous available demand (recursive lag_1 -> carries y_T forward).
    """
    if len(demand_history) < 1:
        raise ValueError("Naive baseline requires at least 1 day of demand history.")
    last_val = max(0.0, float(demand_history[-1]))
    return [last_val for _ in range(forecast_days)]


def recursive_seasonal_naive_forecast(demand_history: Sequence[float], forecast_days: int) -> List[float]:
    """
    Multi-step Seasonal Naive baseline (weekly period = 7):
      Predicted demand equals demand from the same weekday one week earlier (recursive lag_7).
    """
    if len(demand_history) < 7:
        raise ValueError("Seasonal Naive baseline requires at least 7 days of demand history.")
    series = [float(x) for x in demand_history]
    preds: List[float] = []
    for _ in range(forecast_days):
        pred = max(0.0, float(series[-7]))
        preds.append(pred)
        series.append(pred)
    return preds


def evaluate_single_step_baselines(test_df: pd.DataFrame, target_col: str = "DailyDemand") -> Dict[str, Dict[str, float]]:
    """
    Evaluates one-step-ahead Naive (lag_1) and Seasonal Naive (lag_7) baselines
    on a single-step evaluation dataframe containing lag_1 and lag_7.
    """
    y_true = test_df[target_col].to_numpy(dtype=float)
    naive_pred = np.clip(test_df["lag_1"].to_numpy(dtype=float), 0.0, None)
    seasonal_naive_pred = np.clip(test_df["lag_7"].to_numpy(dtype=float), 0.0, None)

    return {
        "Naive Baseline (t-1)": compute_regression_metrics(y_true, naive_pred),
        "Seasonal Naive Baseline (t-7)": compute_regression_metrics(y_true, seasonal_naive_pred),
    }
