"""
backend/backtesting.py
Multi-origin rolling-origin recursive backtesting framework.

Evaluates true production inference conditions:
- At each historical origin O, fits pipelines strictly on data with Date <= O.
- Rolls forward day-by-day autoregressively for steps 1..max_horizon using shared features
  from `backend/features.py` (never peeking at actual future demand or future unit prices).
- Reports aggregated metrics separately by forecast horizon (7, 14, 30 days),
  per-SKU error breakdowns, and direct comparison against recursive Seasonal Naive baselines.
"""

from datetime import timedelta
from typing import Dict, Any, List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.base import clone

from backend.features import (
    FEATURE_COLUMNS,
    TARGET,
    extract_product_forecast_state,
    build_feature_row_from_state,
    advance_product_forecast_state,
)
from backend.baselines import compute_regression_metrics


def _build_actual_lookup(featured_df: pd.DataFrame) -> Dict[tuple, float]:
    """Maps (StockCode, pd.Timestamp(Date)) -> DailyDemand for O(1) evaluation lookup."""
    lookup: Dict[tuple, float] = {}
    dates = pd.to_datetime(featured_df['Date'])
    for sku, dt_val, y_val in zip(featured_df['StockCode'], dates, featured_df[TARGET]):
        lookup[(str(sku), pd.Timestamp(dt_val))] = float(y_val)
    return lookup


def select_rolling_origins(
    featured_df: pd.DataFrame,
    cutoff_date: str = "2011-11-01",
    n_origins: int = 4,
    max_horizon: int = 30,
    min_train_days: int = 120,
) -> List[pd.Timestamp]:
    """
    Selects evenly spaced historical forecast origins strictly prior to `cutoff_date`
    such that `origin + max_horizon` stays strictly `< cutoff_date` and every origin has
    at least `min_train_days` of historical dates before it.
    """
    dates = pd.to_datetime(featured_df['Date'])
    min_date = dates.min()
    cutoff_ts = pd.to_datetime(cutoff_date)

    earliest_origin = min_date + timedelta(days=min_train_days)
    latest_origin = cutoff_ts - timedelta(days=max_horizon + 1)

    if earliest_origin >= latest_origin:
        # Fallback for short synthetic test datasets: space inside available span
        span_days = int((cutoff_ts - min_date).days)
        if span_days < max_horizon + 35:
            raise ValueError(
                f"Insufficient date span ({span_days} days) for rolling-origin backtesting "
                f"with max_horizon={max_horizon}."
            )
        earliest_origin = min_date + timedelta(days=35)
        latest_origin = cutoff_ts - timedelta(days=max_horizon + 1)

    total_window = (latest_origin - earliest_origin).days
    if n_origins <= 1 or total_window == 0:
        return [earliest_origin]

    step_days = np.linspace(0, total_window, n_origins, dtype=int)
    origins = [earliest_origin + timedelta(days=int(d)) for d in step_days]
    # Remove any duplicates if window was small
    unique_origins = sorted(list(dict.fromkeys(origins)))
    return unique_origins


def evaluate_recursive_at_origin(
    fitted_pipeline,
    featured_df: pd.DataFrame,
    origin_date: pd.Timestamp,
    max_horizon: int = 30,
    actual_lookup: Optional[Dict[tuple, float]] = None,
) -> pd.DataFrame:
    """
    Runs a true recursive multi-step forecast across all eligible SKUs starting right after
    `origin_date` (steps 1..max_horizon) using an ALREADY fitted pipeline trained on Date <= origin_date.
    """
    if actual_lookup is None:
        actual_lookup = _build_actual_lookup(featured_df)

    df_hist_all = featured_df.copy()
    df_hist_all['Date'] = pd.to_datetime(df_hist_all['Date'])
    hist_up_to_origin = df_hist_all[df_hist_all['Date'] <= origin_date]

    # Build initial per-SKU causal forecast state as of origin_date
    sku_states: List[Dict[str, Any]] = []
    for sku, p_df in hist_up_to_origin.groupby('StockCode'):
        p_sorted = p_df.sort_values('Date')
        if len(p_sorted) < 30:
            continue
        # Only roll forward SKUs active at or immediately up to origin_date
        if pd.Timestamp(p_sorted['Date'].iloc[-1]) != origin_date:
            continue
        st = extract_product_forecast_state(p_sorted)
        st['snaive_series'] = list(st['demand_series'])  # independent Seasonal Naive buffer
        sku_states.append(st)

    if not sku_states:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for step in range(1, max_horizon + 1):
        target_date = origin_date + timedelta(days=step)
        feature_rows = []
        active_states = []

        for st in sku_states:
            if (st['stock_code'], target_date) not in actual_lookup:
                continue
            feature_rows.append(build_feature_row_from_state(st, target_date))
            active_states.append(st)

        if not feature_rows:
            break

        batch_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
        preds_ml = np.clip(fitted_pipeline.predict(batch_df), 0.0, None)

        for st, pred_val in zip(active_states, preds_ml):
            pred_ml_val = float(pred_val)
            pred_snaive_val = max(0.0, float(st['snaive_series'][-7]))
            y_true_val = actual_lookup[(st['stock_code'], target_date)]

            records.append({
                'origin': str(origin_date.date()),
                'StockCode': st['stock_code'],
                'Date': target_date,
                'horizon_step': step,
                'y_true': y_true_val,
                'y_pred_ml': pred_ml_val,
                'y_pred_snaive': pred_snaive_val,
            })

            # Advance both ML recursive buffer and Seasonal Naive buffer independently
            advance_product_forecast_state(st, pred_ml_val)
            st['snaive_series'].append(pred_snaive_val)

    return pd.DataFrame(records)


def run_rolling_origin_backtest(
    pipeline_template,
    featured_df: pd.DataFrame,
    cutoff_date: str = "2011-11-01",
    n_origins: int = 4,
    horizons: Sequence[int] = (7, 14, 30),
    origins: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Executes multi-origin recursive rolling-origin backtesting strictly prior to `cutoff_date`.
    Returns horizon-level comparisons (ML vs Seasonal Naive), origin logs, and per-SKU errors.
    """
    df = featured_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    max_horizon = max(horizons)

    if origins is not None:
        origin_dates = [pd.to_datetime(o) for o in origins]
    else:
        origin_dates = select_rolling_origins(
            df, cutoff_date=cutoff_date, n_origins=n_origins, max_horizon=max_horizon
        )

    actual_lookup = _build_actual_lookup(df)
    all_preds: List[pd.DataFrame] = []
    origin_summaries: List[Dict[str, Any]] = []

    for origin_dt in origin_dates:
        train_sub = df[df['Date'] <= origin_dt]
        if train_sub.empty:
            continue

        model_o = clone(pipeline_template)
        model_o.fit(train_sub[FEATURE_COLUMNS], train_sub[TARGET])

        preds_df = evaluate_recursive_at_origin(
            fitted_pipeline=model_o,
            featured_df=df,
            origin_date=origin_dt,
            max_horizon=max_horizon,
            actual_lookup=actual_lookup,
        )
        if not preds_df.empty:
            all_preds.append(preds_df)
            origin_m_14 = compute_regression_metrics(
                preds_df[preds_df['horizon_step'] <= 14]['y_true'],
                preds_df[preds_df['horizon_step'] <= 14]['y_pred_ml'],
            )
            origin_sn_14 = compute_regression_metrics(
                preds_df[preds_df['horizon_step'] <= 14]['y_true'],
                preds_df[preds_df['horizon_step'] <= 14]['y_pred_snaive'],
            )
            origin_summaries.append({
                "origin_date": str(origin_dt.date()),
                "train_samples": int(len(train_sub)),
                "skus_evaluated": int(preds_df['StockCode'].nunique()),
                "ml_14d_mae": origin_m_14["MAE"],
                "ml_14d_rmse": origin_m_14["RMSE"],
                "snaive_14d_mae": origin_sn_14["MAE"],
                "snaive_14d_rmse": origin_sn_14["RMSE"],
            })

    if not all_preds:
        raise RuntimeError("Rolling-origin backtesting produced no evaluation predictions.")

    full_preds = pd.concat(all_preds, ignore_index=True)

    # 1. Cumulative horizon metrics (1..7, 1..14, 1..30) & point-step metrics
    horizon_metrics: Dict[str, Dict[str, Any]] = {}
    for h in sorted(horizons):
        sub_h = full_preds[full_preds['horizon_step'] <= h]
        ml_m = compute_regression_metrics(sub_h['y_true'], sub_h['y_pred_ml'])
        sn_m = compute_regression_metrics(sub_h['y_true'], sub_h['y_pred_snaive'])
        horizon_metrics[f"{h}d"] = {
            "horizon_days": int(h),
            "samples": int(len(sub_h)),
            "ml": ml_m,
            "seasonal_naive": sn_m,
            "beats_seasonal_naive_rmse": bool(ml_m["RMSE"] < sn_m["RMSE"]),
            "beats_seasonal_naive_mae": bool(ml_m["MAE"] < sn_m["MAE"]),
        }

    # 2. Per-product aggregated error metrics across the backtesting window (1..max_horizon)
    product_rows: List[Dict[str, Any]] = []
    for sku, g in full_preds.groupby('StockCode'):
        m_ml = compute_regression_metrics(g['y_true'], g['y_pred_ml'])
        m_sn = compute_regression_metrics(g['y_true'], g['y_pred_snaive'])
        product_rows.append({
            "StockCode": str(sku),
            "samples": int(len(g)),
            "actual_mean": round(float(g['y_true'].mean()), 2),
            "pred_mean": round(float(g['y_pred_ml'].mean()), 2),
            "MAE": m_ml["MAE"],
            "RMSE": m_ml["RMSE"],
            "R2": m_ml["R2"],
            "SeasonalNaive_MAE": m_sn["MAE"],
            "SeasonalNaive_RMSE": m_sn["RMSE"],
            "beats_seasonal_naive": bool(m_ml["RMSE"] < m_sn["RMSE"]),
        })

    product_df = pd.DataFrame(product_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    beats_count = int(product_df["beats_seasonal_naive"].sum()) if not product_df.empty else 0
    total_skus = int(len(product_df))

    return {
        "origins": [str(o.date()) for o in origin_dates],
        "origin_details": origin_summaries,
        "horizon_metrics": horizon_metrics,
        "product_wise_summary": {
            "total_skus_evaluated": total_skus,
            "skus_beating_seasonal_naive_rmse": beats_count,
            "win_rate_pct": round((beats_count / total_skus) * 100.0, 1) if total_skus > 0 else 0.0,
            "median_sku_mae": round(float(product_df["MAE"].median()), 4) if not product_df.empty else 0.0,
            "median_sku_rmse": round(float(product_df["RMSE"].median()), 4) if not product_df.empty else 0.0,
            "best_skus": product_df.head(8).to_dict(orient="records"),
            "highest_error_skus": product_df.sort_values("RMSE", ascending=False).head(8).to_dict(orient="records"),
            "all_skus": product_df.to_dict(orient="records"),
        },
    }
