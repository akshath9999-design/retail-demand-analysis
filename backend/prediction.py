"""
backend/prediction.py
Multi-step autoregressive demand forecasting engine using the selected trained ML pipeline.

Guarantees:
- Uses `backend/features.py` so feature names, order, sample ddof=1, and historical logic
  match the training pipeline identically.
- Loads `best_model.pkl` (verified to match the selected winning model in `model_metadata.json`).
- Carries forward the last observed historical unit price unless explicit future prices are supplied.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Any, List, Optional, Sequence

from backend.features import (
    FEATURE_COLUMNS,
    extract_product_forecast_state,
    build_feature_row_from_state,
    advance_product_forecast_state,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DemandForecaster:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = os.path.abspath(models_dir)
        self.model = None
        self.products_catalog: Dict[str, Dict[str, Any]] = {}
        self.featured_df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.load_resources()

    def load_resources(self):
        """Loads selected best model, product catalog, model metadata, and historical feature frame."""
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        catalog_path = os.path.join(self.models_dir, "products_catalog.json")
        featured_path = os.path.join(self.models_dir, "featured_demand.csv.gz")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if os.path.exists(best_model_path):
            self.model = joblib.load(best_model_path)
            print(f"[FORECASTER] Loaded model from {best_model_path}")
        else:
            self.model = None
            print(f"[FORECASTER] Warning: Model not found at {best_model_path}")

        if os.path.exists(catalog_path):
            with open(catalog_path, "r") as f:
                cat_list = json.load(f)
                self.products_catalog = {p['StockCode']: p for p in cat_list}
            print(f"[FORECASTER] Loaded {len(self.products_catalog)} products into catalog.")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        if os.path.exists(featured_path):
            self.featured_df = pd.read_csv(featured_path)
            self.featured_df['Date'] = pd.to_datetime(self.featured_df['Date'])
            print("[FORECASTER] Loaded historical series for autoregressive inference.")

    def get_product_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        stock_code = stock_code.strip().upper()
        return self.products_catalog.get(stock_code)

    def forecast(
        self,
        stock_code: str,
        forecast_days: int = 14,
        future_prices: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generates an autoregressive multi-step demand forecast for the specified product.

        Assumptions:
        - Recursive demand lags & rolling stats update strictly from past actuals and previously
          generated non-negative predictions.
        - Future unit prices assume constant carry-forward from the most recent historical price
          unless `future_prices` is explicitly supplied.
        """
        if forecast_days < 1:
            raise ValueError("forecast_days must be at least 1.")

        stock_code = stock_code.strip().upper()
        prod_info = self.get_product_info(stock_code)
        if not prod_info:
            raise ValueError(f"Product '{stock_code}' not found in active product catalog.")

        if self.featured_df is None or self.model is None:
            raise RuntimeError("Forecaster model or historical data not loaded.")

        p_history = self.featured_df[self.featured_df['StockCode'] == stock_code].sort_values('Date')
        if p_history.empty:
            raise ValueError(f"No historical time-series available for product '{stock_code}'.")

        # Gracefully pad ultra-short histories (< 30 rows) if ever encountered outside normal training SKUs
        if len(p_history) < 30:
            pad_len = 30 - len(p_history)
            first_row = p_history.iloc[[0]].copy()
            pad_rows = pd.concat([first_row] * pad_len, ignore_index=True)
            pad_rows['DailyDemand'] = 0.0
            p_history_ready = pd.concat([pad_rows, p_history], ignore_index=True)
        else:
            p_history_ready = p_history

        state = extract_product_forecast_state(p_history_ready)
        last_date = state['last_date']
        desc = prod_info.get('Description', 'RETAIL ITEM')
        price_mode = "supplied_future_prices" if future_prices is not None else "carry_forward_last_historical_price"

        predictions = []
        for step in range(1, forecast_days + 1):
            next_date = last_date + timedelta(days=step)

            if future_prices is not None and len(future_prices) > 0:
                p_idx = min(step - 1, len(future_prices) - 1)
                state['last_unit_price'] = float(future_prices[p_idx])

            row_dict = build_feature_row_from_state(state, next_date)
            feature_row = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

            raw_pred = float(self.model.predict(feature_row)[0])
            pred_demand = max(0.0, round(raw_pred, 1))

            predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "day_name": next_date.strftime("%a"),
                "predicted_demand": pred_demand,
            })

            advance_product_forecast_state(state, pred_demand)

        pred_values = [p['predicted_demand'] for p in predictions]
        total_forecast_demand = round(sum(pred_values), 1)
        avg_forecast_demand = round(float(np.mean(pred_values)), 1)
        max_forecast_demand = round(float(np.max(pred_values)), 1)
        min_forecast_demand = round(float(np.min(pred_values)), 1)

        recent_tail = p_history.tail(30)
        history_points = [
            {
                "date": row['Date'].strftime("%Y-%m-%d"),
                "actual_demand": round(float(row['DailyDemand']), 1),
            }
            for _, row in recent_tail.iterrows()
        ]

        recent_avg_demand = round(float(recent_tail['DailyDemand'].mean()), 1)
        trend_direction = (
            "INCREASING"
            if avg_forecast_demand > recent_avg_demand * 1.05
            else ("DECREASING" if avg_forecast_demand < recent_avg_demand * 0.95 else "STABLE")
        )

        return {
            "stock_code": stock_code,
            "description": desc,
            "forecast_days": forecast_days,
            "total_forecast_demand": total_forecast_demand,
            "avg_daily_demand": avg_forecast_demand,
            "peak_daily_demand": max_forecast_demand,
            "min_daily_demand": min_forecast_demand,
            "recent_historical_avg": recent_avg_demand,
            "trend_direction": trend_direction,
            "model_used": self.metadata.get("best_model", type(self.model).__name__),
            "price_assumption": price_mode,
            "assumed_unit_price": round(float(state['last_unit_price']), 4),
            "historical_baseline": history_points,
            "forecast_schedule": predictions,
        }


# Global singleton forecaster
forecaster = DemandForecaster()
