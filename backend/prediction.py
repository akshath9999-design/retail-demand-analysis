"""
backend/prediction.py
Multi-step iterative demand forecasting engine using trained ML pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Any, List, Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

class DemandForecaster:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = os.path.abspath(models_dir)
        self.model = None
        self.products_catalog = {}
        self.featured_df = None
        self.metadata = {}
        self.load_resources()

    def load_resources(self):
        """Loads trained model, product catalog, and historical feature frame."""
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        catalog_path = os.path.join(self.models_dir, "products_catalog.json")
        featured_path = os.path.join(self.models_dir, "featured_demand.csv.gz")
        meta_path = os.path.join(self.models_dir, "model_metadata.json")

        if os.path.exists(best_model_path):
            self.model = joblib.load(best_model_path)
            print(f"[FORECASTER] Loaded model from {best_model_path}")
        else:
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
            print(f"[FORECASTER] Loaded historical series for fast autoregressive inference.")

    def get_product_info(self, stock_code: str) -> Optional[Dict[str, Any]]:
        stock_code = stock_code.strip().upper()
        return self.products_catalog.get(stock_code)

    def forecast(
        self,
        stock_code: str,
        forecast_days: int = 14
    ) -> Dict[str, Any]:
        """
        Generates an autoregressive multi-step demand forecast for the specified product.
        Uses the last known historical state and rolls forward day by day.
        """
        stock_code = stock_code.strip().upper()
        prod_info = self.get_product_info(stock_code)
        if not prod_info:
            raise ValueError(f"Product '{stock_code}' not found in active product catalog.")

        if self.featured_df is None or self.model is None:
            raise RuntimeError("Forecaster model or historical data not loaded.")

        # Get the product's historical series
        p_history = self.featured_df[self.featured_df['StockCode'] == stock_code].sort_values('Date')
        if p_history.empty:
            raise ValueError(f"No historical time-series available for product '{stock_code}'.")

        # Extract last 45 days of history for lag creation
        recent_history = p_history.tail(45).copy()
        last_date = recent_history['Date'].max()
        avg_price = float(prod_info.get('AvgPrice', 2.5))
        desc = prod_info.get('Description', 'RETAIL ITEM')

        # Recent daily demand series
        demand_series = recent_history['DailyDemand'].tolist()
        
        predictions = []
        current_date = last_date

        for step in range(1, forecast_days + 1):
            next_date = current_date + timedelta(days=step)
            
            # Construct features strictly from past values
            lag_1 = demand_series[-1] if len(demand_series) >= 1 else 0.0
            lag_7 = demand_series[-7] if len(demand_series) >= 7 else demand_series[-1]
            lag_14 = demand_series[-14] if len(demand_series) >= 14 else lag_7
            lag_30 = demand_series[-30] if len(demand_series) >= 30 else lag_14

            rolling_mean_7 = float(np.mean(demand_series[-7:])) if len(demand_series) >= 7 else float(np.mean(demand_series))
            rolling_mean_14 = float(np.mean(demand_series[-14:])) if len(demand_series) >= 14 else rolling_mean_7
            rolling_mean_30 = float(np.mean(demand_series[-30:])) if len(demand_series) >= 30 else rolling_mean_14
            rolling_std_7 = float(np.std(demand_series[-7:])) if len(demand_series) >= 7 else 0.0
            rolling_trend = (rolling_mean_7 + 0.1) / (rolling_mean_30 + 0.1)
            cv_7 = rolling_std_7 / (rolling_mean_7 + 1.0)

            # Build single-row feature dataframe
            feature_row = pd.DataFrame([{
                'StockCode': stock_code,
                'Date': next_date,
                'day': next_date.day,
                'month': next_date.month,
                'year': next_date.year,
                'dayofweek': next_date.dayofweek,
                'is_weekend': 1 if next_date.dayofweek in [5, 6] else 0,
                'weekofyear': int(next_date.isocalendar()[1]),
                'lag_1': lag_1,
                'lag_7': lag_7,
                'lag_14': lag_14,
                'lag_30': lag_30,
                'rolling_mean_7': rolling_mean_7,
                'rolling_mean_14': rolling_mean_14,
                'rolling_mean_30': rolling_mean_30,
                'rolling_std_7': rolling_std_7,
                'rolling_trend': rolling_trend,
                'cv_7': cv_7,
                'AvgUnitPrice': avg_price
            }])

            # Predict demand
            raw_pred = float(self.model.predict(feature_row)[0])
            pred_demand = max(0.0, round(raw_pred, 1))

            predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "day_name": next_date.strftime("%a"),
                "predicted_demand": pred_demand
            })

            # Append prediction to history to roll forward autoregressively
            demand_series.append(pred_demand)

        # Calculate forecast aggregate statistics
        pred_values = [p['predicted_demand'] for p in predictions]
        total_forecast_demand = round(sum(pred_values), 1)
        avg_forecast_demand = round(float(np.mean(pred_values)), 1)
        max_forecast_demand = round(float(np.max(pred_values)), 1)
        min_forecast_demand = round(float(np.min(pred_values)), 1)

        # Baseline recent 30-day historical points for charting
        history_points = [
            {
                "date": row['Date'].strftime("%Y-%m-%d"),
                "actual_demand": round(float(row['DailyDemand']), 1)
            }
            for _, row in recent_history.tail(30).iterrows()
        ]

        # Recent historical average
        recent_avg_demand = round(float(recent_history['DailyDemand'].tail(30).mean()), 1)
        
        # Momentum / trend direction
        trend_direction = "INCREASING" if avg_forecast_demand > recent_avg_demand * 1.05 else (
            "DECREASING" if avg_forecast_demand < recent_avg_demand * 0.95 else "STABLE"
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
            "historical_baseline": history_points,
            "forecast_schedule": predictions
        }

# Global singleton forecaster
forecaster = DemandForecaster()
