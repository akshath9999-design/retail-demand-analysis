"""
tests/test_api.py
Automated test suite for Retail Product Demand Estimation and Stockout Risk Analysis.
Tests:
  1. Dataset loading tests
  2. Data preprocessing tests
  3. Feature engineering tests
  4. Model training and artifact existence
  5. Model prediction tests
  6. Model evaluation tests
  7. API endpoint status tests
  8. Stockout risk calculation tests (HIGH and LOW risk rules)
  9. Invalid input and error handling tests (404 and 422)
  10. Frontend static assets tests
"""

import os
import json
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.prediction import forecaster
from backend.stockout import analyze_stockout_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_retail_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

client = TestClient(app)

# 1. Dataset Loading Tests
def test_cleaned_dataset_exists_and_valid():
    assert os.path.exists(CLEANED_DATA_PATH), "Cleaned dataset CSV must exist."
    df = pd.read_csv(CLEANED_DATA_PATH, nrows=50)
    expected_cols = {'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'Revenue'}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in cleaned dataset. Found: {df.columns}"
    assert len(df) > 0, "Cleaned dataset must contain records."

# 2. Data Preprocessing Tests
def test_preprocessing_no_cancellations_or_negatives():
    df = pd.read_csv(CLEANED_DATA_PATH, nrows=2000)
    assert not df['InvoiceNo'].astype(str).str.startswith('C').any(), "Cleaned dataset must not contain cancellations."
    assert (df['Quantity'] > 0).all(), "Cleaned dataset must have strictly positive quantities."
    assert (df['UnitPrice'] > 0).all(), "Cleaned dataset must have strictly positive unit prices."

# 3. Feature Engineering Tests
def test_feature_matrix_integrity():
    feat_path = os.path.join(MODELS_DIR, "featured_demand.csv.gz")
    assert os.path.exists(feat_path), "Engineered feature matrix file must exist."
    df = pd.read_csv(feat_path, nrows=100)
    required_features = ['day', 'month', 'dayofweek', 'lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7', 'rolling_trend', 'DailyDemand']
    for feat in required_features:
        assert feat in df.columns, f"Engineered feature {feat} missing."
    # Ensure no NaN in lag/rolling columns
    assert not df['lag_1'].isnull().any(), "Lag 1 feature contains unexpected NaNs."
    assert not df['rolling_mean_7'].isnull().any(), "Rolling mean 7 contains unexpected NaNs."

# 4. Model Training & Serialization Tests
def test_all_models_serialized():
    model_files = [
        "linear_regression.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl",
        "best_model.pkl",
        "products_catalog.json",
        "model_metadata.json"
    ]
    for mf in model_files:
        path = os.path.join(MODELS_DIR, mf)
        assert os.path.exists(path), f"Required model artifact {mf} does not exist."
        assert os.path.getsize(path) > 0, f"Model file {mf} is empty."

# 5. Model Prediction Engine Tests
def test_forecaster_multi_step_prediction():
    res = forecaster.forecast("85123A", forecast_days=7)
    assert res is not None
    assert res["stock_code"] == "85123A"
    assert res["forecast_days"] == 7
    assert len(res["forecast_schedule"]) == 7
    assert res["total_forecast_demand"] >= 0
    assert res["avg_daily_demand"] >= 0
    assert res["trend_direction"] in ["INCREASING", "DECREASING", "STABLE"]

# 6. Model Evaluation Tests
def test_model_metrics_validity():
    metrics_path = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    assert os.path.exists(metrics_path), "Model metrics JSON must exist."
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "metrics" in metrics
    for model_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        assert model_name in metrics["metrics"]
        m = metrics["metrics"][model_name]
        assert "MAE" in m and m["MAE"] > 0
        assert "RMSE" in m and m["RMSE"] > 0
        assert "R2" in m

# 7. Stockout Risk Logic Tests
def test_stockout_risk_high_deficit():
    # If inventory is extremely small compared to 10-day demand, risk must be HIGH
    res = analyze_stockout_risk("85123A", current_inventory=5, supplier_lead_time=10)
    assert res["stockout_status"] == "HIGH"
    assert res["suggested_replenishment"] > 0
    assert res["risk_color"] == "#dc3545"

def test_stockout_risk_low_surplus():
    # If inventory is massive (50,000 units), risk must be LOW
    res = analyze_stockout_risk("85123A", current_inventory=50000, supplier_lead_time=10)
    assert res["stockout_status"] == "LOW"
    assert res["suggested_replenishment"] == 0
    assert res["risk_color"] == "#28a745"

# 8. FastAPI API Endpoints Tests
def test_api_status_endpoint():
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_api_dashboard_endpoint():
    res = client.get("/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert data["total_products"] > 0
    assert data["total_transactions"] > 0
    assert len(data["best_selling_products"]) > 0

def test_api_products_endpoint():
    res = client.get("/products?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["returned"] == 10
    assert len(data["products"]) == 10

def test_api_sales_trends_endpoint():
    res = client.get("/sales-trends")
    assert res.status_code == 200
    data = res.json()
    assert "monthly_sales" in data
    assert "day_of_week" in data
    assert "recent_daily_trajectory" in data

def test_api_predict_post():
    payload = {"product_id": "85123A", "forecast_days": 14}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["data"]["forecast_schedule"]) == 14

def test_api_stockout_post():
    payload = {
        "product_id": "85123A",
        "available_inventory": 20,
        "supplier_lead_time": 7,
        "safety_buffer_pct": 0.20
    }
    res = client.post("/stockout-risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["data"]["stockout_status"] in ["HIGH", "LOW"]

# 9. Error Handling & Validation Tests
def test_api_invalid_product_404():
    payload = {"product_id": "NON_EXISTENT_SKU_XYZ", "forecast_days": 14}
    res = client.post("/predict", json=payload)
    assert res.status_code == 404

def test_api_invalid_forecast_days_422():
    payload = {"product_id": "85123A", "forecast_days": -5}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422

def test_api_invalid_inventory_422():
    payload = {"product_id": "85123A", "available_inventory": -10, "supplier_lead_time": 5}
    res = client.post("/stockout-risk", json=payload)
    assert res.status_code == 422

# 10. Frontend Static Assets Tests
def test_frontend_static_assets_exist():
    assert os.path.exists(os.path.join(BASE_DIR, "frontend", "index.html"))
    assert os.path.exists(os.path.join(BASE_DIR, "frontend", "style.css"))
    assert os.path.exists(os.path.join(BASE_DIR, "frontend", "script.js"))
    # Test GET / serves the HTML
    res = client.get("/")
    assert res.status_code == 200
    assert "DemandIQ" in res.text
