"""
backend/main.py
FastAPI application for Retail Product Demand Estimation and Stockout Risk Analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.prediction import forecaster
from backend.stockout import analyze_stockout_risk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_retail_data.csv")

app = FastAPI(
    title="Retail Product Demand Estimation & Stockout Risk API",
    description="Machine Learning REST API for retail demand forecasting, inventory optimization, and stockout risk evaluation.",
    version="1.0.0"
)

# Enable CORS for full frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if os.path.exists(GRAPHS_DIR):
    app.mount("/graphs", StaticFiles(directory=GRAPHS_DIR), name="graphs")

# ----------------- PYDANTIC SCHEMAS -----------------

class PredictRequest(BaseModel):
    product_id: str = Field(..., description="Stock code of the product (e.g. '85123A')")
    forecast_days: int = Field(default=14, ge=1, le=90, description="Forecasting period in days (1 to 90)")

class StockoutRequest(BaseModel):
    product_id: str = Field(..., description="Stock code of the product")
    available_inventory: int = Field(..., ge=0, description="Current on-hand inventory units")
    supplier_lead_time: int = Field(..., ge=1, le=180, description="Supplier lead time in days")
    safety_buffer_pct: Optional[float] = Field(default=0.20, ge=0.0, le=1.0, description="Safety stock buffer fraction (default: 0.20)")


# ----------------- CACHED DASHBOARD DATA -----------------
_dashboard_cache = None

def get_dashboard_summary():
    global _dashboard_cache
    if _dashboard_cache is not None:
        return _dashboard_cache

    if not os.path.exists(CLEANED_DATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cleaned dataset not available. Please run data pipeline first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    total_transactions = int(df['InvoiceNo'].nunique())
    total_sales_quantity = int(df['Quantity'].sum())
    total_revenue = round(float(df['Revenue'].sum()), 2)
    total_products = int(df['StockCode'].nunique())

    # Top best-selling products
    top_bestsellers = (
        df.groupby(['StockCode', 'Description'])
        .agg(TotalQuantity=('Quantity', 'sum'), TotalRevenue=('Revenue', 'sum'))
        .reset_index()
        .sort_values(by='TotalQuantity', ascending=False)
        .head(10)
        .to_dict(orient='records')
    )

    # Average daily demand
    df['Date'] = pd.to_datetime(df['Date'])
    daily_totals = df.groupby('Date')['Quantity'].sum()
    avg_daily_demand = round(float(daily_totals.mean()), 1)

    # Calculate count of top active modeled products
    catalog = list(forecaster.products_catalog.values())
    active_modeled_count = len(catalog)

    # Approximate high-risk simulation count across modeled catalog assuming typical 7-day inventory
    high_risk_simulated = 0
    for prod in catalog[:50]:
        avg_d = float(prod.get('AvgDailyDemand', 10))
        # Simulated scenario: 40 units in stock vs 7-day lead demand
        if avg_d * 7 >= 40:
            high_risk_simulated += 1

    _dashboard_cache = {
        "total_products": total_products,
        "active_modeled_products": active_modeled_count,
        "total_transactions": total_transactions,
        "total_sales_quantity": total_sales_quantity,
        "total_revenue": total_revenue,
        "avg_daily_retail_demand": avg_daily_demand,
        "simulated_high_risk_skus": high_risk_simulated,
        "date_range": {
            "start": str(df['Date'].min().date()),
            "end": str(df['Date'].max().date())
        },
        "best_selling_products": top_bestsellers
    }
    return _dashboard_cache


# ----------------- API ENDPOINTS -----------------

@app.get("/", summary="Root endpoint - Serves Web Application")
def root():
    """Serves the frontend dashboard index.html."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "status": "online",
        "message": "Retail Product Demand Estimation and Stockout Risk Analysis API is active.",
        "docs_url": "/docs"
    }

@app.get("/status", summary="Check Application and ML Model Status")
def get_status():
    """Returns runtime health and loaded model status."""
    return {
        "status": "healthy",
        "model_loaded": forecaster.model is not None,
        "model_type": str(type(forecaster.model).__name__),
        "products_catalog_size": len(forecaster.products_catalog),
        "historical_data_loaded": forecaster.featured_df is not None,
        "best_model_name": forecaster.metadata.get("best_model", "Gradient Boosting")
    }

@app.get("/products", summary="List Available Products")
def get_products(
    query: Optional[str] = Query(None, description="Search term for StockCode or Description"),
    limit: int = Query(150, ge=1, le=500, description="Max products to return")
):
    """Returns available products with catalog descriptions and historical metrics."""
    catalog = list(forecaster.products_catalog.values())
    if not catalog:
        # Fallback to metadata file if forecaster not yet loaded
        cat_path = os.path.join(MODELS_DIR, "products_catalog.json")
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                catalog = json.load(f)

    if query:
        q = query.strip().upper()
        catalog = [
            p for p in catalog
            if q in str(p.get('StockCode', '')).upper() or q in str(p.get('Description', '')).upper()
        ]

    return {
        "total_available": len(catalog),
        "returned": min(len(catalog), limit),
        "products": catalog[:limit]
    }

@app.get("/dashboard", summary="Dashboard KPI Metrics")
def get_dashboard():
    """Returns aggregate dashboard metrics, total volume, and best sellers."""
    return get_dashboard_summary()

@app.post("/predict", summary="Forecast Future Product Demand")
def predict_demand(req: PredictRequest):
    """
    Accepts product ID and forecast duration.
    Returns multi-step daily predictions, aggregate demand, and trend trajectory.
    """
    stock_code = req.product_id.strip().upper()
    try:
        result = forecaster.forecast(stock_code, forecast_days=req.forecast_days)
        return {
            "success": True,
            "data": result
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast calculation failed: {str(e)}"
        )

@app.post("/stockout-risk", summary="Compute Stockout Risk and Inventory Replenishment")
def calculate_stockout_risk(req: StockoutRequest):
    """
    Calculates stockout risk using the explicit rule:
    Risk = HIGH if forecast demand during lead time >= available inventory.
    """
    stock_code = req.product_id.strip().upper()
    try:
        safety_buf = req.safety_buffer_pct if req.safety_buffer_pct is not None else 0.20
        analysis = analyze_stockout_risk(
            stock_code=stock_code,
            current_inventory=req.available_inventory,
            supplier_lead_time=req.supplier_lead_time,
            safety_stock_threshold_pct=safety_buf
        )
        return {
            "success": True,
            "data": analysis
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stockout risk calculation failed: {str(e)}"
        )

@app.get("/model-comparison", summary="Machine Learning Model Evaluation Comparison")
def get_model_comparison():
    """Returns comparative metrics (MAE, MSE, RMSE, R²) for all regression models."""
    metrics_file = os.path.join(OUTPUTS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(MODELS_DIR, "model_metadata.json")

    if not os.path.exists(metrics_file):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model evaluation metrics file not found."
        )

    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    # Explanation of algorithm behavior
    explanation = {
        "Linear Regression": "Serves as a baseline model with L2 regularization. While interpretable and fast, it struggles to capture non-linear seasonality and complex lag interactions, achieving R² ≈ 0.28.",
        "Random Forest": "Non-linear bagging ensemble of decision trees. Captures non-linear demand thresholds and price elasticities well (R² ≈ 0.31), robust to variance but prone to tree ensemble memory overhead.",
        "Gradient Boosting": "Best-performing model overall (R² ≈ 0.32, lowest RMSE = 71.81). Sequentially corrects previous residuals, excelling at capturing rolling momentum and holiday surge dynamics.",
        "Tuned Gradient Boosting": "Fine-tuned via TimeSeriesSplit chronological cross-validation with optimized tree depth (4), learning rate (0.08), and sample splits to prevent overfitting."
    }

    metrics_data["model_explanations"] = explanation
    return metrics_data

@app.get("/sales-trends", summary="Historical Demand Trends for Charts")
def get_sales_trends():
    """Returns monthly and daily aggregated retail demand trajectories for frontend visualizers."""
    if not os.path.exists(CLEANED_DATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cleaned dataset not available."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    # Monthly trends
    monthly = (
        df.groupby('YearMonth')
        .agg(TotalQuantity=('Quantity', 'sum'), TotalRevenue=('Revenue', 'sum'))
        .reset_index()
        .sort_values('YearMonth')
    )

    # Day of week demand
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_demand = (
        df.groupby('DayOfWeek')['Quantity']
        .sum()
        .reindex(day_order)
        .fillna(0)
        .reset_index()
    )

    # Recent 60 days daily trajectory
    df['Date'] = pd.to_datetime(df['Date'])
    daily = (
        df.groupby('Date')['Quantity']
        .sum()
        .reset_index()
        .sort_values('Date')
        .tail(60)
    )

    return {
        "monthly_sales": {
            "labels": monthly['YearMonth'].tolist(),
            "quantity": monthly['TotalQuantity'].tolist(),
            "revenue": [round(float(r), 2) for r in monthly['TotalRevenue'].tolist()]
        },
        "day_of_week": {
            "labels": dow_demand['DayOfWeek'].tolist(),
            "quantity": dow_demand['Quantity'].tolist()
        },
        "recent_daily_trajectory": {
            "dates": [d.strftime("%Y-%m-%d") for d in daily['Date']],
            "quantity": daily['Quantity'].tolist()
        }
    }

@app.get("/product/{stock_code}/history", summary="Historical Demand Series for a SKU")
def get_product_history(stock_code: str):
    """Returns historical daily demand series for the specified product."""
    code = stock_code.strip().upper()
    info = forecaster.get_product_info(code)
    if not info:
        raise HTTPException(status_code=404, detail=f"Product {code} not found.")

    if forecaster.featured_df is None:
        raise HTTPException(status_code=503, detail="Historical dataset not loaded.")

    p_df = forecaster.featured_df[forecaster.featured_df['StockCode'] == code].sort_values('Date')
    return {
        "stock_code": code,
        "description": info.get('Description', ''),
        "total_demand": int(info.get('TotalDemand', 0)),
        "avg_daily_demand": float(info.get('AvgDailyDemand', 0)),
        "avg_price": float(info.get('AvgPrice', 0)),
        "history": [
            {
                "date": row['Date'].strftime("%Y-%m-%d"),
                "demand": float(row['DailyDemand']),
                "avg_price": float(row['AvgUnitPrice'])
            }
            for _, row in p_df.tail(60).iterrows()
        ]
    }
