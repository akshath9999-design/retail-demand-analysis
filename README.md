# Retail Product Demand Estimation and Stockout Risk Analysis

An end-to-end Machine Learning web application designed to forecast daily retail SKU demand, identify surging products, compare multiple regression algorithms on future unseen data, and calculate warehouse stockout risk through explicit inventory replenishment rules.

---

## 📌 Executive Summary

Retail inventory management requires balancing customer service levels against holding costs. Stockouts lead to lost revenue and customer churn, while overstocking ties up working capital. This application solves that challenge by:
1. **Aggregating and cleaning** 540,000+ transaction records from the **UCI Online Retail dataset**.
2. **Engineering strictly past-looking time-series features** (lag demand, rolling moving averages, volatility, calendar signals) with **zero data leakage**.
3. **Evaluating three regression algorithms** on a chronological future test horizon:
   - Linear Regression (Ridge L2 Baseline)
   - Random Forest Regressor
   - Gradient Boosting Regressor (Champion)
4. **Optimizing hyperparameters** using `TimeSeriesSplit` cross-validation.
5. **Simulating stockout risk** using an explicit inventory replenishment formula that accounts for supplier lead time and customizable safety buffers.
6. **Delivering an interactive web dashboard** built with HTML5, CSS3, JavaScript, and Chart.js, served by a production-ready **FastAPI** backend.

---

## 🛠 Technology Stack

- **Frontend**: Responsive Single Page Application (HTML5, CSS3, ES6+ JavaScript, Chart.js 4.4)
- **Backend**: Python 3.14, FastAPI, Uvicorn, Pydantic v2
- **Machine Learning**: Scikit-learn, Pandas, NumPy, Scipy, Joblib
- **Exploratory Data Analysis**: Matplotlib, Seaborn
- **Testing**: Pytest, FastAPI TestClient

---

## 📂 Project Directory Structure

```text
retail-demand-analysis/
│
├── dataset/
│   ├── Online Retail.xlsx               # Original raw UCI Online Retail dataset
│   └── cleaned_retail_data.csv          # Cleaned dataset (cancellations/negatives removed)
│
├── models/
│   ├── linear_regression.pkl            # Trained Ridge Linear Regression pipeline
│   ├── random_forest.pkl                # Trained Random Forest ensemble pipeline
│   ├── gradient_boosting.pkl            # Trained Gradient Boosting pipeline
│   ├── best_model.pkl                   # Optimized champion model
│   ├── featured_demand.csv.gz           # Daily demand time-series feature matrix
│   ├── products_catalog.json            # Active product catalog metadata
│   └── model_metadata.json              # Full training metrics and hyperparameters
│
├── notebooks/
│   └── eda_and_model_analysis.ipynb     # Interactive Jupyter notebook walkthrough
│
├── outputs/
│   ├── graphs/
│   │   ├── top10_bestsellers.png        # Top best sellers by quantity
│   │   ├── monthly_sales_trend.png      # Monthly volume & revenue trajectory
│   │   ├── daily_demand_trend.png       # Daily demand & 7-day moving average
│   │   ├── top_revenue_products.png     # Highest revenue-generating SKUs
│   │   ├── quantity_distribution.png    # Order batch size distribution
│   │   ├── demand_patterns_over_time.png# Demand distribution by day of week
│   │   ├── high_vs_low_demand.png       # Product quartile categorization
│   │   ├── product_wise_demand.png      # 7-day moving average trajectories for top SKUs
│   │   └── actual_vs_predicted.png      # Unseen future chronological validation plot
│   └── model_metrics.json               # Quantitative comparison metrics (MAE, RMSE, R²)
│
├── backend/
│   ├── __init__.py
│   ├── main.py                          # FastAPI REST application & routing
│   ├── preprocessing.py                 # Data cleaning and feature engineering
│   ├── train.py                         # Model training, tuning, and evaluation
│   ├── prediction.py                    # Multi-step autoregressive demand forecaster
│   ├── stockout.py                      # Explicit inventory risk calculation engine
│   └── eda.py                           # Matplotlib & Seaborn visualization generator
│
├── frontend/
│   ├── index.html                       # Analytics dashboard layout
│   ├── style.css                        # Modern CSS styling (slate/navy theme)
│   └── script.js                        # Client API controller & Chart.js renderer
│
├── tests/
│   ├── __init__.py
│   └── test_api.py                      # Pytest suite (18 automated tests)
│
├── run_full_pipeline.py                 # Master orchestrator script
├── main.py                              # Root execution entrypoint
├── requirements.txt                     # Project dependencies
└── README.md                            # Comprehensive documentation
```

---

## 🔬 Machine Learning Pipeline & Methodology

### 1. Data Cleaning (Phase 3)
- **Initial records**: 541,909 transactions across 4,070 products.
- **Deduplication**: Removed 5,268 duplicate transaction rows.
- **Cancellations**: Filtered out 9,288 invoice records beginning with `'C'`.
- **Negative & Zero Filters**: Excluded 10,624 non-positive quantities and 2,517 non-positive unit prices.
- **Service Codes**: Filtered non-merchandise bookkeeping items (`POST`, `D`, `M`, `BANK CHARGES`, `CRUK`, `DOT`, `AMAZONFEE`).
- **Missing Descriptions**: Imputed from known product catalog descriptions using the same `StockCode`.
- **Cleaned records**: **522,311** valid retail transactions saved to `dataset/cleaned_retail_data.csv`.

### 2. Feature Engineering (Phase 5)
Transactions were aggregated to product-level daily demand ($Quantity$). To prevent **data leakage**, complete daily calendar grids were generated and time-lagged variables were calculated strictly using historical observations ($t-k$):
- **Calendar Signals**: `day`, `month`, `year`, `dayofweek`, `is_weekend`, `weekofyear`.
- **Lag Features**: Demand 1 day ago (`lag_1`), 7 days ago (`lag_7`), 14 days ago (`lag_14`), and 30 days ago (`lag_30`).
- **Rolling Averages**: 7-day, 14-day, and 30-day moving averages (shifted by 1 day).
- **Demand Volatility**: 7-day rolling standard deviation (`rolling_std_7`) and coefficient of variation (`cv_7`).
- **Demand Momentum Ratio**: $\frac{\text{rolling\_mean\_7} + 0.1}{\text{rolling\_mean\_30} + 0.1}$.
- **Price Features**: Historical average unit price (`AvgUnitPrice`).

### 3. Chronological Train-Test Split (Phase 6)
To simulate actual deployment conditions, the data was split **chronologically** rather than randomly:
- **Training Horizon**: December 31, 2010 to October 31, 2011 (45,750 samples).
- **Unseen Future Testing Horizon**: November 1, 2011 to December 9, 2011 (5,850 samples during peak holiday volume).

### 4. Model Evaluation & Comparison (Phases 7 & 8)

All models were evaluated on the future unseen testing dataset:

| Model Algorithm | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Score | Performance Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Ridge)** | 41.2007 | 5,526.23 | 74.3386 | 0.2762 | Fast, interpretable linear baseline. Struggles with non-linear holiday surges. |
| **Random Forest Regressor** | 37.1607 | 5,282.11 | 72.6781 | 0.3082 | Captures non-linear thresholds and price interactions effectively. |
| **Gradient Boosting (Champion)** | **36.6066** | **5,156.88** | **71.8114** | **0.3246** | **Best overall model**. Lowest prediction error and highest explained variance. |
| **Tuned Gradient Boosting** | 37.5562 | 5,413.87 | 73.5790 | 0.2909 | Fine-tuned with `TimeSeriesSplit` cross-validation (Max Depth: 4, LR: 0.08). |

---

## 📦 Stockout Risk Analysis Formulation (Phase 10)

### Dataset Limitation Notice
The UCI Online Retail dataset records transactional sales and does not contain warehouse stock records. To provide actionable decision support without fabricating historical stock, the application uses an **explicit forecast-driven inventory simulation**.

### Mathematical Formulation
Given:
- $I_{\text{avail}}$: Current on-hand available inventory (units).
- $L$: Supplier lead time (days).
- $B$: Configurable safety stock buffer fraction (default 20%).

1. **Forecasted Lead-Time Demand**:
   $$D_{\text{lead}} = \sum_{t=1}^{L} \hat{y}_t$$
   *(where $\hat{y}_t$ is the autoregressive daily demand predicted by the champion Gradient Boosting model)*

2. **Allocated Safety Stock Buffer**:
   $$SS = \max\left(2, \text{round}(D_{\text{lead}} \times B)\right)$$

3. **Projected Remaining Inventory**:
   $$I_{\text{rem}} = I_{\text{avail}} - D_{\text{lead}}$$

4. **Explicit Decision Rule**:
   $$\text{Stockout Risk} = \begin{cases} \mathbf{HIGH}, & \text{if } D_{\text{lead}} \ge I_{\text{avail}} \;\text{ OR }\; I_{\text{rem}} < SS \\ \mathbf{LOW}, & \text{otherwise} \end{cases}$$

5. **Suggested Replenishment Quantity**:
   $$\text{Replenish} = \begin{cases} \max(0, \text{round}(D_{\text{lead}} + SS - I_{\text{avail}})), & \text{if Risk is HIGH} \\ 0, & \text{if Risk is LOW} \end{cases}$$

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Internet connection (for CDN assets like Chart.js)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run End-to-End ML Pipeline (Optional if models already saved)
```bash
python run_full_pipeline.py
```

### Step 3: Launch the Application
```bash
python main.py
```
*Alternatively, start directly via Uvicorn:*
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Step 4: Access the Application
- **Interactive Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Automated Testing

Execute the test suite verifying all 18 unit and integration checks:
```bash
python -m pytest tests/test_api.py -v
```

Test coverage includes:
- Dataset existence and schema integrity.
- Cancellation and negative quantity filtering.
- Feature engineering lag sanity checks (no future leakage).
- Serialization of all 3 regression models and champion model.
- Autoregressive multi-step demand forecaster execution.
- Stockout risk logic under deficit and surplus scenarios.
- All FastAPI endpoints (`/status`, `/dashboard`, `/products`, `/predict`, `/stockout-risk`, `/model-comparison`, `/sales-trends`).
- Input validation and HTTP 404/422 error handlers.

---

## 🌐 API Reference

### 1. `GET /status`
Returns application health, loaded ML model status, and catalog size.

### 2. `GET /dashboard`
Returns high-level retail metrics: total products, total transactions, total units sold, gross revenue, and top 10 best-selling SKUs.

### 3. `GET /products?query={text}&limit={n}`
Returns active product catalog with descriptions, historical demand, and average prices.

### 4. `POST /predict`
Generates a multi-step daily forecast for a given product and prediction window.
```json
// Request
{
  "product_id": "85123A",
  "forecast_days": 14
}

// Response
{
  "success": true,
  "data": {
    "stock_code": "85123A",
    "description": "WHITE HANGING HEART T-LIGHT HOLDER",
    "forecast_days": 14,
    "total_forecast_demand": 1465.6,
    "avg_daily_demand": 104.7,
    "peak_daily_demand": 115.4,
    "trend_direction": "DECREASING",
    "forecast_schedule": [
      { "date": "2011-12-10", "day_name": "Sat", "predicted_demand": 41.4 },
      { "date": "2011-12-11", "day_name": "Sun", "predicted_demand": 76.9 }
    ]
  }
}
```

### 5. `POST /stockout-risk`
Calculates stockout probability and replenishment orders.
```json
// Request
{
  "product_id": "85123A",
  "available_inventory": 50,
  "supplier_lead_time": 10,
  "safety_buffer_pct": 0.20
}

// Response
{
  "success": true,
  "data": {
    "stock_code": "85123A",
    "available_inventory": 50,
    "supplier_lead_time_days": 10,
    "expected_lead_time_demand": 995.7,
    "inventory_remaining_after_lead_time": -945.7,
    "stockout_status": "HIGH",
    "risk_color": "#dc3545",
    "suggested_replenishment": 1145,
    "estimated_runway_days": 0.5,
    "safety_stock_buffer": 199
  }
}
```

### 6. `GET /model-comparison`
Returns evaluation metrics (MAE, MSE, RMSE, R²) for all tested regression algorithms.

### 7. `GET /sales-trends`
Returns monthly, daily, and day-of-week aggregations for Chart.js telemetry.

---

## 👥 Authors & License

Developed as an end-to-end Machine Learning and Inventory Intelligence Web Application using the UCI Online Retail Dataset. Released under the MIT License.
