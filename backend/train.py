"""
backend/train.py
Model training, evaluation, comparison, and hyperparameter tuning module.
Trains:
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
Evaluates on unseen future data using a strict chronological split.
Saves models and metrics for backend API usage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from backend.preprocessing import clean_retail_data, generate_daily_demand_features
from backend.temporal_validation import make_date_splits, get_date_split_ranges
from backend.temporal_encoder import ChronologicalTargetEncoder

NUMERICAL_FEATURES = [
    'day', 'month', 'year', 'dayofweek', 'is_weekend', 'weekofyear',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_trend', 'cv_7', 'AvgUnitPrice'
]
CATEGORICAL_FEATURES = ['StockCode']
FEATURE_COLUMNS = ['StockCode', 'Date'] + NUMERICAL_FEATURES
TARGET = 'DailyDemand'

def build_preprocessor():
    """Builds a scikit-learn column transformer for categorical and numerical features."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', ChronologicalTargetEncoder(date_col='Date', cat_col='StockCode', smooth=10.0, prior_mean=0.0), ['StockCode', 'Date'])
        ],
        remainder='drop'
    )

def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculates MAE, MSE, RMSE, and R2."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }

def train_and_evaluate_models(
    dataset_excel_or_csv: str,
    models_dir: str = "models",
    outputs_dir: str = "outputs",
    cleaned_csv_path: str = "dataset/cleaned_retail_data.csv"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "graphs"), exist_ok=True)

    # 1. Clean data or load existing cleaned CSV
    if os.path.exists(cleaned_csv_path):
        print(f"[TRAIN] Cleaned dataset found at {cleaned_csv_path}. Loading...")
        cleaned_df = pd.read_csv(cleaned_csv_path)
    else:
        cleaned_df = clean_retail_data(dataset_excel_or_csv, output_path=cleaned_csv_path)

    # 2. Generate daily demand time series and features
    featured_df, prod_meta = generate_daily_demand_features(cleaned_df)
    
    # Save product catalog metadata for fast retrieval by API
    products_catalog = prod_meta.to_dict(orient='records')
    with open(os.path.join(models_dir, "products_catalog.json"), "w") as f:
        json.dump(products_catalog, f, indent=2)

    # Save featured df for fast lookup in testing / inference
    featured_df.to_csv(os.path.join(models_dir, "featured_demand.csv.gz"), index=False, compression='gzip')

    # 3. Chronological Train-Test Split (strictly forward in time to prevent data leakage)
    featured_df['Date'] = pd.to_datetime(featured_df['Date'])
    split_date = pd.to_datetime("2011-11-01")
    
    train_mask = featured_df['Date'] < split_date
    test_mask = featured_df['Date'] >= split_date

    X_train = featured_df.loc[train_mask, FEATURE_COLUMNS]
    y_train = featured_df.loc[train_mask, TARGET]
    
    X_test = featured_df.loc[test_mask, FEATURE_COLUMNS]
    y_test = featured_df.loc[test_mask, TARGET]

    dates_test = featured_df.loc[test_mask, 'Date']
    stock_test = featured_df.loc[test_mask, 'StockCode']

    print(f"[TRAIN] Chronological split at {split_date.strftime('%Y-%m-%d')}:")
    print(f"  Training samples: {len(X_train)} (from {featured_df.loc[train_mask, 'Date'].min().date()} to {featured_df.loc[train_mask, 'Date'].max().date()})")
    print(f"  Testing samples:  {len(X_test)} (from {featured_df.loc[test_mask, 'Date'].min().date()} to {featured_df.loc[test_mask, 'Date'].max().date()})")

    # Define the 3 required regression models
    models = {
        "Linear Regression": Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=6,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                random_state=42
            ))
        ])
    }

    metrics_comparison = {}
    test_predictions = {}
    saved_model_paths = {}

    for name, pipeline in models.items():
        print(f"\n[TRAIN] Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Predict on future test set
        y_pred = pipeline.predict(X_test)
        # Ensure predicted demand cannot be negative physically
        y_pred = np.clip(y_pred, 0, None)
        
        metrics = evaluate_predictions(y_test, y_pred)
        metrics_comparison[name] = metrics
        test_predictions[name] = y_pred

        # Save individual model
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(models_dir, filename)
        joblib.dump(pipeline, path)
        saved_model_paths[name] = path
        print(f"[TRAIN] {name} Performance: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R2={metrics['R2']}")

    # 4. Phase 9: Hyperparameter Tuning on best performer using date-based chronological CV
    print("\n[TUNING] Performing Hyperparameter Tuning on Gradient Boosting...")
    train_dates_df = featured_df.loc[train_mask, ['Date']]
    cv_splits = make_date_splits(train_dates_df, n_splits=3, date_col='Date')
    param_dist = {
        'regressor__n_estimators': [100, 150, 200],
        'regressor__max_depth': [4, 5, 7],
        'regressor__learning_rate': [0.04, 0.08, 0.12],
        'regressor__min_samples_split': [4, 8, 12]
    }

    gb_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    search = RandomizedSearchCV(
        gb_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    print(f"[TUNING] Best Parameters: {best_params}")

    # Evaluate tuned model on future test set
    tuned_pred = np.clip(best_pipeline.predict(X_test), 0, None)
    tuned_metrics = evaluate_predictions(y_test, tuned_pred)
    print(f"[TUNING] Tuned Model Test Performance: MAE={tuned_metrics['MAE']}, RMSE={tuned_metrics['RMSE']}, R2={tuned_metrics['R2']}")

    # Save best model
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[TUNING] Best model saved to: {best_model_path}")

    # Record metrics
    metrics_comparison["Tuned Gradient Boosting (Best)"] = tuned_metrics
    test_predictions["Tuned Gradient Boosting"] = tuned_pred

    # Determine winning model by lowest RMSE and MAE
    best_model_name = min(
        metrics_comparison.keys(), 
        key=lambda k: metrics_comparison[k]['RMSE']
    )
    print(f"\n[EVALUATION] Winning Model: {best_model_name}")

    # 5. Generate Actual vs Predicted Plot
    print("[EVALUATION] Generating Actual vs Predicted Demand plot...")
    plt.figure(figsize=(13, 6))
    
    # Aggregate test predictions and actuals by date
    eval_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test.values,
        'Pred_GB': test_predictions["Tuned Gradient Boosting"],
        'Pred_RF': test_predictions["Random Forest"],
        'Pred_LR': test_predictions["Linear Regression"]
    }).groupby('Date').sum().reset_index()

    plt.plot(eval_df['Date'], eval_df['Actual'], label='Actual Daily Demand', color='black', linewidth=2.5)
    plt.plot(eval_df['Date'], eval_df['Pred_GB'], label='Tuned Gradient Boosting', color='#2ca02c', linestyle='--', linewidth=2)
    plt.plot(eval_df['Date'], eval_df['Pred_RF'], label='Random Forest', color='#1f77b4', linestyle=':', linewidth=2)
    plt.plot(eval_df['Date'], eval_df['Pred_LR'], label='Linear Regression', color='#d62728', linestyle='-.', linewidth=1.5)
    
    plt.title(f"Unseen Future Demand: Actual vs Predicted (Nov 1 - Dec 9, 2011)", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Test Horizon Date", fontsize=12)
    plt.ylabel("Total Aggregate Daily Quantity", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "graphs", "actual_vs_predicted.png"), dpi=200)
    plt.close()

    # Save comprehensive metrics json
    output_summary = {
        "split_date": str(split_date.date()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "best_model": best_model_name,
        "best_hyperparameters": {k.replace('regressor__', ''): v for k, v in best_params.items()},
        "metrics": metrics_comparison,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        }
    }

    with open(os.path.join(outputs_dir, "model_metrics.json"), "w") as f:
        json.dump(output_summary, f, indent=2)
    
    # Also save metadata inside models folder
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(output_summary, f, indent=2)

    print(f"[COMPLETE] Model training and evaluation successfully finished.")
    return output_summary

if __name__ == "__main__":
    train_and_evaluate_models(
        dataset_excel_or_csv="dataset/Online Retail.xlsx",
        models_dir="models",
        outputs_dir="outputs",
        cleaned_csv_path="dataset/cleaned_retail_data.csv"
    )
