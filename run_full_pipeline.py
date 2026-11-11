"""
run_full_pipeline.py
End-to-End Orchestrator:
Executes Data Cleaning -> EDA Charts -> Feature Engineering -> Model Training -> Hyperparameter Tuning -> Evaluation.
"""

import os
import sys
import time
import argparse

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.preprocessing import clean_retail_data
from backend.eda import generate_eda_visualizations
from backend.train import train_and_evaluate_models


def main():
    parser = argparse.ArgumentParser(description="Run full Retail Demand Forecasting pipeline.")
    parser.add_argument(
        "--reclean",
        action="store_true",
        help="Force regeneration of cleaned_retail_data.csv from raw Excel source.",
    )
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("STARTING COMPLETE RETAIL DEMAND FORECASTING PIPELINE")
    print("=" * 70)

    raw_excel = os.path.join(BASE_DIR, "dataset", "Online Retail.xlsx")
    cleaned_csv = os.path.join(BASE_DIR, "dataset", "cleaned_retail_data.csv")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    graphs_dir = os.path.join(outputs_dir, "graphs")
    models_dir = os.path.join(BASE_DIR, "models")

    # Remove deprecated artifact name if left over from an earlier run
    legacy_lr_pkl = os.path.join(models_dir, "linear_regression.pkl")
    if os.path.exists(legacy_lr_pkl):
        os.remove(legacy_lr_pkl)

    # Step 1: Preprocessing & Data Cleaning
    print("\n>>> STEP 1: PREPROCESSING & DATA CLEANING")
    if args.reclean or not os.path.exists(cleaned_csv):
        clean_retail_data(raw_excel, output_path=cleaned_csv)
    else:
        print(f"Cleaned data already exists at: {cleaned_csv}")

    # Step 2: Exploratory Data Analysis & Visualizations
    print("\n>>> STEP 2: GENERATING EDA VISUALIZATIONS")
    generate_eda_visualizations(cleaned_csv, graphs_dir)

    # Step 3: Feature Engineering, Encoding Comparison, Model Training, CV Tuning, Backtesting & Evaluation
    print("\n>>> STEP 3: FEATURE ENGINEERING, CHRONOLOGICAL CV, BACKTESTING & TRAINING")
    summary = train_and_evaluate_models(
        dataset_excel_or_csv=cleaned_csv,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        cleaned_csv_path=cleaned_csv,
        force_reclean=False,
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"PIPELINE SUCCESSFULLY COMPLETED IN {elapsed}s")
    print(f"Selected Champion ({summary['selection_criterion']}): {summary['best_model']} [{summary['model_type']}]")
    print(f"Selected SKU Encoding: {summary['encoding_strategy']}")
    print("\nTest Holdout Metrics Table:")
    for model_name, m in summary['metrics'].items():
        print(f"  {model_name:<30} | MAE: {m['MAE']:<7} | RMSE: {m['RMSE']:<7} | R2: {m['R2']:<7}")
    print("\nChronological 3-Fold CV Metrics:")
    for model_name, m in summary['cv_metrics'].items():
        print(f"  {model_name:<30} | CV MAE: {m['MAE']:<7} | CV RMSE: {m['RMSE']:<7} | CV R2: {m['R2']:<7}")
    bt = summary.get('backtesting', {})
    if bt.get('horizon_metrics'):
        print("\nRolling-Origin Recursive Backtest Horizons:")
        for h_key, h_vals in bt['horizon_metrics'].items():
            ml_h = h_vals['ml']
            sn_h = h_vals['seasonal_naive']
            print(f"  {h_key:<4} | ML RMSE: {ml_h['RMSE']:<7} MAE: {ml_h['MAE']:<7} vs S.Naive RMSE: {sn_h['RMSE']:<7} MAE: {sn_h['MAE']:<7} | Win={h_vals['beats_seasonal_naive_rmse']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
