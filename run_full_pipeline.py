"""
run_full_pipeline.py
End-to-End Orchestrator:
Executes Data Cleaning -> EDA Charts -> Feature Engineering -> Model Training -> Hyperparameter Tuning -> Evaluation.
"""

import os
import sys
import time

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.preprocessing import clean_retail_data
from backend.eda import generate_eda_visualizations
from backend.train import train_and_evaluate_models

def main():
    start_time = time.time()
    print("=" * 70)
    print("STARTING COMPLETE RETAIL DEMAND FORECASTING PIPELINE")
    print("=" * 70)

    raw_excel = os.path.join(BASE_DIR, "dataset", "Online Retail.xlsx")
    cleaned_csv = os.path.join(BASE_DIR, "dataset", "cleaned_retail_data.csv")
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    graphs_dir = os.path.join(outputs_dir, "graphs")
    models_dir = os.path.join(BASE_DIR, "models")

    # Step 1: Preprocessing & Data Cleaning
    print("\n>>> STEP 1: PREPROCESSING & DATA CLEANING")
    if not os.path.exists(cleaned_csv):
        clean_retail_data(raw_excel, output_path=cleaned_csv)
    else:
        print(f"Cleaned data already exists at: {cleaned_csv}")

    # Step 2: Exploratory Data Analysis & Visualizations
    print("\n>>> STEP 2: GENERATING EDA VISUALIZATIONS")
    generate_eda_visualizations(cleaned_csv, graphs_dir)

    # Step 3: Feature Engineering, Model Training, Tuning & Evaluation
    print("\n>>> STEP 3: FEATURE ENGINEERING, MODEL TRAINING & HYPERPARAMETER TUNING")
    summary = train_and_evaluate_models(
        dataset_excel_or_csv=cleaned_csv,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        cleaned_csv_path=cleaned_csv
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"PIPELINE SUCCESSFULLY COMPLETED IN {elapsed}s")
    print("Best Model:", summary['best_model'])
    print("Metrics Table:")
    for model_name, m in summary['metrics'].items():
        print(f"  {model_name:<30} | MAE: {m['MAE']:<7} | RMSE: {m['RMSE']:<7} | R2: {m['R2']:<7}")
    print("=" * 70)

if __name__ == "__main__":
    main()
