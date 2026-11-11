"""
backend/train.py
Leakage-free model training, encoding comparison, chronological cross-validation,
validation-based model selection, rolling-origin recursive backtesting, and serialization.

Algorithms Retained:
  1. Ridge Regression (L2 Linear Baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor (+ Tuned Gradient Boosting via date-based CV)
"""

import os
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from backend.features import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET
from backend.preprocessing import clean_retail_data, generate_daily_demand_features
from backend.temporal_validation import make_date_splits, get_date_split_ranges
from backend.temporal_encoder import ChronologicalTargetEncoder
from backend.baselines import (
    compute_regression_metrics,
    evaluate_single_step_baselines,
)
from backend.backtesting import (
    run_rolling_origin_backtest,
    evaluate_recursive_at_origin,
)

RANDOM_SEED = 42
SPLIT_DATE_STR = "2011-11-01"


def build_preprocessor(encoding_mode: str = "chronological_target") -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer with strictly leakage-safe preprocessing.

    Supported encoding modes:
      - 'none': Uses numerical/historical features only (drops StockCode and Date).
      - 'onehot': Uses OneHotEncoder(handle_unknown='ignore') on StockCode.
      - 'chronological_target': Uses strictly out-of-past ChronologicalTargetEncoder on ['StockCode', 'Date'].
    """
    transformers = [('num', StandardScaler(), NUMERICAL_FEATURES)]

    if encoding_mode == "none":
        pass
    elif encoding_mode == "onehot":
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        )
    elif encoding_mode == "chronological_target":
        transformers.append(
            ('cat', ChronologicalTargetEncoder(date_col='Date', cat_col='StockCode', smooth=10.0, prior_mean=0.0), ['StockCode', 'Date'])
        )
    else:
        raise ValueError(f"Unsupported encoding_mode '{encoding_mode}'.")

    return ColumnTransformer(transformers=transformers, remainder='drop')


def evaluate_predictions(y_true, y_pred) -> dict:
    """Calculates MAE, MSE, RMSE, and R2."""
    return compute_regression_metrics(y_true, y_pred)


def _evaluate_pipeline_on_cv(pipeline_template, X_train, y_train, cv_splits) -> dict:
    """Evaluates a cloned pipeline across expanding-window chronological CV folds."""
    all_y_true = []
    all_y_pred = []
    fold_metrics = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
        pipe = clone(pipeline_template)
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = np.clip(pipe.predict(X_train.iloc[va_idx]), 0.0, None)
        y_va = y_train.iloc[va_idx].to_numpy(dtype=float)

        m = compute_regression_metrics(y_va, preds)
        m["fold"] = fold_idx + 1
        m["val_samples"] = int(len(va_idx))
        fold_metrics.append(m)

        all_y_true.append(y_va)
        all_y_pred.append(preds)

    pooled_true = np.concatenate(all_y_true)
    pooled_pred = np.concatenate(all_y_pred)
    overall = compute_regression_metrics(pooled_true, pooled_pred)
    overall["fold_metrics"] = fold_metrics
    return overall


def compare_identity_encodings(X_train, y_train, cv_splits) -> tuple:
    """
    Phase 4 requirement: Compares no product identity encoding, OneHotEncoder baseline,
    and ChronologicalTargetEncoder strictly on chronological validation folds.
    Returns (best_encoding_mode, comparison_results_dict).
    """
    print("\n[ENCODING COMPARISON] Evaluating categorical identity encoding strategies via date-based CV...")
    modes = ["none", "onehot", "chronological_target"]
    results = {}

    for mode in modes:
        ridge_pipe = Pipeline([
            ('preprocessor', build_preprocessor(encoding_mode=mode)),
            ('regressor', Ridge(alpha=1.0))
        ])
        gb_pipe = Pipeline([
            ('preprocessor', build_preprocessor(encoding_mode=mode)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                random_state=RANDOM_SEED
            ))
        ])
        ridge_cv = _evaluate_pipeline_on_cv(ridge_pipe, X_train, y_train, cv_splits)
        gb_cv = _evaluate_pipeline_on_cv(gb_pipe, X_train, y_train, cv_splits)
        results[mode] = {
            "Ridge Regression": {k: v for k, v in ridge_cv.items() if k != "fold_metrics"},
            "Gradient Boosting": {k: v for k, v in gb_cv.items() if k != "fold_metrics"},
        }
        print(
            f"  Mode='{mode:<20}' | Ridge CV RMSE={ridge_cv['RMSE']:<7} R2={ridge_cv['R2']:<6} | "
            f"GB CV RMSE={gb_cv['RMSE']:<7} R2={gb_cv['R2']:<6}"
        )

    # Select best encoding mode by lowest average CV RMSE across Ridge and Gradient Boosting (never peeking at test!)
    selected_mode = min(
        modes,
        key=lambda m: 0.5 * (results[m]["Ridge Regression"]["RMSE"] + results[m]["Gradient Boosting"]["RMSE"])
    )
    print(f"[ENCODING COMPARISON] Selected leakage-safe encoding strategy: '{selected_mode}'")
    return selected_mode, results


def _build_dynamic_explanations(
    metrics_comparison: dict,
    cv_metrics: dict,
    train_metrics: dict,
    best_model_name: str,
    best_params: dict,
    selected_encoding: str,
) -> dict:
    """Builds accurate, dynamically computed model explanations directly from measured metrics."""
    snaive_rmse = metrics_comparison["Seasonal Naive Baseline (t-7)"]["RMSE"]
    explanations = {
        "Naive Baseline (t-1)": (
            f"Predicts previous day's demand verbatim (Test MAE={metrics_comparison['Naive Baseline (t-1)']['MAE']}, "
            f"RMSE={metrics_comparison['Naive Baseline (t-1)']['RMSE']}, R2={metrics_comparison['Naive Baseline (t-1)']['R2']})."
        ),
        "Seasonal Naive Baseline (t-7)": (
            f"Predicts demand from the same weekday 7 days earlier (Test MAE={metrics_comparison['Seasonal Naive Baseline (t-7)']['MAE']}, "
            f"RMSE={snaive_rmse}, R2={metrics_comparison['Seasonal Naive Baseline (t-7)']['R2']})."
        ),
    }

    for model_key in ["Ridge Regression", "Random Forest", "Gradient Boosting", "Tuned Gradient Boosting"]:
        tm = metrics_comparison[model_key]
        cvm = cv_metrics[model_key]
        trm = train_metrics[model_key]
        vs_snaive = "outperforms" if tm["RMSE"] < snaive_rmse else "trails"
        winner_tag = " [SELECTED CHAMPION via Chronological CV]" if model_key == best_model_name else ""
        extra = f" Tuned params: {best_params}." if model_key == "Tuned Gradient Boosting" else ""
        explanations[model_key] = (
            f"{model_key}{winner_tag} using '{selected_encoding}' encoding. "
            f"CV RMSE={cvm['RMSE']} (Train RMSE={trm['RMSE']}, Test MAE={tm['MAE']}, RMSE={tm['RMSE']}, R2={tm['R2']}); "
            f"{vs_snaive} Seasonal Naive baseline RMSE ({snaive_rmse}).{extra}"
        )

    return explanations


def train_and_evaluate_models(
    dataset_excel_or_csv: str,
    models_dir: str = "models",
    outputs_dir: str = "outputs",
    cleaned_csv_path: str = "dataset/cleaned_retail_data.csv",
    force_reclean: bool = False,
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "graphs"), exist_ok=True)

    # 1. Load or clean data
    if not force_reclean and os.path.exists(cleaned_csv_path):
        print(f"[TRAIN] Cleaned dataset found at {cleaned_csv_path}. Loading...")
        cleaned_df = pd.read_csv(cleaned_csv_path)
    else:
        cleaned_df = clean_retail_data(dataset_excel_or_csv, output_path=cleaned_csv_path)

    # 2. Feature engineering strictly restricting product selection to < SPLIT_DATE_STR
    featured_df, prod_meta = generate_daily_demand_features(cleaned_df, cutoff_date=SPLIT_DATE_STR)

    products_catalog = prod_meta.to_dict(orient='records')
    with open(os.path.join(models_dir, "products_catalog.json"), "w") as f:
        json.dump(products_catalog, f, indent=2)

    featured_df.to_csv(os.path.join(models_dir, "featured_demand.csv.gz"), index=False, compression='gzip')

    # 3. Strict Chronological Train-Test Split
    featured_df['Date'] = pd.to_datetime(featured_df['Date'])
    split_date = pd.to_datetime(SPLIT_DATE_STR)

    train_mask = featured_df['Date'] < split_date
    test_mask = featured_df['Date'] >= split_date

    train_df = featured_df.loc[train_mask].copy().reset_index(drop=True)
    test_df = featured_df.loc[test_mask].copy().reset_index(drop=True)

    assert train_df['Date'].max() < test_df['Date'].min(), "Chronological train/test boundary violated!"

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    train_start = str(train_df['Date'].min().date())
    train_end = str(train_df['Date'].max().date())
    test_start = str(test_df['Date'].min().date())
    test_end = str(test_df['Date'].max().date())

    print(f"[TRAIN] Chronological split at {SPLIT_DATE_STR}:")
    print(f"  Training samples: {len(X_train)} ({train_start} to {train_end})")
    print(f"  Testing samples:  {len(X_test)} ({test_start} to {test_end})")

    # 4. Date-based expanding-window chronological CV (strictly inside training set)
    cv_splits = make_date_splits(train_df[['Date']], n_splits=3, date_col='Date')
    cv_fold_ranges = get_date_split_ranges(train_df[['Date']], n_splits=3, date_col='Date')

    # 5. Phase 4: Compare product identity encoding strategies using CV only
    selected_encoding, encoding_comparison = compare_identity_encodings(X_train, y_train, cv_splits)

    # 6. Define candidate ML pipelines
    candidate_templates = {
        "Ridge Regression": Pipeline([
            ('preprocessor', build_preprocessor(encoding_mode=selected_encoding)),
            ('regressor', Ridge(alpha=1.0))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', build_preprocessor(encoding_mode=selected_encoding)),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_split=6,
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', build_preprocessor(encoding_mode=selected_encoding)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                random_state=RANDOM_SEED
            ))
        ]),
    }

    # 7. Tune Gradient Boosting strictly via chronological validation folds
    print("\n[TUNING] Tuning Gradient Boosting via Date-Based Expanding-Window CV...")
    param_dist = {
        'regressor__n_estimators': [100, 150, 200],
        'regressor__max_depth': [3, 4, 5, 7],
        'regressor__learning_rate': [0.04, 0.06, 0.08, 0.12],
        'regressor__min_samples_split': [4, 8, 12],
        'regressor__subsample': [0.8, 0.85, 1.0],
    }

    gb_search_base = Pipeline([
        ('preprocessor', build_preprocessor(encoding_mode=selected_encoding)),
        ('regressor', GradientBoostingRegressor(random_state=RANDOM_SEED))
    ])

    search = RandomizedSearchCV(
        gb_search_base,
        param_distributions=param_dist,
        n_iter=8,
        cv=cv_splits,
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=False,  # Winner selection uses validation scores first; full refit done consistently below
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    clean_best_params = {k.replace('regressor__', ''): v for k, v in best_params.items()}
    print(f"[TUNING] Best Tuned GB Parameters (by CV RMSE): {clean_best_params}")

    candidate_templates["Tuned Gradient Boosting"] = Pipeline([
        ('preprocessor', build_preprocessor(encoding_mode=selected_encoding)),
        ('regressor', GradientBoostingRegressor(
            n_estimators=int(clean_best_params['n_estimators']),
            max_depth=int(clean_best_params['max_depth']),
            learning_rate=float(clean_best_params['learning_rate']),
            min_samples_split=int(clean_best_params['min_samples_split']),
            subsample=float(clean_best_params['subsample']),
            random_state=RANDOM_SEED,
        ))
    ])

    # 8. Evaluate ALL 4 models strictly on chronological validation folds (CV) to pick champion
    print("\n[VALIDATION] Evaluating all candidate models across chronological expanding CV folds...")
    cv_metrics = {}
    for name, template in candidate_templates.items():
        cvm = _evaluate_pipeline_on_cv(template, X_train, y_train, cv_splits)
        cv_metrics[name] = cvm
        print(f"  [CV] {name:<25} | MAE={cvm['MAE']:<7} RMSE={cvm['RMSE']:<7} R2={cvm['R2']:<7}")

    # Pick winning model strictly based on validation performance (lowest pooled chronological CV RMSE)
    best_model_name = min(candidate_templates.keys(), key=lambda k: cv_metrics[k]['RMSE'])
    print(f"\n[SELECTION] Selected Winning Model (by Chronological CV RMSE): {best_model_name}")

    # 9. Refit ALL models on the complete training period and evaluate once on untouched test period
    artifact_filenames = {
        "Ridge Regression": "ridge_model.pkl",
        "Random Forest": "random_forest.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "Tuned Gradient Boosting": "tuned_gradient_boosting.pkl",
    }

    metrics_comparison = evaluate_single_step_baselines(test_df, target_col=TARGET)
    print(
        f"\n[BASELINES] Naive (t-1): MAE={metrics_comparison['Naive Baseline (t-1)']['MAE']}, "
        f"RMSE={metrics_comparison['Naive Baseline (t-1)']['RMSE']}, R2={metrics_comparison['Naive Baseline (t-1)']['R2']}"
    )
    print(
        f"[BASELINES] Seasonal Naive (t-7): MAE={metrics_comparison['Seasonal Naive Baseline (t-7)']['MAE']}, "
        f"RMSE={metrics_comparison['Seasonal Naive Baseline (t-7)']['RMSE']}, R2={metrics_comparison['Seasonal Naive Baseline (t-7)']['R2']}"
    )

    train_metrics = {}
    fitted_models = {}
    test_predictions = {}

    for name, template in candidate_templates.items():
        print(f"\n[TRAIN] Fitting {name} on full training period ({len(X_train)} rows)...")
        fitted_pipe = clone(template)
        fitted_pipe.fit(X_train, y_train)
        fitted_models[name] = fitted_pipe

        train_pred = np.clip(fitted_pipe.predict(X_train), 0.0, None)
        trm = evaluate_predictions(y_train, train_pred)
        train_metrics[name] = trm

        test_pred = np.clip(fitted_pipe.predict(X_test), 0.0, None)
        tm = evaluate_predictions(y_test, test_pred)
        metrics_comparison[name] = tm
        test_predictions[name] = test_pred

        model_path = os.path.join(models_dir, artifact_filenames[name])
        joblib.dump(fitted_pipe, model_path)
        print(
            f"  Train RMSE={trm['RMSE']} | Test MAE={tm['MAE']}, RMSE={tm['RMSE']}, R2={tm['R2']} -> {model_path}"
        )

    # Save the EXACT winning model pipeline as best_model.pkl
    best_pipeline = fitted_models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_pipeline, best_model_path)
    print(f"\n[SERIALIZATION] Saved exact selected winning model '{best_model_name}' to {best_model_path}")

    # 10. Phase 8: Rolling-Origin Recursive Backtesting (strictly on pre-test historical origins)
    print("\n[BACKTESTING] Executing multi-origin recursive backtesting for 7d, 14d, 30d horizons...")
    rolling_bt = run_rolling_origin_backtest(
        pipeline_template=clone(candidate_templates[best_model_name]),
        featured_df=train_df,
        cutoff_date=SPLIT_DATE_STR,
        n_origins=4,
        horizons=(7, 14, 30),
    )

    # Also evaluate recursive multi-step rollouts across the untouched test horizon (origin = 2011-10-31)
    test_origin = train_df['Date'].max()
    test_recursive_df = evaluate_recursive_at_origin(
        fitted_pipeline=best_pipeline,
        featured_df=featured_df,
        origin_date=test_origin,
        max_horizon=30,
    )
    test_horizon_metrics = {}
    if not test_recursive_df.empty:
        for h in (7, 14, 30):
            sub = test_recursive_df[test_recursive_df['horizon_step'] <= h]
            if not sub.empty:
                ml_hm = compute_regression_metrics(sub['y_true'], sub['y_pred_ml'])
                sn_hm = compute_regression_metrics(sub['y_true'], sub['y_pred_snaive'])
                test_horizon_metrics[f"{h}d"] = {
                    "horizon_days": int(h),
                    "samples": int(len(sub)),
                    "ml": ml_hm,
                    "seasonal_naive": sn_hm,
                    "beats_seasonal_naive_rmse": bool(ml_hm["RMSE"] < sn_hm["RMSE"]),
                }

    rolling_bt["test_origin_recursive"] = {
        "origin_date": str(test_origin.date()),
        "horizon_metrics": test_horizon_metrics,
    }

    # 11. Plot Actual vs Predicted aggregate demand on test period
    print("[EVALUATION] Generating Actual vs Predicted Demand plot...")
    plt.figure(figsize=(13, 6))
    eval_df = pd.DataFrame({
        'Date': test_df['Date'],
        'Actual': y_test.to_numpy(dtype=float),
        'Pred_Best': test_predictions[best_model_name],
        'Pred_RF': test_predictions["Random Forest"],
        'Pred_Ridge': test_predictions["Ridge Regression"],
        'Pred_SNaive': np.clip(test_df['lag_7'].to_numpy(dtype=float), 0.0, None),
    }).groupby('Date').sum().reset_index()

    plt.plot(eval_df['Date'], eval_df['Actual'], label='Actual Daily Demand', color='black', linewidth=2.5)
    plt.plot(eval_df['Date'], eval_df['Pred_Best'], label=f'Champion: {best_model_name}', color='#2ca02c', linestyle='--', linewidth=2.2)
    if best_model_name != "Random Forest":
        plt.plot(eval_df['Date'], eval_df['Pred_RF'], label='Random Forest', color='#1f77b4', linestyle=':', linewidth=1.8)
    plt.plot(eval_df['Date'], eval_df['Pred_Ridge'], label='Ridge Regression', color='#d62728', linestyle='-.', linewidth=1.5)
    plt.plot(eval_df['Date'], eval_df['Pred_SNaive'], label='Seasonal Naive (t-7)', color='#9467bd', linestyle=':', linewidth=1.5, alpha=0.8)

    plt.title(f"Unseen Future Demand: Actual vs Predicted ({test_start} to {test_end})", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Test Horizon Date", fontsize=12)
    plt.ylabel("Total Aggregate Daily Quantity", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "graphs", "actual_vs_predicted.png"), dpi=200)
    plt.close()

    # 12. Save complete metadata & metrics
    explanations = _build_dynamic_explanations(
        metrics_comparison=metrics_comparison,
        cv_metrics=cv_metrics,
        train_metrics=train_metrics,
        best_model_name=best_model_name,
        best_params=clean_best_params,
        selected_encoding=selected_encoding,
    )

    winning_regressor = best_pipeline.named_steps['regressor']
    output_summary = {
        "split_date": SPLIT_DATE_STR,
        "train_period": {"start": train_start, "end": train_end},
        "validation_period": {
            "start": cv_fold_ranges[0]["val_start"],
            "end": cv_fold_ranges[-1]["val_end"],
            "folds": cv_fold_ranges,
        },
        "test_period": {"start": test_start, "end": test_end},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "best_model": best_model_name,
        "model_type": type(winning_regressor).__name__,
        "selection_criterion": "Lowest pooled 3-fold chronological CV RMSE",
        "validation_methodology": "Date-grouped expanding-window chronological CV (3 folds) + 4-origin recursive rolling backtest",
        "encoding_strategy": selected_encoding,
        "encoding_comparison": encoding_comparison,
        "best_hyperparameters": clean_best_params,
        "winning_model_params": winning_regressor.get_params(),
        "random_seed": RANDOM_SEED,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "all_input_columns": FEATURE_COLUMNS,
        },
        "metrics": metrics_comparison,
        "train_metrics": train_metrics,
        "cv_metrics": cv_metrics,
        "backtesting": rolling_bt,
        "model_explanations": explanations,
    }

    with open(os.path.join(outputs_dir, "model_metrics.json"), "w") as f:
        json.dump(output_summary, f, indent=2, default=str)

    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(output_summary, f, indent=2, default=str)

    print("[COMPLETE] Model training, validation, backtesting, and serialization finished.")
    return output_summary


if __name__ == "__main__":
    train_and_evaluate_models(
        dataset_excel_or_csv="dataset/Online Retail.xlsx",
        models_dir="models",
        outputs_dir="outputs",
        cleaned_csv_path="dataset/cleaned_retail_data.csv"
    )
