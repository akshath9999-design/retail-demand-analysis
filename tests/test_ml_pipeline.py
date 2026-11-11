"""
tests/test_ml_pipeline.py
Comprehensive ML engineering verification suite covering Phases 2-12:
- Paired cancellation reversal removal & preservation of legitimate bulk orders
- Cutoff-strict product selection & zero test-period leakage
- Strict causality of lag, rolling (ddof=1), variability, sparsity & historical price features
- Exact training-inference feature parity (vectorized vs step-by-step continuation)
- Chronological CV & leakage-free category encoding (OneHotEncoder & ChronologicalTargetEncoder)
- Naive / Seasonal Naive baseline calculations & multi-origin recursive backtesting
- Autoregressive multi-horizon forecasting (1, 7, 14, 30 days) & future price overrides
- Stockout risk classification boundary conditions & input validation
"""

from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.preprocessing import clean_retail_data, generate_daily_demand_features
from backend.features import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    compute_product_features_vectorized,
    extract_product_forecast_state,
    build_feature_row_from_state,
    advance_product_forecast_state,
)
from backend.temporal_validation import make_date_splits
from backend.temporal_encoder import ChronologicalTargetEncoder
from backend.baselines import (
    recursive_naive_forecast,
    recursive_seasonal_naive_forecast,
    evaluate_single_step_baselines,
)
from backend.backtesting import run_rolling_origin_backtest
from backend.prediction import DemandForecaster
from backend.stockout import (
    compute_safety_stock_buffer,
    classify_stockout_risk,
    compute_replenishment_quantity,
    analyze_stockout_risk,
)


def _make_synthetic_transactions():
    """Creates synthetic retail transactions testing cancellations, voids, wholesale, and future SKUs."""
    rows = []
    # SKU_A: 45 pre-cutoff active days, steady demand ~10/day + legitimate wholesale order of 1200 units
    start = pd.Timestamp("2011-08-01 10:00:00")
    for d in range(60):
        ts = start + pd.Timedelta(days=d)
        rows.append({
            "InvoiceNo": f"500{d:03d}",
            "StockCode": "SKU_A",
            "Description": "TEST WIDGET A",
            "Quantity": 1200 if d == 15 else 10 + (d % 4),
            "InvoiceDate": ts,
            "UnitPrice": 2.5 + 0.01 * d,
            "CustomerID": 10001.0,
        })

    # Clerical spike (+50000) immediately voided (-50000) 20 minutes later -> BOTH must be removed
    rows.append({
        "InvoiceNo": "581483",
        "StockCode": "SKU_A",
        "Description": "TEST WIDGET A",
        "Quantity": 50000,
        "InvoiceDate": pd.Timestamp("2011-08-10 09:00:00"),
        "UnitPrice": 2.5,
        "CustomerID": 10099.0,
    })
    rows.append({
        "InvoiceNo": "C581484",
        "StockCode": "SKU_A",
        "Description": "TEST WIDGET A",
        "Quantity": -50000,
        "InvoiceDate": pd.Timestamp("2011-08-10 09:20:00"),
        "UnitPrice": 2.5,
        "CustomerID": 10099.0,
    })

    # Non-product service / gift voucher rows -> must be stripped
    rows.append({
        "InvoiceNo": "599901",
        "StockCode": "GIFT_0001_10",
        "Description": "GIFT VOUCHER",
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2011-08-12 11:00:00"),
        "UnitPrice": 10.0,
        "CustomerID": 10002.0,
    })
    rows.append({
        "InvoiceNo": "599902",
        "StockCode": "POST",
        "Description": "POSTAGE",
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2011-08-12 11:05:00"),
        "UnitPrice": 15.0,
        "CustomerID": 10002.0,
    })

    # SKU_FUTURE: low pre-cutoff volume (only 5 active days), huge post-cutoff volume -> must NOT beat cutoff selection
    for d in range(5):
        rows.append({
            "InvoiceNo": f"600{d:03d}",
            "StockCode": "SKU_FUTURE",
            "Description": "POST CUTOFF SURGE ITEM",
            "Quantity": 2,
            "InvoiceDate": pd.Timestamp("2011-10-01") + pd.Timedelta(days=d),
            "UnitPrice": 5.0,
            "CustomerID": 10003.0,
        })
    for d in range(30):
        rows.append({
            "InvoiceNo": f"610{d:03d}",
            "StockCode": "SKU_FUTURE",
            "Description": "POST CUTOFF SURGE ITEM",
            "Quantity": 5000,
            "InvoiceDate": pd.Timestamp("2011-11-05") + pd.Timedelta(days=d),
            "UnitPrice": 5.0,
            "CustomerID": 10003.0,
        })

    return pd.DataFrame(rows)


def test_clean_retail_data_paired_reversals_and_wholesale_preservation(tmp_path):
    raw_df = _make_synthetic_transactions()
    in_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "cleaned.csv"
    raw_df.to_csv(in_csv, index=False)

    cleaned = clean_retail_data(str(in_csv), output_path=str(out_csv))

    assert "581483" not in cleaned["InvoiceNo"].values, "Immediately voided invoice 581483 must be removed."
    assert "C581484" not in cleaned["InvoiceNo"].values, "Cancellation C581484 must be removed."
    assert "GIFT_0001_10" not in cleaned["StockCode"].values, "GIFT_* voucher must be stripped."
    assert "POST" not in cleaned["StockCode"].values, "Service code POST must be stripped."
    assert 1200 in cleaned["Quantity"].values, "Legitimate uncancelled wholesale order (1200) must be preserved."


def test_cutoff_product_selection_no_future_leakage(tmp_path):
    raw_df = _make_synthetic_transactions()
    in_csv = tmp_path / "raw.csv"
    raw_df.to_csv(in_csv, index=False)
    cleaned = clean_retail_data(str(in_csv))

    featured_df, meta = generate_daily_demand_features(
        cleaned,
        min_sales_days=10,
        top_n_products=10,
        cutoff_date="2011-11-01",
    )

    assert "SKU_A" in meta["StockCode"].values
    assert "SKU_FUTURE" not in meta["StockCode"].values, "SKU with < min_sales_days pre-cutoff must not enter model."
    assert pd.to_datetime(featured_df["Date"]).is_monotonic_increasing


def test_feature_strict_causality_no_future_leakage():
    dates = pd.date_range("2011-06-01", periods=50, freq="D")
    base = pd.DataFrame({
        "Date": dates,
        "DailyDemand": [float((i * 3) % 17 + 1) for i in range(50)],
        "ObservedUnitPrice": [2.0 + 0.05 * i if i % 3 != 0 else np.nan for i in range(50)],
    })
    perturbed = base.copy()
    # Mutate ONLY day index 40 onwards
    perturbed.loc[40:, "DailyDemand"] = 99999.0
    perturbed.loc[40:, "ObservedUnitPrice"] = 8888.0

    f_base = compute_product_features_vectorized(base)
    f_pert = compute_product_features_vectorized(perturbed)

    # At step <= 40, strictly past features (<= 39) MUST remain identical bitwise
    for col in NUMERICAL_FEATURES:
        pd.testing.assert_series_equal(f_base.loc[:40, col], f_pert.loc[:40, col], check_names=False)


def test_vectorized_vs_autoregressive_step_feature_parity():
    dates = pd.date_range("2011-05-01", periods=100, freq="D")
    demands = [float(0.0 if i % 5 == 0 else (i * 7) % 23 + 2) for i in range(100)]
    prices = [3.0 + 0.02 * i if i % 2 == 0 else np.nan for i in range(100)]
    df_prod = pd.DataFrame({
        "StockCode": "PARITY_SKU",
        "Date": dates,
        "DailyDemand": demands,
        "ObservedUnitPrice": prices,
    })

    vec = compute_product_features_vectorized(df_prod)
    # Trim warm-up just like training does (rows 30..99 -> 70 rows)
    trimmed = vec.iloc[30:].copy()

    # Roll forward step-by-step starting from history ending at trimmed row 39 (40 trimmed rows >= 30)
    history_slice = trimmed.iloc[:40].copy()
    state = extract_product_forecast_state(history_slice)

    for offset in range(1, 11):
        target_row_idx = 40 + offset - 1
        expected_row = trimmed.iloc[target_row_idx]
        target_date = pd.Timestamp(expected_row["Date"])

        step_features = build_feature_row_from_state(state, target_date)

        # Demand-derived, expanding-sum, sparsity, and calendar features MUST match vectorized to < 1e-9
        demand_cols = [c for c in NUMERICAL_FEATURES if c not in ("AvgUnitPrice", "hist_avg_price")]
        for col in demand_cols:
            assert np.isclose(
                float(step_features[col]), float(expected_row[col]), atol=1e-9
            ), f"Parity mismatch on '{col}' at step {offset}: {step_features[col]} != {expected_row[col]}"

        # Advance state with actual demand and observed price to mirror training row generation
        actual_t = float(expected_row["DailyDemand"])
        obs_t = expected_row["ObservedUnitPrice"]
        if pd.notna(obs_t):
            p_cnt_prev = int(round(float(expected_row["hist_price_count"])))
            state["hist_avg_price"] = (
                float(state["hist_avg_price"]) * p_cnt_prev + float(obs_t)
            ) / (p_cnt_prev + 1)
            state["last_unit_price"] = float(obs_t)
        advance_product_forecast_state(state, actual_t)


def test_chronological_target_encoder_handles_unseen_and_prevents_leakage():
    df = pd.DataFrame({
        "Date": pd.to_datetime([
            "2011-01-01", "2011-01-01",
            "2011-01-02", "2011-01-02",
            "2011-01-03", "2011-01-03",
        ]),
        "StockCode": ["A", "A", "A", "B", "A", "UNSEEN"],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    y = pd.Series([10.0, 20.0, 100.0, 200.0, 50.0, 999.0])

    enc = ChronologicalTargetEncoder(date_col="Date", cat_col="StockCode", smooth=10.0)
    tr = enc.fit_transform(df.iloc[:4], y.iloc[:4])

    # Day 2011-01-01 cannot see its own targets (10, 20); must fall back to prior
    assert tr[0, 0] == tr[1, 0]
    # Unseen category on test slice transforms cleanly to prior mean without raising
    te = enc.transform(df.iloc[4:])
    assert te.shape == (2, 1)
    assert np.isfinite(te).all()


def test_baselines_exact_math():
    hist = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    naive = recursive_naive_forecast(hist, forecast_days=4)
    assert naive == [70.0, 70.0, 70.0, 70.0]

    snaive = recursive_seasonal_naive_forecast(hist, forecast_days=9)
    assert snaive == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 10.0, 20.0]

    eval_df = pd.DataFrame({
        "DailyDemand": [10.0, 20.0],
        "lag_1": [10.0, 20.0],
        "lag_7": [8.0, 18.0],
    })
    bm = evaluate_single_step_baselines(eval_df)
    assert bm["Naive Baseline (t-1)"]["RMSE"] == 0.0
    assert bm["Seasonal Naive Baseline (t-7)"]["MAE"] == 2.0


def test_rolling_origin_backtest_no_future_leakage_multi_horizon():
    # Build 2 SKUs with 120 daily rows each after warm-up (150 raw days -> 120 rows)
    frames = []
    for sku, level in [("SKU_1", 20.0), ("SKU_2", 35.0)]:
        dates = pd.date_range("2011-04-01", periods=150, freq="D")
        df_p = pd.DataFrame({
            "StockCode": sku,
            "Date": dates,
            "DailyDemand": [level + (i % 7) * 2.0 for i in range(150)],
            "ObservedUnitPrice": [3.0 for _ in range(150)],
        })
        feat = compute_product_features_vectorized(df_p).iloc[30:].copy()
        frames.append(feat)

    featured_df = pd.concat(frames, ignore_index=True).sort_values(["Date", "StockCode"]).reset_index(drop=True)

    preproc = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["StockCode"]),
    ])
    pipe = Pipeline([("preprocessor", preproc), ("regressor", Ridge(alpha=1.0))])

    bt = run_rolling_origin_backtest(
        pipeline_template=pipe,
        featured_df=featured_df,
        cutoff_date="2011-08-29",
        n_origins=2,
        horizons=(7, 14, 30),
    )

    assert len(bt["origins"]) == 2
    for h_key in ["7d", "14d", "30d"]:
        assert h_key in bt["horizon_metrics"]
        assert bt["horizon_metrics"][h_key]["samples"] > 0
        assert bt["horizon_metrics"][h_key]["ml"]["RMSE"] >= 0
    assert bt["product_wise_summary"]["total_skus_evaluated"] == 2


def test_forecaster_future_prices_and_date_continuity(tmp_path):
    # Verify Date progression & future_prices override using real trained catalog + mock model if needed
    # Here we test directly against project's serialized DemandForecaster
    fc = DemandForecaster()
    assert fc.model is not None

    res_default = fc.forecast("85123A", forecast_days=30)
    assert len(res_default["forecast_schedule"]) == 30
    assert res_default["price_assumption"] == "carry_forward_last_historical_price"

    dates = pd.to_datetime([r["date"] for r in res_default["forecast_schedule"]])
    diffs = dates.to_series().diff().dropna().dt.days
    assert (diffs == 1).all(), "Forecast dates must increment sequentially by exactly 1 calendar day."
    assert all(r["predicted_demand"] >= 0.0 for r in res_default["forecast_schedule"])

    res_override = fc.forecast("85123A", forecast_days=7, future_prices=[9.99] * 7)
    assert res_override["price_assumption"] == "supplied_future_prices"
    assert res_override["assumed_unit_price"] == 9.99


def test_stockout_boundary_conditions_and_validation():
    # Pure unit boundary checks
    ss_0 = compute_safety_stock_buffer(lead_time_demand=100.0, safety_stock_threshold_pct=0.0)
    assert ss_0 == 0.0

    ss_20 = compute_safety_stock_buffer(lead_time_demand=100.0, safety_stock_threshold_pct=0.20)
    assert ss_20 == 20.0

    # Exact threshold: current=120, demand=100 -> remaining=20 == ss=20 -> LOW
    status_eq, rem_eq, _ = classify_stockout_risk(120, 100.0, ss_20)
    assert status_eq == "LOW"
    assert rem_eq == 20.0
    assert compute_replenishment_quantity(status_eq, 120, 100.0, ss_20) == 0

    # Strictly below safety threshold: current=119 -> remaining=19 < 20 -> HIGH
    status_low, rem_low, _ = classify_stockout_risk(119, 100.0, ss_20)
    assert status_low == "HIGH"
    assert rem_low == 19.0
    assert compute_replenishment_quantity(status_low, 119, 100.0, ss_20) == 1

    # Zero demand with zero stock & zero buffer: 0 - 0 = 0 >= 0 -> LOW
    assert compute_safety_stock_buffer(0.0, 0.20) == 0.0
    status_z, _, _ = classify_stockout_risk(0, 0.0, 0.0)
    assert status_z == "LOW"

    # End-to-end validation checks
    with pytest.raises(ValueError):
        analyze_stockout_risk("85123A", current_inventory=-1, supplier_lead_time=7)
    with pytest.raises(ValueError):
        analyze_stockout_risk("85123A", current_inventory=50, supplier_lead_time=0)
    with pytest.raises(ValueError):
        analyze_stockout_risk("85123A", current_inventory=50, supplier_lead_time=7, safety_stock_threshold_pct=1.5)
