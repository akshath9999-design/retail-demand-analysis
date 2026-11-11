"""
tests/test_chronological_encoder.py
Comprehensive test suite verifying zero-leakage properties of ChronologicalTargetEncoder:
  1. Future-target leakage prevention
  2. Same-day target leakage prevention
  3. Fixed initial prior (default 0.0) with zero Day 0 target influence
  4. Exact row-order preservation on shuffled inputs
  5. Unseen SKU handling during transform
  6. Validation transform uses training statistics only (no updates or target access)
  7. Scikit-learn clone, Pipeline, and ColumnTransformer compatibility
  8. RandomizedSearchCV integration with date-based cross-validation splits
  9. Empty and invalid inputs error handling
 10. Correct transformed feature count (exactly 18 features out from 17 num + 2 cat/date)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from backend.temporal_encoder import ChronologicalTargetEncoder
from backend.temporal_validation import make_date_splits
from backend.train import build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS


@pytest.fixture
def synthetic_panel_df():
    """Generates a small multi-product, multi-date synthetic dataset."""
    dates = [
        date(2021, 1, 1),
        date(2021, 1, 2),
        date(2021, 1, 3),
        date(2021, 1, 4),
        date(2021, 1, 5)
    ]
    rows = []
    for d in dates:
        for sku in ['SKU_A', 'SKU_B', 'SKU_C']:
            rows.append({
                'Date': d,
                'StockCode': sku,
                'DailyDemand': 10.0 if sku == 'SKU_A' else (20.0 if sku == 'SKU_B' else 30.0)
            })
    return pd.DataFrame(rows)


def test_fixed_initial_prior_no_day0_target_influence():
    """Requirement 4: Initial date encodings must equal prior_mean (default 0.0) regardless of Day 0 targets."""
    df = pd.DataFrame({
        'Date': [date(2021, 1, 1), date(2021, 1, 1)],
        'StockCode': ['SKU_A', 'SKU_B'],
        'DailyDemand': [9999.0, 50000.0]
    })
    encoder = ChronologicalTargetEncoder(date_col='Date', cat_col='StockCode', smooth=10.0, prior_mean=0.0)
    enc = encoder.fit_transform(df[['StockCode', 'Date']], df['DailyDemand'])

    assert enc.shape == (2, 1)
    # Day 0 encodings must be exactly the fixed prior (0.0), completely unaffected by 9999.0 or 50000.0
    assert np.allclose(enc[:, 0], [0.0, 0.0])


def test_future_target_leakage():
    """Requirement 3 & 10: Changing targets on future dates (D > D_current) must have 0 impact on earlier encodings."""
    dates = [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]
    df = pd.DataFrame({
        'Date': [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
        'StockCode': ['A', 'B', 'A', 'B', 'A', 'B'],
        'DailyDemand': [10.0, 20.0, 15.0, 25.0, 100.0, 200.0]
    })
    
    # Run baseline encoder
    enc_baseline = ChronologicalTargetEncoder(smooth=5.0, prior_mean=0.0)
    res_baseline = enc_baseline.fit_transform(df[['StockCode', 'Date']], df['DailyDemand'])

    # Modify Day 3 (future) targets dramatically
    df_altered = df.copy()
    df_altered.loc[df_altered['Date'] == dates[2], 'DailyDemand'] = [99999.0, 88888.0]

    enc_altered = ChronologicalTargetEncoder(smooth=5.0, prior_mean=0.0)
    res_altered = enc_altered.fit_transform(df_altered[['StockCode', 'Date']], df_altered['DailyDemand'])

    # Encodings for Day 1 and Day 2 (indices 0, 1, 2, 3) must be strictly identical
    np.testing.assert_allclose(res_baseline[:4, 0], res_altered[:4, 0], err_msg="Future targets leaked into earlier dates!")
    
    # Encodings for Day 3 must also be identical because Day 3 uses history < Day 3
    np.testing.assert_allclose(res_baseline[4:, 0], res_altered[4:, 0], err_msg="Day 3 target leaked into Day 3 encoding!")


def test_same_day_target_leakage():
    """Requirement 3 & 10: Same-day targets must never influence same-day encodings."""
    d1 = date(2021, 1, 1)
    d2 = date(2021, 1, 2)
    # On day 2, SKU 'A' appears twice with different targets: 100.0 and 500.0
    df = pd.DataFrame({
        'Date': [d1, d1, d2, d2],
        'StockCode': ['A', 'B', 'A', 'A'],
        'DailyDemand': [10.0, 20.0, 100.0, 500.0]
    })

    encoder = ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0)
    enc = encoder.fit_transform(df[['StockCode', 'Date']], df['DailyDemand'])

    # Both rows of SKU 'A' on day 2 must receive the EXACT same encoding based strictly on day 1
    # Day 1 history: SKU 'A' sum=10, count=1. Global sum=30, count=2 -> global_mean = 15.0
    # Expected: (10 + 10 * 15) / (1 + 10) = 160 / 11 ≈ 14.54545
    expected_day2_A = (10.0 + 10.0 * 15.0) / (1.0 + 10.0)
    assert np.isclose(enc[2, 0], expected_day2_A)
    assert np.isclose(enc[3, 0], expected_day2_A)
    assert enc[2, 0] == enc[3, 0]


def test_row_order_preservation(synthetic_panel_df):
    """Requirement 2 & 10: Shuffling input rows must not change their relative output encodings."""
    df_sorted = synthetic_panel_df.sort_values(['Date', 'StockCode']).reset_index(drop=True)
    
    enc1 = ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0)
    res_sorted = enc1.fit_transform(df_sorted[['StockCode', 'Date']], df_sorted['DailyDemand'])
    df_sorted['encoded'] = res_sorted[:, 0]

    # Shuffle the dataframe completely
    df_shuffled = df_sorted.sample(frac=1.0, random_state=42).reset_index(drop=True)
    expected_shuffled_enc = df_shuffled['encoded'].to_numpy()

    enc2 = ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0)
    res_shuffled = enc2.fit_transform(df_shuffled[['StockCode', 'Date']], df_shuffled['DailyDemand'])

    np.testing.assert_allclose(res_shuffled[:, 0], expected_shuffled_enc)


def test_non_contiguous_index_alignment():
    """Verify that DataFrames and Series with filtered/arbitrary indices align properly."""
    df = pd.DataFrame({
        'Date': [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
        'StockCode': ['A', 'A', 'A'],
        'DailyDemand': [10.0, 20.0, 30.0]
    }, index=[105, 42, 999])

    encoder = ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0)
    enc = encoder.fit_transform(df[['StockCode', 'Date']], df['DailyDemand'])

    assert enc.shape == (3, 1)
    # Day 1: prior (0.0)
    assert enc[0, 0] == 0.0
    # Day 2: (10 + 10 * 10) / (1 + 10) = 110 / 11 = 10.0
    assert np.isclose(enc[1, 0], 10.0)
    # Day 3: (30 + 10 * 15) / (2 + 10) = 180 / 12 = 15.0
    assert np.isclose(enc[2, 0], 15.0)


def test_unseen_sku_handling():
    """Requirement 6 & 10: Unseen SKUs in transform() receive the final global training mean."""
    train_df = pd.DataFrame({
        'Date': [date(2021, 1, 1), date(2021, 1, 2)],
        'StockCode': ['A', 'B'],
        'DailyDemand': [10.0, 30.0]
    })
    encoder = ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0)
    encoder.fit(train_df[['StockCode', 'Date']], train_df['DailyDemand'])

    # Final training global mean = (10 + 30) / 2 = 20.0
    assert encoder.final_global_mean_ == 20.0

    test_df = pd.DataFrame({
        'Date': [date(2021, 1, 3)],
        'StockCode': ['NEW_UNKNOWN_SKU']
    })
    test_enc = encoder.transform(test_df[['StockCode', 'Date']])

    assert test_enc.shape == (1, 1)
    assert np.isclose(test_enc[0, 0], 20.0)


def test_validation_transform_uses_training_statistics_only():
    """Requirement 6 & 10: transform() does not update cumulative state or access validation targets."""
    train_df = pd.DataFrame({
        'Date': [date(2021, 1, 1), date(2021, 1, 2)],
        'StockCode': ['A', 'B'],
        'DailyDemand': [10.0, 20.0]
    })
    encoder = ChronologicalTargetEncoder(smooth=5.0, prior_mean=0.0)
    encoder.fit(train_df[['StockCode', 'Date']], train_df['DailyDemand'])

    val_df = pd.DataFrame({
        'Date': [date(2021, 1, 3), date(2021, 1, 4)],
        'StockCode': ['A', 'A']
    })

    # Multiple successive transforms on validation data must yield identical results
    val_enc1 = encoder.transform(val_df[['StockCode', 'Date']])
    val_enc2 = encoder.transform(val_df[['StockCode', 'Date']])

    np.testing.assert_allclose(val_enc1, val_enc2)
    # The statistics must not have been modified
    assert encoder.final_global_mean_ == 15.0
    assert encoder.final_sku_count_['A'] == 1


def test_sklearn_clone_and_pipeline_compatibility():
    """Requirement 7 & 10: Transformer must clone cleanly and integrate into a Pipeline."""
    encoder = ChronologicalTargetEncoder(smooth=12.5, prior_mean=5.0)
    cloned_encoder = clone(encoder)

    assert cloned_encoder.smooth == 12.5
    assert cloned_encoder.prior_mean == 5.0
    assert not hasattr(cloned_encoder, 'is_fitted_')

    # Test within ColumnTransformer and Pipeline
    ct = ColumnTransformer(
        transformers=[
            ('cat', cloned_encoder, ['StockCode', 'Date'])
        ],
        remainder='drop'
    )
    pipe = Pipeline([
        ('ct', ct),
        ('model', Ridge())
    ])

    df = pd.DataFrame({
        'Date': [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)],
        'StockCode': ['A', 'B', 'A'],
        'DailyDemand': [5.0, 15.0, 25.0]
    })

    pipe.fit(df[['StockCode', 'Date']], df['DailyDemand'])
    preds = pipe.predict(df[['StockCode', 'Date']])
    assert len(preds) == len(df)


def test_randomized_search_cv_with_date_splits():
    """Requirement 7, 9 & 10: Pipeline with ChronologicalTargetEncoder tunes cleanly in RandomizedSearchCV."""
    # Create 15 days of data across 3 products
    dates = [date(2021, 1, d) for d in range(1, 16)]
    rows = []
    for d in dates:
        for sku in ['P1', 'P2']:
            rows.append({
                'Date': d,
                'StockCode': sku,
                'num_feat': float(d.day),
                'DailyDemand': float(d.day * 2)
            })
    df = pd.DataFrame(rows)

    ct = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['num_feat']),
            ('cat', ChronologicalTargetEncoder(smooth=10.0, prior_mean=0.0), ['StockCode', 'Date'])
        ]
    )
    pipe = Pipeline([
        ('preprocessor', ct),
        ('regressor', Ridge())
    ])

    splits = make_date_splits(df[['Date']], n_splits=3, date_col='Date')
    param_dist = {
        'regressor__alpha': [0.1, 1.0, 10.0]
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=3,
        cv=splits,
        scoring='neg_mean_absolute_error',
        random_state=42
    )

    X = df[['StockCode', 'Date', 'num_feat']]
    y = df['DailyDemand']
    search.fit(X, y)

    assert search.best_estimator_ is not None
    assert search.best_score_ is not None
    best_preds = search.predict(X)
    assert len(best_preds) == len(X)


def test_empty_and_invalid_inputs():
    """Requirement 10: Encoder raises descriptive errors for null, empty, or mismatched inputs."""
    encoder = ChronologicalTargetEncoder()

    with pytest.raises(ValueError, match="cannot be None"):
        encoder.fit_transform(None, [1, 2])

    with pytest.raises(ValueError, match="is empty"):
        encoder.fit_transform(pd.DataFrame(), [1])

    df = pd.DataFrame({'StockCode': ['A'], 'Date': [None]})
    with pytest.raises(ValueError, match="contains null, NaN, or NaT"):
        encoder.fit_transform(df, [10.0])

    df_valid = pd.DataFrame({'StockCode': ['A'], 'Date': [date(2021, 1, 1)]})
    with pytest.raises(ValueError, match="Target y is required"):
        encoder.fit_transform(df_valid, None)

    with pytest.raises(ValueError, match="Length mismatch"):
        encoder.fit_transform(df_valid, [10.0, 20.0])


def test_correct_transformed_feature_count():
    """Requirement 8 & 10: Passing FEATURE_COLUMNS through build_preprocessor() produces exactly 18 features."""
    preprocessor = build_preprocessor()

    # Create dummy row containing all FEATURE_COLUMNS
    dummy_data = {col: [1.0] for col in NUMERICAL_FEATURES}
    dummy_data['StockCode'] = ['TEST_SKU']
    dummy_data['Date'] = [date(2021, 1, 1)]
    df = pd.DataFrame(dummy_data)
    y = np.array([5.0])

    X_trans = preprocessor.fit_transform(df[FEATURE_COLUMNS], y)

    # 17 numerical + 1 categorical = 18 features (Date is NOT emitted)
    assert X_trans.shape == (1, 18), f"Expected 18 features, got {X_trans.shape[1]}"
    assert len(NUMERICAL_FEATURES) == 17
    assert 'Date' not in [t[0] for t in preprocessor.transformers_ if t[0] != 'remainder']
