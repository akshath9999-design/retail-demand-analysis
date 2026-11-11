"""
tests/test_temporal_validation.py
Unit tests verifying date-based expanding-window chronological cross-validation.
Checks:
  1. Training dates precede validation dates (strictly earlier).
  2. No dates overlap between training and validation.
  3. Products from the same date are never divided across folds.
  4. Training windows expand chronologically across successive folds.
  5. Invalid input is handled properly (empty data, bad n_splits, insufficient dates, missing date column, NaNs).
  6. Compatibility with scikit-learn search estimators using positional indices.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

from backend.temporal_validation import make_date_splits, get_date_split_ranges


@pytest.fixture
def multi_product_daily_df():
    """
    Creates a simulated multi-product daily sales DataFrame where multiple products
    (SKU_A, SKU_B, SKU_C) are grouped together on each date across 20 distinct days.
    """
    dates = pd.date_range("2021-01-01", periods=20, freq="D")
    products = ["SKU_A", "SKU_B", "SKU_C"]

    rows = []
    for d in dates:
        for p in products:
            rows.append({
                "Date": d,
                "StockCode": p,
                "Demand": np.random.randint(5, 50),
                "Price": 2.5
            })

    # Shuffle row order to simulate non-sorted concatenation
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def test_training_dates_precede_validation_dates(multi_product_daily_df):
    """Verify for every fold that every training date is strictly earlier than every validation date."""
    df = multi_product_daily_df
    splits = make_date_splits(df, n_splits=3, date_col="Date")
    assert len(splits) == 3

    date_array = pd.to_datetime(df["Date"]).dt.date.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_dates = date_array[train_idx]
        val_dates = date_array[val_idx]

        max_train_date = train_dates.max()
        min_val_date = val_dates.min()

        assert max_train_date < min_val_date, (
            f"Fold {fold_idx}: Max training date ({max_train_date}) must be strictly earlier "
            f"than min validation date ({min_val_date})."
        )


def test_no_dates_overlap_between_train_and_val(multi_product_daily_df):
    """Verify that train and validation date sets in each fold have zero overlap (disjoint)."""
    df = multi_product_daily_df
    splits = make_date_splits(df, n_splits=4, date_col="Date")

    date_array = pd.to_datetime(df["Date"]).dt.date.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_date_set = set(date_array[train_idx])
        val_date_set = set(date_array[val_idx])

        overlap = train_date_set.intersection(val_date_set)
        assert len(overlap) == 0, f"Fold {fold_idx} has overlapping dates: {overlap}"


def test_products_from_same_date_never_divided_across_folds(multi_product_daily_df):
    """
    Verify that all products belonging to the same date are assigned together.
    No date should have some products in train and some in validation, or split across sets.
    """
    df = multi_product_daily_df
    splits = make_date_splits(df, n_splits=3, date_col="Date")

    date_array = pd.to_datetime(df["Date"]).dt.date.to_numpy()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_dates = set(date_array[train_idx])
        val_dates = set(date_array[val_idx])

        # Check for every unique date in the dataset
        for d in np.unique(date_array):
            date_mask = (date_array == d)
            date_indices = set(np.flatnonzero(date_mask))

            in_train = date_indices.issubset(set(train_idx))
            in_val = date_indices.issubset(set(val_idx))
            in_neither = date_indices.isdisjoint(set(train_idx)) and date_indices.isdisjoint(set(val_idx))

            # Date rows must be either ALL in train, ALL in val, or ALL unselected for this fold
            assert in_train or in_val or in_neither, (
                f"Fold {fold_idx}: Date {d} rows were partially split between sets!"
            )


def test_training_windows_expand_chronologically(multi_product_daily_df):
    """
    Verify that training windows expand chronologically:
    Fold 0 train count < Fold 1 train count < Fold 2 train count.
    Also verify earlier train dates are subsets of later train dates.
    """
    df = multi_product_daily_df
    splits = make_date_splits(df, n_splits=3, date_col="Date")

    date_array = pd.to_datetime(df["Date"]).dt.date.to_numpy()

    prev_train_dates = set()
    prev_train_size = 0

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        curr_train_dates = set(date_array[train_idx])
        curr_train_size = len(train_idx)

        assert curr_train_size > prev_train_size, (
            f"Fold {fold_idx} train size ({curr_train_size}) did not expand beyond fold {fold_idx - 1} ({prev_train_size})."
        )
        assert prev_train_dates.issubset(curr_train_dates), (
            f"Fold {fold_idx} training set does not contain all training dates from previous fold."
        )

        prev_train_dates = curr_train_dates
        prev_train_size = curr_train_size


def test_invalid_input_handling():
    """Verify that invalid inputs raise descriptive ValueErrors or TypeErrors."""
    # 1. Empty DataFrame
    with pytest.raises(ValueError, match="empty"):
        make_date_splits(pd.DataFrame(), n_splits=3)

    # 2. None input
    with pytest.raises(ValueError, match="None"):
        make_date_splits(None, n_splits=3)

    # 3. Invalid n_splits (< 1 or non-integer)
    sample_df = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=10)})
    with pytest.raises(ValueError, match="integer >= 1"):
        make_date_splits(sample_df, n_splits=0)
    with pytest.raises(ValueError, match="integer >= 1"):
        make_date_splits(sample_df, n_splits="three")

    # 4. Missing date column
    with pytest.raises(ValueError, match="not found"):
        make_date_splits(sample_df, n_splits=3, date_col="NonExistentDate")

    # 5. Insufficient unique dates (e.g. 3 dates for 3 splits requires at least 4 dates)
    short_df = pd.DataFrame({"Date": ["2021-01-01", "2021-01-02", "2021-01-03"]})
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        make_date_splits(short_df, n_splits=3)

    # 6. Dates with NaNs
    nan_df = pd.DataFrame({"Date": ["2021-01-01", None, "2021-01-03", "2021-01-04"]})
    with pytest.raises(ValueError, match="null, NaN, or NaT"):
        make_date_splits(nan_df, n_splits=2)


def test_sklearn_randomizedsearch_compatibility(multi_product_daily_df):
    """
    Verify that the generated positional indices integrate seamlessly
    with scikit-learn's RandomizedSearchCV.
    """
    df = multi_product_daily_df
    # Add non-sequential index to verify positional index safety
    df.index = np.random.permutation(len(df)) * 7 + 100

    splits = make_date_splits(df, n_splits=3, date_col="Date")

    X = df[["Demand", "Price"]]
    y = df["Demand"] * 2 + 1.5

    search = RandomizedSearchCV(
        estimator=Ridge(),
        param_distributions={"alpha": [0.1, 1.0, 10.0]},
        n_iter=3,
        cv=splits,
        random_state=42
    )

    # Must fit without error
    search.fit(X, y)
    assert search.best_estimator_ is not None
    assert search.best_score_ is not None


def test_get_date_split_ranges(multi_product_daily_df):
    """Verify that get_date_split_ranges returns accurate metadata."""
    ranges = get_date_split_ranges(multi_product_daily_df, n_splits=3, date_col="Date")
    assert len(ranges) == 3
    for r in ranges:
        assert r["train_start"] <= r["train_end"]
        assert r["val_start"] <= r["val_end"]
        assert r["train_end"] < r["val_start"]
        assert r["train_samples"] > 0
        assert r["val_samples"] > 0
