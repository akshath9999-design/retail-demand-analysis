"""
backend/temporal_validation.py
Reusable chronological date-based cross-validation module.
Provides expanding-window time-series splits where:
  - All rows/products sharing the same date stay in the same fold.
  - Every training date strictly precedes every validation date.
  - Training windows expand chronologically.
  - Returns positional row indices compatible with scikit-learn search estimators.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd


def _extract_date_series(
    data: Union[pd.DataFrame, pd.Series],
    date_col: str = "Date"
) -> pd.Series:
    """Extracts and validates a normalized datetime.date Series from DataFrame or Series."""
    if data is None:
        raise ValueError("Input data cannot be None.")

    if len(data) == 0:
        raise ValueError("Input data is empty.")

    if isinstance(data, pd.Series):
        raw_dates = data
    elif isinstance(data, pd.DataFrame):
        if date_col in data.columns:
            raw_dates = data[date_col]
        elif date_col.lower() in data.columns:
            raw_dates = data[date_col.lower()]
        else:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame columns: {list(data.columns)}")
    else:
        raise TypeError(f"Expected pandas DataFrame or Series, got {type(data).__name__}.")

    if raw_dates.isna().any():
        raise ValueError("Date data contains null, NaN, or NaT values.")

    try:
        # Convert to datetime and normalize to date objects for clean equality checks
        parsed_dates = pd.to_datetime(raw_dates).dt.date
    except Exception as e:
        raise ValueError(f"Failed to parse dates into datetime format: {str(e)}")

    return parsed_dates


def make_date_splits(
    df: Union[pd.DataFrame, pd.Series],
    n_splits: int = 3,
    date_col: str = "Date"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates an expanding-window chronological cross-validation split based on unique dates.

    Requirements satisfied:
      - Unique dates sorted chronologically.
      - Dates divided into expanding training and subsequent validation windows.
      - All products belonging to the same date remain together in the same fold.
      - Every training date is strictly earlier than every validation date in that fold.
      - No validation date appears in its corresponding training set.
      - Returns positional indices (0 to len(df)-1) compatible with scikit-learn.
      - Raises clear ValueErrors for insufficient dates, invalid n_splits, or empty folds.

    Parameters:
      df: pd.DataFrame with a date column, or pd.Series of dates.
      n_splits: Number of cross-validation folds (expanding windows).
      date_col: Name of the date column if df is a DataFrame (default 'Date').

    Returns:
      List of (train_indices, val_indices) tuples of numpy integer arrays.
    """
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError(f"n_splits must be an integer >= 1, got {n_splits}.")

    date_series = _extract_date_series(df, date_col=date_col)

    # Extract unique dates and sort them strictly in ascending chronological order
    unique_dates = np.sort(pd.Series(date_series.unique()).values)
    n_unique_dates = len(unique_dates)

    if n_unique_dates < n_splits + 1:
        raise ValueError(
            f"Insufficient unique dates ({n_unique_dates}) for n_splits={n_splits}. "
            f"At least {n_splits + 1} unique dates are required to form {n_splits} expanding folds."
        )

    # Partition unique dates into (n_splits + 1) contiguous chunks
    date_chunks = np.array_split(unique_dates, n_splits + 1)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    date_array = date_series.to_numpy()

    for fold_idx in range(n_splits):
        # Expanding training window: all chunks up to fold_idx
        train_dates = np.concatenate(date_chunks[: fold_idx + 1])
        # Subsequent validation window: chunk fold_idx + 1
        val_dates = date_chunks[fold_idx + 1]

        if len(train_dates) == 0:
            raise ValueError(f"Fold {fold_idx}: Training date window is empty.")
        if len(val_dates) == 0:
            raise ValueError(f"Fold {fold_idx}: Validation date window is empty.")

        # Sanity check: strictly earlier and disjoint
        max_train_date = train_dates[-1]
        min_val_date = val_dates[0]
        if max_train_date >= min_val_date:
            raise ValueError(
                f"Fold {fold_idx}: Chronological violation detected! "
                f"Max train date ({max_train_date}) >= min validation date ({min_val_date})."
            )

        train_date_set = set(train_dates)
        val_date_set = set(val_dates)

        if not train_date_set.isdisjoint(val_date_set):
            raise ValueError(f"Fold {fold_idx}: Overlapping dates detected between train and validation sets.")

        # Assign rows based strictly on date membership
        train_mask = np.isin(date_array, train_dates)
        val_mask = np.isin(date_array, val_dates)

        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)

        if len(train_indices) == 0:
            raise ValueError(f"Fold {fold_idx}: No row samples mapped to training dates.")
        if len(val_indices) == 0:
            raise ValueError(f"Fold {fold_idx}: No row samples mapped to validation dates.")

        splits.append((train_indices, val_indices))

    return splits


def get_date_split_ranges(
    df: Union[pd.DataFrame, pd.Series],
    n_splits: int = 3,
    date_col: str = "Date"
) -> List[Dict[str, Any]]:
    """
    Convenience inspector function returning date ranges, days, and sample counts for each fold.
    """
    date_series = _extract_date_series(df, date_col=date_col)
    splits = make_date_splits(df, n_splits=n_splits, date_col=date_col)
    
    date_array = date_series.to_numpy()
    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_dates = date_array[train_idx]
        val_dates = date_array[val_idx]

        fold_summaries.append({
            "fold": fold_idx + 1,
            "train_start": str(train_dates.min()),
            "train_end": str(train_dates.max()),
            "train_unique_dates": int(len(np.unique(train_dates))),
            "train_samples": int(len(train_idx)),
            "val_start": str(val_dates.min()),
            "val_end": str(val_dates.max()),
            "val_unique_dates": int(len(np.unique(val_dates))),
            "val_samples": int(len(val_idx)),
        })

    return fold_summaries
