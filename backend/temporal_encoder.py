"""
backend/temporal_encoder.py
Chronological, expanding-window target encoder for time-series and panel data.
Guarantees zero temporal target leakage:
  - For any date D, encodings are calculated strictly using target observations from dates < D.
  - Same-day targets never influence same-day encodings.
  - Initial dates use a fixed configurable prior (default 0.0), never Day 0 targets.
  - Emits exactly 1 numeric encoded feature, preserving row order.
  - Scikit-learn compatible (inherits BaseEstimator, TransformerMixin).
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class ChronologicalTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Chronological out-of-past Target Encoder.

    Parameters:
      date_col: Name or position of the date column in X (default 'Date').
      cat_col: Name or position of the categorical column in X (default 'StockCode').
      smooth: Smoothing weight m for m-estimate empirical Bayes shrinkage (default 10.0).
              Formula: (prior_sku_target_sum + smooth * prior_global_mean) / (prior_sku_count + smooth)
      prior_mean: Fixed neutral baseline prior for the initial date(s) before any historical
                  observations exist (default 0.0). Never accesses Day 0 targets.
    """

    def __init__(
        self,
        date_col: str = "Date",
        cat_col: str = "StockCode",
        smooth: float = 10.0,
        prior_mean: float = 0.0
    ):
        self.date_col = date_col
        self.cat_col = cat_col
        self.smooth = smooth
        self.prior_mean = prior_mean

    def _extract_inputs(self, X):
        """Extracts category array and date array while preserving row count."""
        if X is None:
            raise ValueError("Input X cannot be None.")
        if len(X) == 0:
            raise ValueError("Input X is empty.")

        if isinstance(X, pd.DataFrame):
            # Extract categorical column
            if self.cat_col in X.columns:
                cats = X[self.cat_col].astype(str).to_numpy()
            else:
                cats = X.iloc[:, 0].astype(str).to_numpy()

            # Extract date column
            if self.date_col in X.columns:
                raw_dates = X[self.date_col]
            else:
                raw_dates = X.iloc[:, 1]
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] < 2:
                raise ValueError(f"Expected 2D array with at least 2 columns (cat, date), got shape {X.shape}")
            cats = X[:, 0].astype(str)
            raw_dates = X[:, 1]
        else:
            raise TypeError(f"Unsupported data type for X: {type(X).__name__}")

        try:
            dates = pd.to_datetime(raw_dates).dt.date.to_numpy()
        except Exception as e:
            raise ValueError(f"Failed to parse date column into dates: {str(e)}")

        if pd.Series(dates).isna().any():
            raise ValueError("Date data contains null, NaN, or NaT values.")

        return cats, dates

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits on X, y using strictly out-of-past expanding-window target statistics.
        Returns a 2D array of shape (n_samples, 1) preserving exact input row order.
        """
        if y is None:
            raise ValueError("Target y is required for fit_transform in ChronologicalTargetEncoder.")

        cats, dates = self._extract_inputs(X)
        y_arr = np.asarray(y, dtype=np.float64)

        if len(cats) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(cats)} rows, y has {len(y_arr)} values.")

        n_samples = len(cats)
        out_encodings = np.empty((n_samples, 1), dtype=np.float64)

        # Extract sorted unique dates to process strictly chronologically
        unique_dates = np.sort(pd.Series(dates).unique())

        # Cumulative statistics strictly prior to current date
        sku_target_sum = {}
        sku_target_count = {}
        global_target_sum = 0.0
        global_target_count = 0

        # Fixed prior mean for initial dates with no prior history (default 0.0)
        fixed_prior = float(self.prior_mean)

        for current_date in unique_dates:
            # Find all rows belonging to current date
            mask_d = (dates == current_date)
            idx_d = np.flatnonzero(mask_d)

            # Global historical mean strictly prior to current_date
            if global_target_count > 0:
                current_global_mean = global_target_sum / global_target_count
            else:
                # Initial dates before any prior history: strictly use fixed prior
                current_global_mean = fixed_prior

            # Step 1: Calculate encodings for ALL rows on current_date using only history < current_date
            for idx in idx_d:
                cat = cats[idx]
                prior_sum = sku_target_sum.get(cat, 0.0)
                prior_count = sku_target_count.get(cat, 0)

                # Smoothing formula: (prior_sku_sum + smooth * current_global_mean) / (prior_count + smooth)
                encoded_val = (prior_sum + self.smooth * current_global_mean) / (prior_count + self.smooth)
                out_encodings[idx, 0] = encoded_val

            # Step 2: SAME-DAY PROTECTION:
            # Update cumulative statistics ONLY after every row on current_date has been encoded
            for idx in idx_d:
                cat = cats[idx]
                target_val = y_arr[idx]
                sku_target_sum[cat] = sku_target_sum.get(cat, 0.0) + target_val
                sku_target_count[cat] = sku_target_count.get(cat, 0) + 1
                global_target_sum += target_val
                global_target_count += 1

        # Retain final training statistics for future transform()
        self.final_global_mean_ = (
            global_target_sum / global_target_count if global_target_count > 0 else fixed_prior
        )
        self.final_sku_sum_ = sku_target_sum
        self.final_sku_count_ = sku_target_count

        # Precompute final smoothed encodings for all observed categories
        final_encodings = {}
        for cat, s_sum in sku_target_sum.items():
            s_cnt = sku_target_count[cat]
            final_encodings[cat] = (s_sum + self.smooth * self.final_global_mean_) / (s_cnt + self.smooth)

        self.final_encodings_ = final_encodings
        self.is_fitted_ = True

        return out_encodings

    def fit(self, X, y=None, **fit_params):
        """Fits the encoder using fit_transform."""
        self.fit_transform(X, y, **fit_params)
        return self

    def transform(self, X):
        """
        Transforms X using the precomputed final training encodings.
        Never accesses validation/test targets.
        Unseen SKUs receive the final global training mean.
        Returns a 2D array of shape (n_samples, 1).
        """
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError("ChronologicalTargetEncoder is not fitted. Call fit or fit_transform first.")

        cats, _ = self._extract_inputs(X)
        n_samples = len(cats)
        out_encodings = np.empty((n_samples, 1), dtype=np.float64)

        fallback = self.final_global_mean_
        final_map = self.final_encodings_

        for i in range(n_samples):
            cat = cats[i]
            out_encodings[i, 0] = final_map.get(cat, fallback)

        return out_encodings

    def get_feature_names_out(self, input_features=None):
        """Returns feature names out for sklearn ColumnTransformer compatibility."""
        return np.array([f"{self.cat_col}_encoded"], dtype=object)

