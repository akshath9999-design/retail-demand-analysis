"""
backend/features.py
Shared, strictly causal feature engineering definitions for training,
rolling-origin backtesting, and recursive autoregressive forecasting.

Guarantees 100% training-inference feature parity and zero future leakage:
- All lag, rolling, variability, product-history, and price features are strictly past-looking (<= t - 1).
- Standard deviation calculations uniformly use sample ddof=1 across training and inference.
- Calendar features depend exclusively on the target prediction Date.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd

# Official England & Wales / UK Bank Holidays covering the dataset timeline (2010-2012)
UK_BANK_HOLIDAYS = {
    date(2010, 12, 25),
    date(2010, 12, 26),
    date(2010, 12, 27),
    date(2010, 12, 28),
    date(2011, 1, 1),
    date(2011, 1, 3),
    date(2011, 4, 22),  # Good Friday
    date(2011, 4, 25),  # Easter Monday
    date(2011, 4, 29),  # Royal Wedding
    date(2011, 5, 2),   # Early May bank holiday
    date(2011, 5, 30),  # Spring bank holiday
    date(2011, 8, 29),  # Summer bank holiday
    date(2011, 12, 25),
    date(2011, 12, 26),
    date(2011, 12, 27),
    date(2012, 1, 1),
    date(2012, 1, 2),
}

NUMERICAL_FEATURES: List[str] = [
    # Calendar indicators
    'day',
    'month',
    'year',
    'dayofweek',
    'is_weekend',
    'weekofyear',
    'is_month_end',
    'is_uk_holiday',
    # Past demand lags
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_30',
    # Past rolling averages & variability (ddof=1)
    'rolling_mean_7',
    'rolling_mean_14',
    'rolling_mean_30',
    'rolling_std_7',
    'rolling_std_14',
    'rolling_std_30',
    'rolling_trend',
    'cv_7',
    # Strictly historical product-level demand dynamics
    'hist_expanding_mean',
    'zero_demand_ratio_14',
    'days_since_last_sale',
    # Strictly historical price features (<= t - 1)
    'AvgUnitPrice',
    'hist_avg_price',
]

CATEGORICAL_FEATURES: List[str] = ['StockCode']
FEATURE_COLUMNS: List[str] = ['StockCode', 'Date'] + NUMERICAL_FEATURES
TARGET: str = 'DailyDemand'


def _to_date(dt_val: Union[date, datetime, pd.Timestamp]) -> date:
    if isinstance(dt_val, pd.Timestamp):
        return dt_val.date()
    if isinstance(dt_val, datetime):
        return dt_val.date()
    return dt_val


def compute_product_features_vectorized(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strictly past-looking features for a single product's complete daily dataframe.
    Expects `df_prod` sorted chronologically by `Date` with contiguous daily rows and columns:
      - Date (datetime64)
      - DailyDemand (float)
      - ObservedUnitPrice (float or NaN on days without transactions)
    """
    out = df_prod.copy()
    dates = pd.to_datetime(out['Date'])
    date_objs = dates.dt.date

    # 1. Calendar features (known strictly from Date)
    out['day'] = dates.dt.day
    out['month'] = dates.dt.month
    out['year'] = dates.dt.year
    out['dayofweek'] = dates.dt.dayofweek
    out['is_weekend'] = out['dayofweek'].isin([5, 6]).astype(int)
    out['weekofyear'] = dates.dt.isocalendar().week.astype(int)
    out['is_month_end'] = dates.dt.is_month_end.astype(int)
    out['is_uk_holiday'] = date_objs.isin(UK_BANK_HOLIDAYS).astype(int)

    # 2. Strictly past demand lags
    demand = out['DailyDemand'].astype(float)
    out['lag_1'] = demand.shift(1)
    out['lag_7'] = demand.shift(7)
    out['lag_14'] = demand.shift(14)
    out['lag_30'] = demand.shift(30)

    # 3. Rolling statistics strictly on shifted demand (<= t - 1) using sample std (ddof=1)
    demand_shifted = demand.shift(1)
    out['rolling_mean_7'] = demand_shifted.rolling(window=7, min_periods=3).mean()
    out['rolling_mean_14'] = demand_shifted.rolling(window=14, min_periods=7).mean()
    out['rolling_mean_30'] = demand_shifted.rolling(window=30, min_periods=14).mean()

    out['rolling_std_7'] = demand_shifted.rolling(window=7, min_periods=3).std(ddof=1).fillna(0.0)
    out['rolling_std_14'] = demand_shifted.rolling(window=14, min_periods=7).std(ddof=1).fillna(0.0)
    out['rolling_std_30'] = demand_shifted.rolling(window=30, min_periods=14).std(ddof=1).fillna(0.0)

    out['rolling_trend'] = (out['rolling_mean_7'] + 0.1) / (out['rolling_mean_30'] + 0.1)
    out['cv_7'] = out['rolling_std_7'] / (out['rolling_mean_7'] + 1.0)

    # 4. Product-level strictly historical cumulative and sparsity features (<= t - 1)
    out['hist_expanding_mean'] = demand_shifted.expanding(min_periods=1).mean()
    # Store causal prior observation count (0, 1, 2, ...) as metadata for exact state continuation
    out['hist_obs_count'] = np.arange(len(out), dtype=float)
    out['zero_demand_ratio_14'] = (
        (demand_shifted == 0.0).astype(float).rolling(window=14, min_periods=1).mean()
    )

    # Days since last positive historical demand strictly < t
    demand_vals = demand.to_numpy(dtype=float)
    days_since = np.full(len(demand_vals), np.nan, dtype=float)
    last_sale_idx: Optional[int] = None
    for i in range(len(demand_vals)):
        if last_sale_idx is not None:
            days_since[i] = float(i - last_sale_idx)
        else:
            days_since[i] = float(i + 1)
        if demand_vals[i] > 0.0:
            last_sale_idx = i
    out['days_since_last_sale'] = days_since

    # 5. Strictly historical price features (using only prices observed strictly before day t)
    # Never use backfill (bfill) or full-dataset averages!
    obs_price = out['ObservedUnitPrice'].astype(float)
    past_price = obs_price.shift(1)
    out['AvgUnitPrice'] = past_price.ffill()
    out['hist_avg_price'] = past_price.expanding(min_periods=1).mean().ffill()
    out['hist_price_count'] = past_price.expanding().count().astype(float)

    return out


def extract_product_forecast_state(
    p_history: pd.DataFrame,
    future_price_override: Optional[float] = None
) -> Dict[str, object]:
    """
    Extracts exact causal history state at the end of `p_history` (sorted ascending by Date)
    to roll forward into day `last_date + 1` with zero training-inference skew.
    """
    if p_history.empty or len(p_history) < 30:
        raise ValueError(
            f"At least 30 rows of product history are required for forecasting, got {len(p_history)}."
        )

    p_sorted = p_history.sort_values('Date')
    last_row = p_sorted.iloc[-1]
    last_date = pd.Timestamp(last_row['Date'])
    demand_series = [float(x) for x in p_sorted['DailyDemand'].tolist()]

    # Exact causal cumulative demand statistics incorporating `last_row['DailyDemand']`
    if 'hist_expanding_mean' in p_sorted.columns and 'hist_obs_count' in p_sorted.columns:
        prev_cnt = int(round(float(last_row['hist_obs_count'])))
        prev_sum = float(last_row['hist_expanding_mean']) * prev_cnt
        cum_sum = prev_sum + float(last_row['DailyDemand'])
        cum_count = prev_cnt + 1
    else:
        cum_sum = float(sum(demand_series))
        cum_count = len(demand_series)

    # Exact days_since_last_sale for step 1 (`last_date + 1`)
    if float(last_row['DailyDemand']) > 0.0:
        next_days_since_sale = 1.0
    elif 'days_since_last_sale' in p_sorted.columns and pd.notna(last_row['days_since_last_sale']):
        next_days_since_sale = float(last_row['days_since_last_sale']) + 1.0
    else:
        next_days_since_sale = float(len(demand_series) + 1)
        for offset in range(1, len(demand_series) + 1):
            if demand_series[-offset] > 0.0:
                next_days_since_sale = float(offset)
                break

    # Causal latest & expanding average price incorporating `last_date`'s observed price (if any)
    obs_last = last_row.get('ObservedUnitPrice')
    last_unit_price = float(last_row['AvgUnitPrice']) if 'AvgUnitPrice' in p_sorted.columns and pd.notna(last_row['AvgUnitPrice']) else 1.0
    hist_avg_price = float(last_row['hist_avg_price']) if 'hist_avg_price' in p_sorted.columns and pd.notna(last_row['hist_avg_price']) else last_unit_price

    if future_price_override is not None:
        last_unit_price = float(future_price_override)
    elif obs_last is not None and pd.notna(obs_last):
        obs_val = float(obs_last)
        if 'hist_price_count' in p_sorted.columns and pd.notna(last_row['hist_price_count']):
            p_cnt = int(round(float(last_row['hist_price_count'])))
            hist_avg_price = (hist_avg_price * p_cnt + obs_val) / (p_cnt + 1)
        last_unit_price = obs_val

    return {
        'stock_code': str(last_row['StockCode']),
        'last_date': last_date,
        'demand_series': demand_series,
        'cum_sum': cum_sum,
        'cum_count': cum_count,
        'next_days_since_sale': next_days_since_sale,
        'last_unit_price': last_unit_price,
        'hist_avg_price': hist_avg_price,
    }


def _sample_std(values: Sequence[float]) -> float:
    """Computes sample standard deviation (ddof=1) matching pandas rolling().std(ddof=1)."""
    n = len(values)
    if n < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def build_step_feature_row(
    stock_code: str,
    next_date: Union[date, datetime, pd.Timestamp],
    demand_history: Sequence[float],
    last_unit_price: float,
    hist_avg_price: float,
    prior_expanding_sum: Optional[float] = None,
    prior_expanding_count: Optional[int] = None,
    prior_days_since_last_sale: Optional[float] = None,
) -> Dict[str, object]:
    """
    Constructs a single feature dictionary for `next_date` strictly from historical/predicted
    demand values (`demand_history` up to t-1) and historical price values.

    Matches `compute_product_features_vectorized` identically down to floating-point ddof=1 parity.
    When continuing from a warm-up-trimmed frame (like featured_demand.csv.gz), supply
    `prior_expanding_sum`, `prior_expanding_count`, and `prior_days_since_last_sale` to avoid
    truncation bias.
    """
    ts = pd.Timestamp(next_date)
    d_obj = _to_date(ts)
    n = len(demand_history)
    if n < 30:
        raise ValueError(
            f"At least 30 days of historical demand are required for feature construction, got {n}."
        )

    lag_1 = float(demand_history[-1])
    lag_7 = float(demand_history[-7])
    lag_14 = float(demand_history[-14])
    lag_30 = float(demand_history[-30])

    w7 = demand_history[-7:]
    w14 = demand_history[-14:]
    w30 = demand_history[-30:]

    rolling_mean_7 = float(np.mean(w7))
    rolling_mean_14 = float(np.mean(w14))
    rolling_mean_30 = float(np.mean(w30))

    rolling_std_7 = _sample_std(w7)
    rolling_std_14 = _sample_std(w14)
    rolling_std_30 = _sample_std(w30)

    rolling_trend = (rolling_mean_7 + 0.1) / (rolling_mean_30 + 0.1)
    cv_7 = rolling_std_7 / (rolling_mean_7 + 1.0)

    if prior_expanding_sum is not None and prior_expanding_count is not None and prior_expanding_count > 0:
        hist_expanding_mean = float(prior_expanding_sum / prior_expanding_count)
    else:
        hist_expanding_mean = float(np.mean(demand_history))

    zero_demand_ratio_14 = float(np.mean([1.0 if float(v) == 0.0 else 0.0 for v in w14]))

    if prior_days_since_last_sale is not None:
        days_since = float(prior_days_since_last_sale)
    else:
        days_since = float(n + 1)
        for offset in range(1, n + 1):
            if float(demand_history[-offset]) > 0.0:
                days_since = float(offset)
                break

    is_month_end = 1 if (d_obj + timedelta(days=1)).month != d_obj.month else 0
    is_uk_holiday = 1 if d_obj in UK_BANK_HOLIDAYS else 0

    return {
        'StockCode': stock_code,
        'Date': ts,
        'day': int(ts.day),
        'month': int(ts.month),
        'year': int(ts.year),
        'dayofweek': int(ts.dayofweek),
        'is_weekend': 1 if ts.dayofweek in (5, 6) else 0,
        'weekofyear': int(ts.isocalendar()[1]),
        'is_month_end': is_month_end,
        'is_uk_holiday': is_uk_holiday,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'lag_14': lag_14,
        'lag_30': lag_30,
        'rolling_mean_7': rolling_mean_7,
        'rolling_mean_14': rolling_mean_14,
        'rolling_mean_30': rolling_mean_30,
        'rolling_std_7': rolling_std_7,
        'rolling_std_14': rolling_std_14,
        'rolling_std_30': rolling_std_30,
        'rolling_trend': rolling_trend,
        'cv_7': cv_7,
        'hist_expanding_mean': hist_expanding_mean,
        'zero_demand_ratio_14': zero_demand_ratio_14,
        'days_since_last_sale': days_since,
        'AvgUnitPrice': float(last_unit_price),
        'hist_avg_price': float(hist_avg_price),
    }


def build_feature_row_from_state(
    state: Dict[str, object],
    next_date: Union[date, datetime, pd.Timestamp],
) -> Dict[str, object]:
    """Convenience wrapper generating the next step feature row from a forecast state dict."""
    return build_step_feature_row(
        stock_code=str(state['stock_code']),
        next_date=next_date,
        demand_history=state['demand_series'],  # type: ignore[arg-type]
        last_unit_price=float(state['last_unit_price']),  # type: ignore[arg-type]
        hist_avg_price=float(state['hist_avg_price']),  # type: ignore[arg-type]
        prior_expanding_sum=float(state['cum_sum']),  # type: ignore[arg-type]
        prior_expanding_count=int(state['cum_count']),  # type: ignore[arg-type]
        prior_days_since_last_sale=float(state['next_days_since_sale']),  # type: ignore[arg-type]
    )


def advance_product_forecast_state(state: Dict[str, object], predicted_demand: float) -> None:
    """Appends a newly predicted demand value to the product state in place for recursive rollout."""
    val = max(0.0, float(predicted_demand))
    state['demand_series'].append(val)  # type: ignore[attr-defined]
    state['cum_sum'] = float(state['cum_sum']) + val  # type: ignore[arg-type]
    state['cum_count'] = int(state['cum_count']) + 1  # type: ignore[arg-type]
    if val > 0.0:
        state['next_days_since_sale'] = 1.0
    else:
        state['next_days_since_sale'] = float(state['next_days_since_sale']) + 1.0  # type: ignore[arg-type]

