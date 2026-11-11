"""
backend/preprocessing.py
Data preprocessing and feature engineering module for Retail Demand Forecasting.

Key Guarantees:
1. Removes exact duplicates, C-prefixed cancellations, non-positive quantities/prices,
   non-merchandise service codes (including GIFT_* vouchers and C2 carriage), and
   1-to-1 same-day/24h paired void reversals without truncating legitimate bulk wholesale orders.
2. Prevents test-period leakage in product selection and catalog metadata by restricting
   product selection (`top_n_products`, `min_sales_days`) strictly to `< cutoff_date`.
3. Builds complete active daily calendars from each SKU's `first_sale_date` forward.
   Note: Zero-filled `DailyDemand = 0.0` indicates zero recorded transactions, which does
   not necessarily imply zero underlying unconstrained demand (stockouts/closures are unrecorded).
4. Computes strictly causal historical features via `backend/features.py` and sorts globally
   by `['Date', 'StockCode']`.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd

from backend.features import compute_product_features_vectorized

SERVICE_CODES = {
    'POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'CRUK',
    'DOT', 'AMAZONFEE', 'S', 'B', 'C2', 'GIFT'
}


def _find_matched_reversal_indices(
    df: pd.DataFrame,
    window_hours: float = 24.0
) -> set:
    """
    Identifies 1-to-1 positive transaction indices that were immediately voided/reversed
    by a matching `C*` cancellation invoice for the same (CustomerID, StockCode, Quantity)
    within `window_hours` after the positive order (InvoiceDate_can >= InvoiceDate_pos).

    This eliminates anomalous clerical error spikes (e.g. +80,995/-80,995 voided 12 mins later
    or +74,215/-74,215 voided 16 mins later) without truncating legitimate uncancelled B2B bulk demand.
    """
    valid_cust = df['CustomerID'].notna()
    is_cancel = df['InvoiceNo'].str.startswith('C')

    cancels = df.loc[
        is_cancel & valid_cust & (df['Quantity'] < 0),
        ['CustomerID', 'StockCode', 'Quantity', 'InvoiceDate']
    ].copy()
    positives = df.loc[
        (~is_cancel) & valid_cust & (df['Quantity'] > 0) & (df['UnitPrice'] > 0),
        ['CustomerID', 'StockCode', 'Quantity', 'InvoiceDate']
    ].copy()

    if cancels.empty or positives.empty:
        return set()

    cancels['AbsQty'] = cancels['Quantity'].abs()
    cancels = cancels.sort_values('InvoiceDate')
    positives = positives.sort_values('InvoiceDate')

    # Group positives by (CustomerID, StockCode, Quantity) for fast 1-to-1 chronological matching
    pos_lookup = {}
    for idx, row in positives.iterrows():
        key = (row['CustomerID'], row['StockCode'], int(row['Quantity']) if float(row['Quantity']).is_integer() else float(row['Quantity']))
        pos_lookup.setdefault(key, []).append((idx, row['InvoiceDate']))

    voided_pos_indices = set()
    max_delta = pd.Timedelta(hours=window_hours)

    for _, c_row in cancels.iterrows():
        abs_qty = c_row['AbsQty']
        key = (c_row['CustomerID'], c_row['StockCode'], int(abs_qty) if float(abs_qty).is_integer() else float(abs_qty))
        candidates = pos_lookup.get(key)
        if not candidates:
            continue

        c_time = c_row['InvoiceDate']
        # Match most recent unmatched positive order within [c_time - window_hours, c_time]
        best_pos_list_idx = None
        for i in range(len(candidates) - 1, -1, -1):
            p_idx, p_time = candidates[i]
            if p_idx in voided_pos_indices:
                continue
            if p_time <= c_time and (c_time - p_time) <= max_delta:
                best_pos_list_idx = i
                break
        if best_pos_list_idx is not None:
            voided_pos_indices.add(candidates[best_pos_list_idx][0])

    return voided_pos_indices


def clean_retail_data(
    input_path: str,
    output_path: Optional[str] = None,
    qty_upper_quantile: Optional[float] = None,
    reversal_window_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Cleans raw retail transaction data from Excel or CSV:
    - Removes exact duplicates.
    - Standardizes InvoiceNo, StockCode, and chronological InvoiceDate.
    - Removes paired short-window (<= 24h) exact cancellation reversals (both C* and original voided order).
    - Filters out remaining C-prefixed cancellations and non-positive quantities/unit prices.
    - Removes non-product service/voucher codes (SERVICE_CODES, GIFT_*, len < 3).
    - Preserves legitimate high-volume wholesale demand (no arbitrary full-dataset quantile truncation
      unless qty_upper_quantile is explicitly passed).
    - Imputes missing descriptions causally without future text leakage.
    """
    print(f"[PREPROCESSING] Loading data from: {input_path}")
    if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    initial_count = len(df)
    print(f"[PREPROCESSING] Initial records: {initial_count}")

    # 1. Remove exact duplicate records
    df = df.drop_duplicates().copy()
    duplicates_removed = initial_count - len(df)
    print(f"[PREPROCESSING] Removed {duplicates_removed} duplicate records.")

    # 2. Standardize identifiers and timestamps
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip().str.upper()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.sort_values(['InvoiceDate', 'InvoiceNo', 'StockCode']).reset_index(drop=True)

    # 3. Identify & remove immediate 1-to-1 paired cancellation reversals (<= reversal_window_hours)
    voided_pos_indices = _find_matched_reversal_indices(df, window_hours=reversal_window_hours)
    if voided_pos_indices:
        print(f"[PREPROCESSING] Removed {len(voided_pos_indices)} immediately voided/reversed positive transactions.")
        df = df.drop(index=list(voided_pos_indices))

    # 4. Filter out all cancellation invoices and invalid quantities/prices
    df = df[~df['InvoiceNo'].str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

    # 5. Filter out non-product service/voucher stock codes
    df = df[~df['StockCode'].isin(SERVICE_CODES)]
    df = df[~df['StockCode'].str.startswith('GIFT_')]
    df = df[df['StockCode'].str.len() >= 3].copy()

    # 6. Optional explicit quantile cap (disabled by default to preserve legitimate bulk wholesale demand)
    if qty_upper_quantile is not None and 0.0 < qty_upper_quantile < 1.0:
        cap_qty = df['Quantity'].quantile(qty_upper_quantile)
        df = df[df['Quantity'] <= cap_qty].copy()

    # 7. Handle missing product descriptions causally along chronological order
    # Forward-fill within StockCode first, then fill any remaining leading missing descriptions
    # using the StockCode's earliest observed non-null description, else "UNKNOWN PRODUCT".
    raw_desc = df['Description'].astype('object')
    cleaned_desc = raw_desc.where(raw_desc.notna(), np.nan)
    cleaned_desc = cleaned_desc.apply(lambda x: str(x).strip().upper() if pd.notna(x) else np.nan)
    df['Description'] = cleaned_desc
    df['Description'] = df.groupby('StockCode')['Description'].ffill()
    df['Description'] = df.groupby('StockCode')['Description'].bfill()
    df['Description'] = df['Description'].fillna("UNKNOWN PRODUCT")

    # 8. Standardized Date and Revenue
    df['Date'] = df['InvoiceDate'].dt.date
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    print(f"[PREPROCESSING] Cleaned records: {len(df)} across {df['StockCode'].nunique()} unique products.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[PREPROCESSING] Cleaned data saved to: {output_path}")

    return df.reset_index(drop=True)


def generate_daily_demand_features(
    df: pd.DataFrame,
    min_sales_days: int = 25,
    top_n_products: int = 150,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = "2011-11-01",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates transactions into daily demand per product and creates strictly past-looking
    time-series features using `backend/features.py`.

    Zero-Leakage Guarantees:
    - Product selection (`valid_products`) and `prod_meta` are computed strictly on records
      with `Date < cutoff_date` when `cutoff_date` is specified.
    - Each product's daily calendar runs from its first historical sale date (strictly < cutoff_date)
      to `max_date`, ensuring real observed initial prices and zero pre-launch artificial zero months.
    - Zero-transaction days within an active lifecycle are filled with `DailyDemand = 0.0`
      (representing zero recorded sales; note that unrecorded stockouts/closures may cause zero sales
       even when latent customer demand exists).
    - Final feature dataframe is globally sorted by `['Date', 'StockCode']`.
    """
    print("[FEATURE ENGINEERING] Aggregating daily demand per product...")

    daily = (
        df.groupby(['StockCode', 'Date'])
        .agg(
            DailyDemand=('Quantity', 'sum'),
            DailyRevenue=('Revenue', 'sum'),
            ObservedUnitPrice=('UnitPrice', 'mean'),
            TransactionCount=('InvoiceNo', 'nunique'),
            Description=('Description', 'first')
        )
        .reset_index()
    )
    daily['Date'] = pd.to_datetime(daily['Date'])

    # Strictly restrict product selection & catalog statistics to training period (< cutoff_date)
    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        selection_daily = daily[daily['Date'] < cutoff_ts].copy()
        if selection_daily.empty:
            selection_daily = daily.copy()
    else:
        cutoff_ts = None
        selection_daily = daily.copy()

    prod_meta = (
        selection_daily.groupby('StockCode')
        .agg(
            Description=('Description', 'first'),
            TotalDemand=('DailyDemand', 'sum'),
            TotalRevenue=('DailyRevenue', 'sum'),
            ActiveDays=('Date', 'nunique'),
            AvgDailyDemand=('DailyDemand', 'mean'),
            AvgPrice=('ObservedUnitPrice', 'mean'),
            FirstSaleDate=('Date', 'min'),
        )
        .reset_index()
        .sort_values(by='TotalDemand', ascending=False)
    )

    if cutoff_ts is not None:
        min_first_date_limit = cutoff_ts - pd.Timedelta(days=31)
        eligible = prod_meta[
            (prod_meta['ActiveDays'] >= min_sales_days) &
            (prod_meta['FirstSaleDate'] <= min_first_date_limit)
        ]
    else:
        eligible = prod_meta[prod_meta['ActiveDays'] >= min_sales_days]

    valid_products = eligible.head(top_n_products)['StockCode'].tolist()

    # Preserve FirstSaleDate cleanly as string in catalog metadata for JSON serialization
    prod_meta_export = prod_meta.copy()
    prod_meta_export['FirstSaleDate'] = prod_meta_export['FirstSaleDate'].dt.strftime('%Y-%m-%d')

    print(
        f"[FEATURE ENGINEERING] Selected {len(valid_products)} active training-period products "
        f"(cutoff={cutoff_date})."
    )

    max_date = daily['Date'].max()
    dfs = []

    for code in valid_products:
        p_daily = daily[daily['StockCode'] == code].set_index('Date').sort_index()
        meta_row = prod_meta.loc[prod_meta['StockCode'] == code].iloc[0]
        p_desc = meta_row['Description']
        p_first_date = meta_row['FirstSaleDate']

        # Build complete active calendar strictly from first observed sale date to max_date
        product_dates = pd.date_range(p_first_date, max_date, freq='D')
        reindexed = p_daily.reindex(product_dates)
        reindexed['StockCode'] = code
        reindexed['Description'] = p_desc
        reindexed['DailyDemand'] = reindexed['DailyDemand'].fillna(0.0)
        reindexed['DailyRevenue'] = reindexed['DailyRevenue'].fillna(0.0)
        reindexed['TransactionCount'] = reindexed['TransactionCount'].fillna(0.0)
        reindexed['Date'] = reindexed.index

        # Compute all causal lag, variability, product-history & historical price features
        featured_prod = compute_product_features_vectorized(reindexed)
        dfs.append(featured_prod)

    featured_df = pd.concat(dfs, ignore_index=True)

    # Drop initial 30-day warm-up window where lag_30 / rolling_mean_30 / causal prices are unavailable
    featured_df = featured_df.dropna(
        subset=['lag_30', 'rolling_mean_30', 'AvgUnitPrice', 'hist_avg_price']
    ).copy()

    # Global chronological sort: Date ascending, then StockCode ascending
    featured_df = featured_df.sort_values(['Date', 'StockCode']).reset_index(drop=True)

    # Filter catalog to modeled products so API catalog matches modeled SKUs accurately
    modeled_meta = prod_meta_export[prod_meta_export['StockCode'].isin(valid_products)].reset_index(drop=True)

    print(
        f"[FEATURE ENGINEERING] Created globally chronological dataset with "
        f"{len(featured_df)} samples across {featured_df['StockCode'].nunique()} SKUs "
        f"({featured_df.shape[1]} columns)."
    )
    return featured_df, modeled_meta
