"""
backend/preprocessing.py
Data preprocessing and feature engineering module for Retail Demand Forecasting.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

SERVICE_CODES = {
    'POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'CRUK', 
    'DOT', 'AMAZONFEE', 'S', 'B', 'GIFT'
}

def clean_retail_data(
    input_path: str,
    output_path: str = None,
    qty_upper_quantile: float = 0.9995
) -> pd.DataFrame:
    """
    Cleans raw retail transaction data from Excel or CSV:
    - Removes duplicates.
    - Handles missing descriptions and customer IDs.
    - Filters out cancelled transactions (InvoiceNo starting with 'C').
    - Removes negative/zero quantities and unit prices.
    - Removes non-product service codes.
    - Filters extreme transaction anomalies.
    - Standardizes date formats.
    """
    print(f"[PREPROCESSING] Loading data from: {input_path}")
    if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    initial_count = len(df)
    print(f"[PREPROCESSING] Initial records: {initial_count}")

    # 1. Remove duplicate records
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    print(f"[PREPROCESSING] Removed {duplicates_removed} duplicate records.")

    # 2. Convert InvoiceNo and StockCode to string
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df['StockCode'] = df['StockCode'].astype(str).str.strip().str.upper()

    # 3. Filter out cancellations (InvoiceNo beginning with 'C')
    df = df[~df['InvoiceNo'].str.startswith('C')]

    # 4. Filter positive quantities and valid unit prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # 5. Filter out non-product service stock codes
    df = df[~df['StockCode'].isin(SERVICE_CODES)]
    df = df[df['StockCode'].str.len() >= 3]

    # 6. Handle missing product descriptions
    # Fill missing descriptions from the most frequent description for the same StockCode
    desc_mapping = (
        df.dropna(subset=['Description'])
        .groupby('StockCode')['Description']
        .agg(lambda s: s.mode()[0] if not s.empty else "UNKNOWN PRODUCT")
        .to_dict()
    )
    df['Description'] = df['Description'].fillna(df['StockCode'].map(desc_mapping)).fillna("UNKNOWN PRODUCT")
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()

    # 7. Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date

    # 8. Check and filter extreme single-transaction outliers
    cap_qty = df['Quantity'].quantile(qty_upper_quantile)
    df = df[df['Quantity'] <= cap_qty]

    # 9. Calculate transaction revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    print(f"[PREPROCESSING] Cleaned records: {len(df)} across {df['StockCode'].nunique()} unique products.")

    # Save cleaned dataset if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[PREPROCESSING] Cleaned data saved to: {output_path}")

    return df


def generate_daily_demand_features(
    df: pd.DataFrame,
    min_sales_days: int = 25,
    top_n_products: int = 150
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregates transactions into daily demand per product and creates time-based and lag features.
    Guarantees zero data leakage by strictly using past observations (shift >= 1).

    Returns:
        featured_df: DataFrame with model-ready features and target.
        product_metadata: Summary dataframe of all available products.
    """
    print("[FEATURE ENGINEERING] Aggregating daily demand per product...")

    # Aggregate by StockCode and Date
    daily = (
        df.groupby(['StockCode', 'Date'])
        .agg(
            DailyDemand=('Quantity', 'sum'),
            DailyRevenue=('Revenue', 'sum'),
            AvgUnitPrice=('UnitPrice', 'mean'),
            TransactionCount=('InvoiceNo', 'nunique'),
            Description=('Description', 'first')
        )
        .reset_index()
    )
    daily['Date'] = pd.to_datetime(daily['Date'])

    # Metadata of all products
    prod_meta = (
        daily.groupby('StockCode')
        .agg(
            Description=('Description', 'first'),
            TotalDemand=('DailyDemand', 'sum'),
            TotalRevenue=('DailyRevenue', 'sum'),
            ActiveDays=('Date', 'nunique'),
            AvgDailyDemand=('DailyDemand', 'mean'),
            AvgPrice=('AvgUnitPrice', 'mean')
        )
        .reset_index()
        .sort_values(by='TotalDemand', ascending=False)
    )

    # Filter to high-activity products for robust time-series forecasting
    valid_products = prod_meta[
        (prod_meta['ActiveDays'] >= min_sales_days)
    ].head(top_n_products)['StockCode'].tolist()

    print(f"[FEATURE ENGINEERING] Selected {len(valid_products)} high-volume products for supervised modeling.")

    # Create complete date grid for each product to accurately account for zero-demand days
    min_date = daily['Date'].min()
    max_date = daily['Date'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')

    dfs = []
    for code in valid_products:
        p_daily = daily[daily['StockCode'] == code].set_index('Date')
        p_desc = prod_meta.loc[prod_meta['StockCode'] == code, 'Description'].values[0]
        avg_price = prod_meta.loc[prod_meta['StockCode'] == code, 'AvgPrice'].values[0]

        # Reindex across complete daily calendar
        reindexed = p_daily.reindex(all_dates)
        reindexed['StockCode'] = code
        reindexed['Description'] = p_desc
        reindexed['DailyDemand'] = reindexed['DailyDemand'].fillna(0.0)
        reindexed['DailyRevenue'] = reindexed['DailyRevenue'].fillna(0.0)
        reindexed['AvgUnitPrice'] = reindexed['AvgUnitPrice'].fillna(avg_price)
        reindexed['TransactionCount'] = reindexed['TransactionCount'].fillna(0.0)
        reindexed['Date'] = reindexed.index

        # Feature Creation (Strictly Past Information)
        # Calendar features
        reindexed['day'] = reindexed['Date'].dt.day
        reindexed['month'] = reindexed['Date'].dt.month
        reindexed['year'] = reindexed['Date'].dt.year
        reindexed['dayofweek'] = reindexed['Date'].dt.dayofweek
        reindexed['is_weekend'] = reindexed['dayofweek'].isin([5, 6]).astype(int)
        reindexed['weekofyear'] = reindexed['Date'].dt.isocalendar().week.astype(int)

        # Lag features (past demand)
        reindexed['lag_1'] = reindexed['DailyDemand'].shift(1)
        reindexed['lag_7'] = reindexed['DailyDemand'].shift(7)
        reindexed['lag_14'] = reindexed['DailyDemand'].shift(14)
        reindexed['lag_30'] = reindexed['DailyDemand'].shift(30)

        # Rolling statistics (using closed='left' or shift(1) to prevent target leakage)
        demand_shifted = reindexed['DailyDemand'].shift(1)
        reindexed['rolling_mean_7'] = demand_shifted.rolling(window=7, min_periods=3).mean()
        reindexed['rolling_mean_14'] = demand_shifted.rolling(window=14, min_periods=7).mean()
        reindexed['rolling_mean_30'] = demand_shifted.rolling(window=30, min_periods=14).mean()
        reindexed['rolling_std_7'] = demand_shifted.rolling(window=7, min_periods=3).std().fillna(0.0)

        # Rolling trend indicator: ratio of 7-day to 30-day rolling mean
        reindexed['rolling_trend'] = (reindexed['rolling_mean_7'] + 0.1) / (reindexed['rolling_mean_30'] + 0.1)

        # Historical variability
        reindexed['cv_7'] = reindexed['rolling_std_7'] / (reindexed['rolling_mean_7'] + 1.0)

        dfs.append(reindexed)

    featured_df = pd.concat(dfs, ignore_index=True)

    # Drop warm-up rows where 30-day lags are not yet available
    featured_df = featured_df.dropna(subset=['lag_30', 'rolling_mean_30']).reset_index(drop=True)

    print(f"[FEATURE ENGINEERING] Created dataset with {len(featured_df)} samples and {featured_df.shape[1]} columns.")
    return featured_df, prod_meta
