"""
backend/eda.py
Exploratory Data Analysis module for Retail Demand Forecasting.
Generates comprehensive visualizations and summary metrics.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_eda_visualizations(cleaned_csv_path: str, output_dir: str):
    """
    Generates all required EDA plots and saves them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[EDA] Loading cleaned data from {cleaned_csv_path}...")
    df = pd.read_csv(cleaned_csv_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    # Set overall aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("deep")

    # 1. Top 10 Best-Selling Products by Quantity
    print("[EDA] Generating Chart 1: Top 10 Best-Selling Products...")
    plt.figure(figsize=(12, 6))
    top_qty = (
        df.groupby(['StockCode', 'Description'])['Quantity']
        .sum()
        .reset_index()
        .sort_values(by='Quantity', ascending=False)
        .head(10)
    )
    # Truncate description for display
    top_qty['ShortDesc'] = top_qty['Description'].str.slice(0, 24) + " (" + top_qty['StockCode'] + ")"
    sns.barplot(data=top_qty, y='ShortDesc', x='Quantity', palette="Blues_r")
    plt.title("Top 10 Best-Selling Retail Products by Total Quantity Sold", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Total Quantity Sold (Units)", fontsize=12)
    plt.ylabel("Product Description (Stock Code)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top10_bestsellers.png"), dpi=200)
    plt.close()

    # 2. Monthly Sales Trends
    print("[EDA] Generating Chart 2: Monthly Sales Trends...")
    plt.figure(figsize=(12, 5))
    monthly_sales = df.groupby('YearMonth').agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalRevenue=('Revenue', 'sum')
    ).reset_index()
    # Sort chronologically
    monthly_sales = monthly_sales.sort_values('YearMonth')
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color = '#1f77b4'
    ax1.set_xlabel('Year-Month', fontsize=12)
    ax1.set_ylabel('Total Units Sold', color=color, fontsize=12)
    ax1.plot(monthly_sales['YearMonth'], monthly_sales['TotalQuantity'], marker='o', color=color, linewidth=2.5, label='Units Sold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Total Revenue (£)', color=color2, fontsize=12)
    ax2.plot(monthly_sales['YearMonth'], monthly_sales['TotalRevenue'], marker='s', color=color2, linewidth=2.5, linestyle='--', label='Revenue (£)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    plt.title("Monthly Retail Demand and Revenue Trends (Dec 2010 - Dec 2011)", fontsize=14, weight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_sales_trend.png"), dpi=200)
    plt.close()

    # 3. Daily Demand Trends
    print("[EDA] Generating Chart 3: Daily Demand Trends...")
    plt.figure(figsize=(14, 5))
    daily_trend = df.groupby('Date')['Quantity'].sum().reset_index()
    plt.plot(daily_trend['Date'], daily_trend['Quantity'], color='#2ca02c', alpha=0.6, label='Daily Demand')
    daily_trend['7D_MA'] = daily_trend['Quantity'].rolling(window=7, min_periods=1).mean()
    plt.plot(daily_trend['Date'], daily_trend['7D_MA'], color='#d62728', linewidth=2.2, label='7-Day Moving Average')
    plt.title("Daily Retail Demand Trajectory & 7-Day Moving Average", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Daily Quantity Sold", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "daily_demand_trend.png"), dpi=200)
    plt.close()

    # 4. Highest Revenue-Generating Products
    print("[EDA] Generating Chart 4: Highest Revenue Products...")
    plt.figure(figsize=(12, 6))
    top_rev = (
        df.groupby(['StockCode', 'Description'])['Revenue']
        .sum()
        .reset_index()
        .sort_values(by='Revenue', ascending=False)
        .head(10)
    )
    top_rev['ShortDesc'] = top_rev['Description'].str.slice(0, 24) + " (" + top_rev['StockCode'] + ")"
    sns.barplot(data=top_rev, y='ShortDesc', x='Revenue', palette="YlOrRd_r")
    plt.title("Top 10 Highest Revenue-Generating Products (£)", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Total Sales Revenue (£)", fontsize=12)
    plt.ylabel("Product Description", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_revenue_products.png"), dpi=200)
    plt.close()

    # 5. Distribution of Product Quantities
    print("[EDA] Generating Chart 5: Quantity Distribution...")
    plt.figure(figsize=(10, 5))
    # Exclude extreme tail for visual clarity
    clipped_qty = df['Quantity'][df['Quantity'] <= df['Quantity'].quantile(0.98)]
    sns.histplot(clipped_qty, bins=40, kde=True, color='#4c72b0')
    plt.title("Distribution of Transaction Quantities (Up to 98th Percentile)", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Quantity Per Transaction", fontsize=12)
    plt.ylabel("Transaction Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantity_distribution.png"), dpi=200)
    plt.close()

    # 6. Demand Patterns Over Time (Day of Week Analysis)
    print("[EDA] Generating Chart 6: Demand by Day of Week...")
    plt.figure(figsize=(10, 5))
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_demand = df.groupby('DayOfWeek')['Quantity'].sum().reindex(day_order).fillna(0).reset_index()
    sns.barplot(data=dow_demand, x='DayOfWeek', y='Quantity', palette="mako")
    plt.title("Retail Demand Distribution Across Days of the Week", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Total Quantity Sold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demand_patterns_over_time.png"), dpi=200)
    plt.close()

    # 7. High-Demand vs Low-Demand Products
    print("[EDA] Generating Chart 7: High-Demand vs Low-Demand Products...")
    plt.figure(figsize=(10, 5))
    product_totals = df.groupby('StockCode')['Quantity'].sum()
    quantiles = pd.qcut(product_totals, q=4, labels=['Low Demand (Q1)', 'Moderate (Q2)', 'High (Q3)', 'Very High (Q4)'])
    tier_summary = quantiles.value_counts().loc[['Low Demand (Q1)', 'Moderate (Q2)', 'High (Q3)', 'Very High (Q4)']]
    
    plt.bar(tier_summary.index.astype(str), tier_summary.values, color=['#9ecae1', '#6baed6', '#3182bd', '#08519c'])
    plt.title("Product Categorization into Demand Quartiles", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Demand Quartile Tier", fontsize=12)
    plt.ylabel("Number of Distinct Products", fontsize=12)
    for i, v in enumerate(tier_summary.values):
        plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "high_vs_low_demand.png"), dpi=200)
    plt.close()

    # 8. Product-Wise Demand Comparison (Top 8 Active Products)
    print("[EDA] Generating Chart 8: Product-Wise Demand Trajectories...")
    plt.figure(figsize=(14, 6))
    top8_codes = df.groupby('StockCode')['Quantity'].sum().nlargest(6).index.tolist()
    daily_top8 = (
        df[df['StockCode'].isin(top8_codes)]
        .groupby(['Date', 'StockCode'])['Quantity']
        .sum()
        .reset_index()
    )
    for code in top8_codes:
        sub = daily_top8[daily_top8['StockCode'] == code]
        sub_7d = sub.set_index('Date')['Quantity'].rolling(window=7, min_periods=1).mean()
        plt.plot(sub_7d.index, sub_7d.values, label=f"Stock {code}", linewidth=2)
    plt.title("7-Day Moving Average Demand Trajectories for Top 6 Retail SKUs", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Demand (7D MA Units)", fontsize=12)
    plt.legend(title="Stock Code", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "product_wise_demand.png"), dpi=200)
    plt.close()

    print(f"[EDA] All 8 visualizations successfully generated and saved to {output_dir}")

if __name__ == "__main__":
    import sys
    cleaned_csv = sys.argv[1] if len(sys.argv) > 1 else "dataset/cleaned_retail_data.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/graphs"
    generate_eda_visualizations(cleaned_csv, out_dir)
