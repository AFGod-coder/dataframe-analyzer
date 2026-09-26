"""
Exploratory Data Analysis (EDA) for the sales dataset.

Covers section 2.1 (Dataset Construction) of the project rubric:
    1. Confirms the dataset is a balanced panel (same number of rows
       for every store-item combination).
    2. Distribution of sales by store.
    3. Distribution of sales by product.
    4. Sales trend over time (monthly) to check for seasonality.

Run after `scripts/run_pipeline.py`, since it reads the cleaned and
transformed dataset produced by the pipeline.
"""

import logging
import os

import pandas as pd

from dataframe_analyzer.visualization.eda_utils import EdaPlotter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

INPUT_PATH = "data/processed/clean_dataframe.csv"
FIGURES_DIR = "reports/figures/eda"


def check_balanced_dataset(df: pd.DataFrame) -> None:
    """Confirm every store-item combination has the same number of rows.

    A "balanced panel" here means the dataset has a full daily history
    for every store-item combination, with no missing days. This is
    verified by counting rows per (store, item) group and checking that
    only one distinct row count exists across all groups.

    Args:
        df (pd.DataFrame): The cleaned dataset, with 'store' and 'item'
            columns.
    """
    rows_per_combination = df.groupby(['store', 'item']).size()
    unique_row_counts = rows_per_combination.unique()

    n_combinations = rows_per_combination.shape[0]
    logger.info(f"Store-item combinations found: {n_combinations}")
    logger.info(f"Distinct row counts across combinations: {unique_row_counts}")

    if len(unique_row_counts) == 1:
        logger.info(
            f"Dataset IS balanced: every combination has exactly "
            f"{unique_row_counts[0]} rows."
        )
    else:
        logger.warning(
            "Dataset is NOT balanced: combinations have different row counts."
        )


def plot_sales_by_store(df: pd.DataFrame, plotter: EdaPlotter) -> None:
    """Plot total sales per store, sorted from highest to lowest.

    Args:
        df (pd.DataFrame): The cleaned dataset.
        plotter (EdaPlotter): Plotter instance used to draw the chart.
    """
    sales_by_store = (
        df.groupby('store')['sales']
        .sum()
        .reset_index()
        .sort_values('sales', ascending=False)
    )
    logger.info(f"Total sales by store:\n{sales_by_store}")

    plotter.plot_bar(
        df=sales_by_store,
        x='store',
        y='sales',
        title='Total Sales by Store',
        xlabel='Store',
        ylabel='Total Sales',
        orient='v',
        save_path=os.path.join(FIGURES_DIR, "sales_by_store.png"),
    )


def plot_sales_by_product(df: pd.DataFrame, plotter: EdaPlotter) -> None:
    """Plot total sales per product (item), sorted, using horizontal bars.

    Horizontal orientation is used because there are 50 products: with
    vertical bars the x-axis labels would overlap and become unreadable.

    Args:
        df (pd.DataFrame): The cleaned dataset.
        plotter (EdaPlotter): Plotter instance used to draw the chart.
    """
    sales_by_item = (
        df.groupby('item')['sales']
        .sum()
        .reset_index()
        .sort_values('sales', ascending=False)
    )
    logger.info(f"Top 5 products by total sales:\n{sales_by_item.head()}")

    plotter.plot_bar(
        df=sales_by_item,
        x='item',
        y='sales',
        title='Total Sales by Product',
        xlabel='Product (item)',
        ylabel='Total Sales',
        orient='h',
        figsize=(12, 14),
        save_path=os.path.join(FIGURES_DIR, "sales_by_product.png"),
    )


def plot_sales_trend(df: pd.DataFrame, plotter: EdaPlotter) -> None:
    """Plot total monthly sales over time to inspect trend and seasonality.

    Args:
        df (pd.DataFrame): The cleaned dataset, with 'year' and 'month'
            columns already extracted by the pipeline.
        plotter (EdaPlotter): Plotter instance used to draw the chart.
    """
    sales_by_month = (
        df.groupby(['year', 'month'])['sales']
        .sum()
        .reset_index()
        .sort_values(['year', 'month'])
    )
    sales_by_month['period'] = (
        sales_by_month['year'].astype(str) + '-' + sales_by_month['month'].astype(str).str.zfill(2)
    )
    logger.info(f"Monthly sales range: {sales_by_month['period'].iloc[0]} to {sales_by_month['period'].iloc[-1]}")

    plotter.plot_line(
        df=sales_by_month,
        x='period',
        y='sales',
        title='Monthly Sales Trend (All Stores and Products)',
        xlabel='Month',
        ylabel='Total Sales',
        save_path=os.path.join(FIGURES_DIR, "sales_trend_monthly.png"),
    )


def plot_sales_by_weekday(df: pd.DataFrame, plotter: EdaPlotter) -> None:
    """Plot average sales by day of week, to spot a weekly pattern.

    This complements the monthly trend chart: it shows whether sales
    behave differently on weekends versus weekdays, which is useful
    context for the 'day_of_week' / 'is_weekend' features added to the
    pipeline.

    Args:
        df (pd.DataFrame): The cleaned dataset, with 'day_of_week' and
            'day_of_week_name' columns already extracted by the pipeline.
        plotter (EdaPlotter): Plotter instance used to draw the chart.
    """
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_by_weekday = (
        df.groupby('day_of_week_name')['sales']
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )
    logger.info(f"Average sales by day of week:\n{sales_by_weekday}")

    plotter.plot_bar(
        df=sales_by_weekday,
        x='day_of_week_name',
        y='sales',
        title='Average Sales by Day of Week',
        xlabel='Day of Week',
        ylabel='Average Sales',
        orient='v',
        save_path=os.path.join(FIGURES_DIR, "sales_by_weekday.png"),
    )


if __name__ == "__main__":
    logger.info("EDA started.")

    df_clean = pd.read_csv(INPUT_PATH)
    plotter = EdaPlotter(style="whitegrid")

    logger.info("Step 1/4: checking whether the dataset is a balanced panel...")
    check_balanced_dataset(df_clean)

    logger.info("Step 2/4: distribution of sales by store...")
    plot_sales_by_store(df_clean, plotter)

    logger.info("Step 3/4: distribution of sales by product...")
    plot_sales_by_product(df_clean, plotter)

    logger.info("Step 4/4: sales trend over time (monthly) and by day of week...")
    plot_sales_trend(df_clean, plotter)
    plot_sales_by_weekday(df_clean, plotter)

    logger.info(f"EDA finished. Charts saved under: {FIGURES_DIR}")
