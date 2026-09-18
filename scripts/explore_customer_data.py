"""
Quick exploration script for the customer shopping dataset.
"""

import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

from dataframe_analyzer.data.transform import DataTransform

FILE_PATH = 'data/raw/customer_shopping_data.csv'

if __name__ == "__main__":
    logger.info(f"Loading file: {FILE_PATH}")
    df_raw = pd.read_csv(FILE_PATH)
    logger.info(f"Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")

    df_raw.info()
    print(df_raw.head())

    df_raw = DataTransform.transform_to_date(df_raw, 'invoice_date')
    df_raw.info()

    print(df_raw.describe())
    logger.info("Exploration finished.")
