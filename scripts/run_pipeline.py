"""Run the full data pipeline: load, clean, transform and save."""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.data.cleaner import DataCleaner
from dataframe_analyzer.data.transform import DataTransform, DatePart

FILE_PATH = "data/raw/train.csv"
OUTPUT_PATH = "data/processed/clean_dataframe.csv"

if __name__ == "__main__":
    logger.info("Pipeline started.")

    loader = DataLoader(FILE_PATH)
    df_raw = loader.load()

    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    logger.info("Extracting date parts (day, month, year)...")
    df_clean = DataTransform.transform_to_date(df_clean, 'date')
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.DAY)
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.MONTH)
    df_clean = DataTransform.add_date_part_column(df_clean, 'date', DatePart.YEAR)

    logger.info("Extracting weekday and weekend features...")
    df_clean = DataTransform.add_weekday_column(df_clean, 'date')
    df_clean = DataTransform.add_weekend_flag(df_clean, 'date')
    # NOTE: is_holiday is not created here on purpose. The source dataset
    # does not say in which country the 10 stores are, so a holiday
    # calendar could be wrong. Call
    # DataTransform.add_holiday_flag(df, 'date', country=...) only when
    # the real location of the stores is known.

    df_clean = df_clean.drop(columns=['date'])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Pipeline finished. Output saved to: {OUTPUT_PATH}")
