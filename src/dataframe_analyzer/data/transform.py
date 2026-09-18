"""
Converts a column to dates and extracts date parts (day, month, year).
"""
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatePart(Enum):
    """The part of a date that can be extracted into its own column."""
    DAY = 'day'
    MONTH = 'month'
    YEAR = 'year'


class DataTransform:
    """Static helpers to work with date columns.

    All methods are static. This class does not store any state.
    """
    @staticmethod
    def transform_to_date(df: pd.DataFrame, column_date_name: str, format: str = None) -> pd.DataFrame:
        """Convert a column to pandas datetime format.

        Values that cannot be parsed become missing (NaT) instead of
        raising an error.

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the column to convert.
            format (str, optional): Expected date format. If None,
                pandas guesses the format. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with the column converted to dates.
        """
        if column_date_name in df.columns:
            df[column_date_name] = pd.to_datetime(
                df[column_date_name],
                format=format,
                errors='coerce'
            )
            logger.info(f"Column {column_date_name} converted to date")
        return df

    @staticmethod
    def add_date_part_column(df: pd.DataFrame, column_date_name: str, part: DatePart) -> pd.DataFrame:
        """Add a new column with one part of a date column.

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the datetime column.
            part (DatePart): Which part to extract (DAY, MONTH, or YEAR).

        Returns:
            pd.DataFrame: The DataFrame with the new column added.
        """
        if part == DatePart.DAY:
            df['day_of_month'] = df[column_date_name].dt.day
        elif part == DatePart.MONTH:
            df['month'] = df[column_date_name].dt.month
        elif part == DatePart.YEAR:
            df['year'] = df[column_date_name].dt.year
        logger.info(f"")
        
        return df
