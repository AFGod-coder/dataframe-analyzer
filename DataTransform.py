"""
DataTransform class.
Provides static methods to convert and extract date parts from a DataFrame.
"""
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatePart(Enum):
    """
    Enumeration for date parts.
    """
    DAY = 'day'
    MONTH = 'month'
    YEAR = 'year'


class DataTransform:
    """
    Contains only static methods to transform date columns.
    No internal state is stored.
    """
    @staticmethod
    def transform_to_date(df: pd.DataFrame, column_date_name: str, format: str = None) -> pd.DataFrame:
        """
        Converts a column to datetime format.

        Parameters:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the column to convert.
            format (str, optional): Specific date format. Default is None.

        Returns:
            pd.DataFrame: The DataFrame with the converted column.
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
        """
        Adds a new column with the specified date part (day, month, or year).

        Parameters:
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