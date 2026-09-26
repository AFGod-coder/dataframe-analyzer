"""
Converts a column to dates and extracts date parts (day, month, year,
day of week, weekend flag, and holiday flag).
"""
from enum import Enum
import logging

import pandas as pd
import holidays

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
            invalid_dates = df[column_date_name].isna().sum()
            logger.info(f"Column '{column_date_name}' converted to date ({invalid_dates} invalid value(s)).")
        else:
            logger.warning(f"Column '{column_date_name}' not found. Skipping date conversion.")
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
            logger.info("Added column 'day_of_month' from '%s'.", column_date_name)
        elif part == DatePart.MONTH:
            df['month'] = df[column_date_name].dt.month
            logger.info("Added column 'month' from '%s'.", column_date_name)
        elif part == DatePart.YEAR:
            df['year'] = df[column_date_name].dt.year
            logger.info("Added column 'year' from '%s'.", column_date_name)

        return df

    @staticmethod
    def add_weekday_column(df: pd.DataFrame, column_date_name: str) -> pd.DataFrame:
        """Add the day of the week as both a number and a name.

        Adds two columns:
            - 'day_of_week' (int): 0 = Monday ... 6 = Sunday (pandas convention).
            - 'day_of_week_name' (str): the English weekday name (e.g. "Monday").

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the datetime column.

        Returns:
            pd.DataFrame: The DataFrame with the two new columns added.
        """
        if column_date_name in df.columns:
            df['day_of_week'] = df[column_date_name].dt.weekday
            df['day_of_week_name'] = df[column_date_name].dt.day_name()
            logger.info("Added columns 'day_of_week' and 'day_of_week_name' from '%s'.", column_date_name)
        else:
            logger.warning("Column '%s' not found. Skipping weekday extraction.", column_date_name)
        return df

    @staticmethod
    def add_weekend_flag(df: pd.DataFrame, column_date_name: str) -> pd.DataFrame:
        """Add a boolean column marking Saturdays and Sundays.

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the datetime column.

        Returns:
            pd.DataFrame: The DataFrame with the new 'is_weekend' column.
        """
        if column_date_name in df.columns:
            df['is_weekend'] = df[column_date_name].dt.weekday >= 5
            logger.info("Added column 'is_weekend' from '%s'.", column_date_name)
        else:
            logger.warning("Column '%s' not found. Skipping weekend flag.", column_date_name)
        return df

    @staticmethod
    def add_holiday_flag(df: pd.DataFrame, column_date_name: str, country: str) -> pd.DataFrame:
        """Add a boolean column marking public holidays for a given country.

        Not called by default in run_pipeline.py: the source dataset
        (Kaggle's "Store Item Demand Forecasting Challenge") does not
        state which country or region its 10 stores belong to, and its
        official description/discussion pages don't mention one either.
        Picking a calendar (e.g. US or Colombia) without that information
        would be an unverifiable assumption presented as fact, so this
        method is kept available but opt-in: call it explicitly, with the
        correct country code, only if/when the actual store location is
        confirmed (e.g. by the professor).

        Args:
            df (pd.DataFrame): The DataFrame to modify.
            column_date_name (str): Name of the datetime column.
            country (str): ISO country code accepted by the `holidays`
                package (e.g. "CO" for Colombia, "US" for the United States).
                Required: there is no default, precisely to avoid guessing.

        Returns:
            pd.DataFrame: The DataFrame with the new 'is_holiday' column.
        """
        if column_date_name in df.columns:
            years = df[column_date_name].dt.year.dropna().unique().tolist()
            country_holidays = holidays.country_holidays(country, years=years)
            df['is_holiday'] = df[column_date_name].dt.date.isin(country_holidays)
            logger.info(
                "Added column 'is_holiday' from '%s' using '%s' calendar (%d holiday dates).",
                column_date_name, country, len(country_holidays)
            )
        else:
            logger.warning("Column '%s' not found. Skipping holiday flag.", column_date_name)
        return df
