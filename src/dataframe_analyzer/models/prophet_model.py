"""
Prophet model for demand forecasting.

Given a series with columns [date, sales] (already filtered for a
specific store-item combination), this module handles the
chronological split, training, prediction and metric calculation.
"""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Trains a Prophet model on a single store-item time series."""

    def __init__(self, seasonality_mode: str = "multiplicative") -> None:
        """Store the seasonality mode used when the model is trained.

        Args:
            seasonality_mode (str, optional): Prophet's seasonality_mode
                ("additive" or "multiplicative"). Defaults to "multiplicative".
        """
        self.seasonality_mode = seasonality_mode
        self.model = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Filter one store-item series and rename columns to Prophet's ds/y format.

        Args:
            df (pd.DataFrame): Full dataset with columns date, store, item, sales.
            store (int): Store id to filter.
            item (int): Item id to filter.

        Returns:
            pd.DataFrame: Series with columns ds, y, sorted chronologically.
        """
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]]
        serie = serie.rename(columns={"date": "ds", "sales": "y"})
        serie["ds"] = pd.to_datetime(serie["ds"])
        return serie.sort_values("ds").reset_index(drop=True)

    def split_train_test(self, serie: pd.DataFrame, test_days: int = 90):
        """Chronological train/test split (last `test_days` days go to test).

        Args:
            serie (pd.DataFrame): Series with columns ds, y.
            test_days (int): Number of trailing days reserved for testing.

        Returns:
            tuple: (train_df, test_df, cutoff_date)
        """
        cutoff = serie["ds"].max() - pd.Timedelta(days=test_days)
        train = serie[serie["ds"] <= cutoff]
        test = serie[serie["ds"] > cutoff]
        logger.info(f"Chronological cutoff: {cutoff.date()} | train={len(train)} rows, test={len(test)} rows")
        return train, test, cutoff

    def fit(self, train: pd.DataFrame) -> Prophet:
        """Train the Prophet model on the training set.

        Args:
            train (pd.DataFrame): Training series with columns ds, y.

        Returns:
            Prophet: The fitted Prophet model (also stored in self.model).
        """
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
        )
        self.model.fit(train)
        return self.model

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Predict sales for the dates present in the test set.

        Args:
            test (pd.DataFrame): Test series with column ds.

        Returns:
            pd.DataFrame: Predictions with columns ds, yhat.
        """
        future = test[["ds"]].copy()
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat"]]

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute MAE, RMSE and MAPE.

        Args:
            y_true (np.ndarray): Real sales values.
            y_pred (np.ndarray): Predicted sales values.

        Returns:
            dict: {"MAE": ..., "RMSE": ..., "MAPE": ...}
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}