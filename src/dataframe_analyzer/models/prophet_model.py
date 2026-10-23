"""Prophet model for demand forecasting.

The input is a series for one store and one item. This module splits
the data by date, trains the model, predicts and calculates the metrics.
"""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

# Minimum number of training days needed to use yearly seasonality.
YEARLY_SEASONALITY_MIN_DAYS = 365


class ProphetForecaster:
    """Train a Prophet model on the time series of one store and one item."""

    def __init__(self, seasonality_mode: str = "multiplicative") -> None:
        """Store the seasonality mode for training.

        Args:
            seasonality_mode (str, optional): Prophet seasonality mode
                ("additive" or "multiplicative"). Defaults to
                "multiplicative".
        """
        self.seasonality_mode = seasonality_mode
        self.model = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Get the series of one store and item with the Prophet column names.

        Args:
            df (pd.DataFrame): Full dataset with the columns `date`,
                `store`, `item` and `sales`.
            store (int): Store id.
            item (int): Item id.

        Returns:
            pd.DataFrame: Series with the columns `ds` and `y`, in date
            order.
        """
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]]
        serie = serie.rename(columns={"date": "ds", "sales": "y"})
        serie["ds"] = pd.to_datetime(serie["ds"])
        return serie.sort_values("ds").reset_index(drop=True)

    def split_train_test(self, serie: pd.DataFrame, test_days: int = 90):
        """Split the series by date: the last `test_days` days go to the test.

        Args:
            serie (pd.DataFrame): Series with the columns `ds` and `y`.
            test_days (int): Number of last days kept for the test.

        Returns:
            tuple: `(train_df, test_df, cutoff_date)`.
        """
        cutoff = serie["ds"].max() - pd.Timedelta(days=test_days)
        train = serie[serie["ds"] <= cutoff]
        test = serie[serie["ds"] > cutoff]
        logger.info(f"Chronological cutoff: {cutoff.date()} | train={len(train)} rows, test={len(test)} rows")
        return train, test, cutoff

    def fit(self, train: pd.DataFrame) -> Prophet:
        """Train the Prophet model with the training data.

        Args:
            train (pd.DataFrame): Training series with the columns `ds`
                and `y`.

        Returns:
            Prophet: The trained model (also saved in `self.model`).
        """
        # Use yearly seasonality only when the training data covers at
        # least one full year. All training data of the project dataset
        # (5 years) covers more than one year, so the results are the same
        # as with yearly seasonality always on. For shorter user files the
        # rule is important (tested on Kaggle store 1, items 5, 15 and 28,
        # with backtests of 90, 60 and 48 days). With 640 training days,
        # yearly seasonality cut the MAE by half (for example 17.67 ->
        # 8.34). With 240 days, it doubled the MAE (11.58 -> 20.46),
        # because the model learned a cycle that the data did not show.
        # The "auto" rule of Prophet (>= 730 days) was too strict for the
        # first case.
        span_days = (pd.to_datetime(train["ds"]).max() - pd.to_datetime(train["ds"]).min()).days
        self.model = Prophet(
            yearly_seasonality=span_days >= YEARLY_SEASONALITY_MIN_DAYS,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode,
        )
        self.model.fit(train)
        return self.model

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Predict sales for the dates in the test data.

        Args:
            test (pd.DataFrame): Test data with the column `ds`.

        Returns:
            pd.DataFrame: Predictions with the columns `ds` and `yhat`.
        """
        future = test[["ds"]].copy()
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat"]]

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate MAE, RMSE and MAPE.

        In MAPE, days with 0 real sales are divided by 1 instead of 0.

        Args:
            y_true (np.ndarray): Real sales values.
            y_pred (np.ndarray): Predicted sales values.

        Returns:
            dict: Keys `MAE`, `RMSE` and `MAPE` (MAPE in percent).
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}