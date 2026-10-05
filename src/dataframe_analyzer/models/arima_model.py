"""
ARIMA model for demand forecasting.

Given a series with columns [date, sales] (already filtered for a
specific store-item combination), this module handles the
chronological split, training, prediction and metric calculation.

ARIMA is curve-based in the same sense Prophet is: once fit, it can
forecast any number of steps directly ahead of the last training
point via its own internal state (no external lag features need to
be fed back in), so it exposes the exact same ds/y interface
ForecastService already uses for Prophet, with no adapter needed.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class ArimaForecaster:
    """Trains an ARIMA(p, d, q) model on a single store-item time series."""

    def __init__(self, order: tuple = (7, 1, 1)) -> None:
        """Store the (p, d, q) order used when the model is trained.

        Args:
            order (tuple, optional): (p, d, q) -- p=autoregressive terms,
                d=differencing order, q=moving-average terms. Defaults to
                (7, 1, 1): a weekly autoregressive lag (p=7) to capture the
                weekly seasonality found in the EDA (section 2.1, Figura 4),
                one difference (d=1) to remove the trend identified in the
                EDA (Figura 3), and a single moving-average term (q=1).
                Fixed rather than auto-selected per series, so every one of
                the 500 store-item combinations is evaluated under the same
                model structure -- the same reasoning behind using a single
                set of hyperparameters for Random Forest/XGBoost.
        """
        self.order = order
        self.model = None
        self._last_date = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Filter one store-item series and rename columns to ds/y format.

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

    def fit(self, train: pd.DataFrame):
        """Train the ARIMA model on the training set.

        Args:
            train (pd.DataFrame): Training series with columns ds, y.

        Returns:
            The fitted statsmodels ARIMA results object (also stored in self.model).
        """
        self._last_date = train["ds"].max()
        self.model = ARIMA(train["y"].values, order=self.order).fit()
        return self.model

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Forecast `len(test)` steps directly ahead of the last training date.

        Args:
            test (pd.DataFrame): Dates to forecast, column ds (must be
                contiguous and immediately follow the dates fit() was
                trained on -- the same assumption ForecastService already
                relies on for Prophet).

        Returns:
            pd.DataFrame: Predictions with columns ds, yhat.
        """
        horizon = len(test)
        yhat = self.model.forecast(steps=horizon)
        return pd.DataFrame({"ds": test["ds"].values, "yhat": np.asarray(yhat)})

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
