"""Adapters that give Random Forest, XGBoost and LSTM the Prophet interface.

`ForecastService` uses the same methods for every model, as in
`ProphetForecaster`:

    prepare_series(df, store, item) -> DataFrame[ds, y]
    split_train_test(series, test_days) -> (train, test, cutoff)   # ds/y slices
    fit(train)                                                      # train is a ds/y DataFrame
    predict(dates_df)  -> DataFrame[ds, yhat]                       # dates_df has a 'ds' column
    compute_metrics(y_true, y_pred) -> dict

Why we use adapters and do not change the model classes:
Prophet fits a curve, so it can predict any future date directly.
Random Forest, XGBoost and LSTM use lag or window features made from
real past sales, and these values do not exist after the training data.
`ForecastService` calls `predict()` for the backtest window and for the
future horizon. In both cases the dates are consecutive and start just
after the training data. So both cases are the same task: forecast
`len(dates_df)` days ahead in a recursive way, and use the model's own
predictions as lag values when the real history ends. Each adapter does
this and uses the model classes without changing them.

Random Forest and XGBoost use the same features (calendar features,
lag_1, lag_7, lag_14, lag_30 and rolling_mean_7, made by
`build_features()`). For this reason the recursive logic is in
`_TreeModelAdapter`, and each adapter only sets its own model class.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dataframe_analyzer.models.lstm_model import LSTMForecaster
from dataframe_analyzer.models.random_forest_model import RandomForestForecaster
from dataframe_analyzer.models.xgboost_model import XGBoostForecaster

logger = logging.getLogger(__name__)


def _date_split(series: pd.DataFrame, test_days: int):
    """Split a series by date, with the same rule that Prophet uses.

    Args:
        series (pd.DataFrame): Columns `ds` and `y`.
        test_days (int): Number of last days kept for the test.

    Returns:
        tuple: `(train, test, cutoff)`, where `cutoff` is the last date of
        the training part.
    """
    cutoff = series["ds"].max() - pd.Timedelta(days=test_days)
    train = series[series["ds"] <= cutoff].reset_index(drop=True)
    test = series[series["ds"] > cutoff].reset_index(drop=True)
    return train, test, cutoff


class _TreeModelAdapter:
    """Recursive forecast logic for tree models with lag and calendar features.

    A subclass only sets `_inner_cls` to the model class to use
    (`RandomForestForecaster` or `XGBoostForecaster`). Both classes have
    the same `prepare_series`, `build_features`, `fit`, `model` and
    `feature_columns`, so the other methods work for both.
    """

    _inner_cls = None  # The subclass sets the model class.

    def __init__(self) -> None:
        """Create the inner model. No data is stored yet."""
        self._inner = self._inner_cls()
        self._history = None  # Last ds/y data used to fit the model.

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Get the daily sales series of one store and item.

        Args:
            df (pd.DataFrame): Clean dataset with the columns `date`,
                `store`, `item` and `sales`.
            store (int): Store id.
            item (int): Item id.

        Returns:
            pd.DataFrame: Series with the columns `ds` and `y`.
        """
        serie = self._inner.prepare_series(df, store=store, item=item)
        return serie.rename(columns={"date": "ds", "sales": "y"})

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        """Split the series into a training part and a test part by date.

        Args:
            series (pd.DataFrame): Columns `ds` and `y`.
            test_days (int): Number of last days kept for the test.

        Returns:
            tuple: `(train, test, cutoff)`.
        """
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Train the model with a ds/y series.

        The series can be the training part of the backtest or the full
        series.

        Args:
            train (pd.DataFrame): Columns `ds` and `y`, in date order.
        """
        self._history = train.rename(columns={"ds": "date", "y": "sales"})
        features = self._inner.build_features(self._history)
        self._inner.fit(features)

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Forecast `len(dates)` days after the training data, one day at a time.

        For each day, the method builds one row of features: calendar
        features from the date, and lag and rolling features from the
        real history plus the predictions made before. The trained model
        predicts that day, and the prediction is used as a lag for the
        next day. This is the same idea as
        `LSTMForecaster.predict_recursive()`.

        Args:
            dates (pd.DataFrame): Column `ds` with the dates to forecast.

        Returns:
            pd.DataFrame: Columns `ds` and `yhat`.
        """
        max_lag = 30
        history_sales = list(self._history["sales"].tail(max_lag + 7).values)

        preds = []
        for target_date in pd.to_datetime(dates["ds"]):
            ts = pd.Timestamp(target_date)
            row = {
                "day_of_week": ts.dayofweek,
                "month": ts.month,
                "day": ts.day,
                "is_weekend": int(ts.dayofweek >= 5),
                "lag_1": history_sales[-1],
                "lag_7": history_sales[-7],
                "lag_14": history_sales[-14],
                "lag_30": history_sales[-30],
                "rolling_mean_7": np.mean(history_sales[-7:]),
            }
            x = pd.DataFrame([row])[self._inner.feature_columns]
            yhat = float(self._inner.model.predict(x)[0])
            preds.append(yhat)
            history_sales.append(yhat)  # The prediction becomes the next lag.

        return pd.DataFrame({"ds": dates["ds"].values, "yhat": preds})

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate the error metrics of a forecast.

        Args:
            y_true (np.ndarray): Real values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: Keys `MAE`, `RMSE` and `MAPE`.
        """
        return self._inner_cls.compute_metrics(y_true, y_pred)


class RandomForestAdapter(_TreeModelAdapter):
    """Give `RandomForestForecaster` the common interface of the service."""

    _inner_cls = RandomForestForecaster


class XGBoostAdapter(_TreeModelAdapter):
    """Give `XGBoostForecaster` the common interface of the service."""

    _inner_cls = XGBoostForecaster


class LSTMAdapter:
    """Give `LSTMForecaster` the common interface of the service."""

    def __init__(self) -> None:
        """Create the inner LSTM model. No data is stored yet."""
        self._inner = LSTMForecaster()
        self._history = None  # Last ds/y data used to fit the model.

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Get the daily sales series of one store and item.

        Args:
            df (pd.DataFrame): Clean dataset with the columns `date`,
                `store`, `item` and `sales`.
            store (int): Store id.
            item (int): Item id.

        Returns:
            pd.DataFrame: Series with the columns `ds` and `y`.
        """
        serie = self._inner.prepare_series(df, store=store, item=item)
        return serie.rename(columns={"date": "ds", "sales": "y"})

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        """Split the series into a training part and a test part by date.

        Args:
            series (pd.DataFrame): Columns `ds` and `y`.
            test_days (int): Number of last days kept for the test.

        Returns:
            tuple: `(train, test, cutoff)`.
        """
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Train the LSTM with a ds/y series.

        The scaler uses only this series, so no future data is used. The
        method builds sliding windows and trains with all of them. There
        is no test part here, because `ForecastService` already splits
        the data into training and test parts.

        Args:
            train (pd.DataFrame): Columns `ds` and `y`, in date order.
        """
        self._history = train.rename(columns={"ds": "date", "y": "sales"})
        values = self._history["sales"].values.reshape(-1, 1)
        self._inner.scaler = MinMaxScaler()
        scaled = self._inner.scaler.fit_transform(values)
        X, y = self._inner.create_windows(scaled)
        self._inner.fit(X, y)

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Forecast `len(dates)` days after the training data, one day at a time.

        The method uses `LSTMForecaster.predict_recursive()` without
        changes. It builds an array with the last `window` real scaled
        sales values and then `horizon` empty values (zeros). In this way
        the line `history = scaled_values[:-test_days]` inside that method
        takes exactly the real values.

        Args:
            dates (pd.DataFrame): Column `ds` with the dates to forecast.

        Returns:
            pd.DataFrame: Columns `ds` and `yhat`.
        """
        horizon = len(dates)
        window = self._inner.window
        last_window_sales = self._history["sales"].tail(window).values.reshape(-1, 1)
        scaled_window = self._inner.scaler.transform(last_window_sales)
        placeholder = np.zeros((horizon, 1))
        scaled_values = np.vstack([scaled_window, placeholder])

        preds = self._inner.predict_recursive(scaled_values, test_days=horizon)
        return pd.DataFrame({"ds": dates["ds"].values, "yhat": preds})

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate the error metrics of a forecast.

        Args:
            y_true (np.ndarray): Real values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: Keys `MAE`, `RMSE` and `MAPE`.
        """
        return LSTMForecaster.compute_metrics(y_true, y_pred)
