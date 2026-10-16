"""
Adapters that make RandomForestForecaster, XGBoostForecaster and
LSTMForecaster expose the same interface ForecastService already uses
for ProphetForecaster:

    prepare_series(df, store, item) -> DataFrame[ds, y]
    split_train_test(series, test_days) -> (train, test, cutoff)   # ds/y slices
    fit(train)                                                      # train is a ds/y DataFrame
    predict(dates_df)  -> DataFrame[ds, yhat]                       # dates_df has a 'ds' column
    compute_metrics(y_true, y_pred) -> dict

Why an adapter instead of changing the model classes themselves:
Prophet is curve-based, so it can predict ANY future date directly.
Random Forest, XGBoost and LSTM instead depend on lag/window features
computed from real past sales, which do not exist beyond the training
data. Both calls ForecastService makes to `predict()` -- the backtest
window and the future horizon -- ask for dates that are contiguous and
immediately follow whatever the model was just fit on. That means both
cases reduce to the same operation: forecast `len(dates_df)` days
ahead, recursively, using the model's own previous predictions as the
lag inputs once real history runs out. Each adapter below implements
exactly that, reusing the already-reviewed model classes without
modifying them.

Random Forest and XGBoost share the exact same feature engineering
(calendar features + lag_1/7/14/30 + rolling_mean_7, via build_features()
on their respective model classes), so the recursive-forecast logic is
factored into `_TreeModelAdapter` and both adapters just plug in their
own inner model class.
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
    """Plain chronological split on a ds/y series (same rule Prophet uses).

    Args:
        series (pd.DataFrame): Columns ds, y.
        test_days (int): Number of trailing days reserved for testing.

    Returns:
        tuple: (train, test, cutoff)
    """
    cutoff = series["ds"].max() - pd.Timedelta(days=test_days)
    train = series[series["ds"] <= cutoff].reset_index(drop=True)
    test = series[series["ds"] > cutoff].reset_index(drop=True)
    return train, test, cutoff


class _TreeModelAdapter:
    """Shared recursive-forecast logic for lag/calendar-feature tree models.

    Subclasses only need to set `_inner_cls` to the model class to wrap
    (RandomForestForecaster or XGBoostForecaster) -- both expose the
    same `prepare_series`, `build_features`, `fit`, `model` and
    `feature_columns` surface, so every other method here works for
    either one unchanged.
    """

    _inner_cls = None  # set by subclass

    def __init__(self) -> None:
        self._inner = self._inner_cls()
        self._history = None  # ds/y DataFrame the model was last fit on

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        serie = self._inner.prepare_series(df, store=store, item=item)
        return serie.rename(columns={"date": "ds", "sales": "y"})

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit on a ds/y slice (either the backtest train split or the full series).

        Args:
            train (pd.DataFrame): Columns ds, y, sorted chronologically.
        """
        self._history = train.rename(columns={"ds": "date", "y": "sales"})
        features = self._inner.build_features(self._history)
        self._inner.fit(features)

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Recursive multi-step forecast for `len(dates)` days after fit().

        Builds one row of features at a time (calendar features from the
        requested date, lag/rolling features from real history plus the
        predictions made so far) and feeds it to the trained model, one
        day at a time -- the same recursive idea used by LSTMForecaster
        .predict_recursive(), applied here because neither tree model
        ships one.

        Args:
            dates (pd.DataFrame): Column ds with the dates to forecast.

        Returns:
            pd.DataFrame: Columns ds, yhat.
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
            history_sales.append(yhat)  # feed the prediction back in as the next lag

        return pd.DataFrame({"ds": dates["ds"].values, "yhat": preds})

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return self._inner_cls.compute_metrics(y_true, y_pred)


class RandomForestAdapter(_TreeModelAdapter):
    """Adapts RandomForestForecaster to ForecastService's uniform interface."""

    _inner_cls = RandomForestForecaster


class XGBoostAdapter(_TreeModelAdapter):
    """Adapts XGBoostForecaster to ForecastService's uniform interface."""

    _inner_cls = XGBoostForecaster


class LSTMAdapter:
    """Adapts LSTMForecaster to ForecastService's uniform interface."""

    def __init__(self) -> None:
        self._inner = LSTMForecaster()
        self._history = None  # ds/y DataFrame the model was last fit on

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        serie = self._inner.prepare_series(df, store=store, item=item)
        return serie.rename(columns={"date": "ds", "sales": "y"})

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit on a ds/y slice (either the backtest train split or the full series).

        Scales using ONLY this slice (no future leakage), builds sliding
        windows and trains on all of them -- there is no internal held-out
        test set here, since ForecastService already handles the train/test
        split at a higher level.

        Args:
            train (pd.DataFrame): Columns ds, y, sorted chronologically.
        """
        self._history = train.rename(columns={"ds": "date", "y": "sales"})
        values = self._history["sales"].values.reshape(-1, 1)
        self._inner.scaler = MinMaxScaler()
        scaled = self._inner.scaler.fit_transform(values)
        X, y = self._inner.create_windows(scaled)
        self._inner.fit(X, y)

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Recursive multi-step forecast for `len(dates)` days after fit().

        Reuses LSTMForecaster.predict_recursive() as-is: builds an array
        made of the last `window` real (scaled) sales values followed by
        `horizon` placeholder values, so that method's own
        `history = scaled_values[:-test_days]` slicing lines up correctly.

        Args:
            dates (pd.DataFrame): Column ds with the dates to forecast.

        Returns:
            pd.DataFrame: Columns ds, yhat.
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
        return LSTMForecaster.compute_metrics(y_true, y_pred)
