"""LSTM model for demand forecasting.

The input is a series with the columns `date` and `sales` for one store
and one item. This module scales the data, builds windows, splits the
data by date, trains the model, predicts and calculates the metrics.

There are two ways to evaluate the model:
    - One step ahead (`predict`): each test day is predicted from the 30
      real days before it. This can be compared with the tree models.
    - Recursive, many steps (`predict_recursive`): the whole test period
      is predicted without any real sales from that period. Each
      prediction goes into the next window. This can be compared with
      Prophet.
"""

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Hide TensorFlow INFO and WARNING logs.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Avoid the oneDNN message and keep results reproducible.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense  # type: ignore

tf.get_logger().setLevel("ERROR")  # Hide the GPU support warning.

logger = logging.getLogger(__name__)

# Fixed seed, so training gives the same result in every run. This makes
# the experiments reproducible.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class LSTMForecaster:
    """Train an LSTM model on the time series of one store and one item."""

    def __init__(self, window: int = 30, epochs: int = 20, batch_size: int = 16):
        """Store the training settings. The network is built in `fit`.

        Args:
            window (int): Number of past days used to predict the next day.
            epochs (int): Number of training passes over the data.
            batch_size (int): Number of windows in each training batch.
        """
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Get the series of one store and item, in date order.

        Args:
            df (pd.DataFrame): Full dataset with the columns `date`,
                `store`, `item` and `sales`.
            store (int): Store id.
            item (int): Item id.

        Returns:
            pd.DataFrame: Series with the columns `date` and `sales`, in
            date order.
        """
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]]
        serie["date"] = pd.to_datetime(serie["date"])
        return serie.sort_values("date").reset_index(drop=True)

    def scale(self, serie: pd.DataFrame, test_days: int = 90) -> np.ndarray:
        """Scale the sales values to the range [0, 1].

        The scaler learns the minimum and maximum only from the training
        period, so no information from the test period is used. The test
        values use the same scaler, so they can be a little outside
        [0, 1]. This is not a problem.

        The LSTM needs scaled inputs. The tree models (XGBoost and Random
        Forest) do not.

        Args:
            serie (pd.DataFrame): Series with the column `sales`.
            test_days (int): Number of last days kept for the test. The
                scaler does not use them.

        Returns:
            np.ndarray: Scaled values of the whole series, shape (n, 1).
        """
        values = serie["sales"].values.reshape(-1, 1)
        train_values = values[:-test_days] if test_days > 0 else values
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_values)
        return self.scaler.transform(values)

    def create_windows(self, scaled_values: np.ndarray):
        """Build sliding windows: `window` past days predict the next day.

        Args:
            scaled_values (np.ndarray): Scaled values of the series.

        Returns:
            tuple: Arrays `(X, y)` for the LSTM. `X` has the shape
            (samples, timesteps, features).
        """
        X, y = [], []
        for i in range(self.window, len(scaled_values)):
            X.append(scaled_values[i - self.window:i, 0])
            y.append(scaled_values[i, 0])
        X = np.array(X).reshape(-1, self.window, 1)
        y = np.array(y)
        return X, y

    def split_train_test(self, X: np.ndarray, y: np.ndarray, test_days: int = 90):
        """Split the windows by date: the last `test_days` go to the test.

        Args:
            X (np.ndarray): Input windows.
            y (np.ndarray): Target values.
            test_days (int): Number of last windows kept for the test.

        Returns:
            tuple: `(X_train, X_test, y_train, y_test)`.
        """
        X_train, X_test = X[:-test_days], X[-test_days:]
        y_train, y_test = y[:-test_days], y[-test_days:]
        logger.info(f"train={len(X_train)} windows, test={len(X_test)} windows")
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build and train the LSTM network.

        The seed is set again before the network is built. In this way
        each series gives the same result, in any order of processing.

        Args:
            X_train (np.ndarray): Training input windows.
            y_train (np.ndarray): Training targets.

        Returns:
            tf.keras.Model: The trained network (also saved in
            `self.model`).
        """
        tf.keras.utils.set_random_seed(SEED)
        self.model = Sequential([
            Input(shape=(self.window, 1)),
            LSTM(50, activation="tanh"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict one step ahead and return values in real sales units.

        Each test day is predicted from the `window` real days before it
        (these can include real days of the test period).

        Args:
            X_test (np.ndarray): Test input windows.

        Returns:
            np.ndarray: Predicted sales (real scale).
        """
        pred_scaled = self.model.predict(X_test, verbose=0)
        return self.scaler.inverse_transform(pred_scaled).flatten()

    def predict_recursive(self, scaled_values: np.ndarray, test_days: int = 90) -> np.ndarray:
        """Predict the whole test period in a recursive way.

        The method starts from the last `window` days before the test
        period and predicts `test_days` days ahead. Each prediction is
        added to the window and used to predict the next day, so the
        model never sees real sales of the test period. Prophet is
        evaluated in the same way (train until the cutoff, then predict
        the whole period).

        Args:
            scaled_values (np.ndarray): Scaled values of the whole series,
                shape (n, 1).
            test_days (int): Number of days to predict.

        Returns:
            np.ndarray: Forecast for the test period (real scale), shape
            (test_days,).
        """
        history = scaled_values[:-test_days, 0]
        window_values = list(history[-self.window:])
        preds_scaled = []
        for _ in range(test_days):
            x = np.array(window_values[-self.window:]).reshape(1, self.window, 1)
            next_scaled = float(self.model(x, training=False).numpy()[0, 0])
            preds_scaled.append(next_scaled)
            window_values.append(next_scaled)
        preds = np.array(preds_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(preds).flatten()

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Change scaled target values back to real sales units.

        Args:
            y_scaled (np.ndarray): Scaled target values.

        Returns:
            np.ndarray: Real-scale target values.
        """
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

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