"""
LSTM model for demand forecasting.

Given a series with columns [date, sales] (already filtered for a
specific store-item combination), this module handles scaling,
windowing, the chronological split, training, prediction and metric
calculation.
"""

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # oculta logs INFO/WARNING de TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # evita el aviso de oneDNN y mejora reproducibilidad

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense  # type: ignore

tf.get_logger().setLevel("ERROR")  # oculta el warning de GPU support

logger = logging.getLogger(__name__)

# Semilla fija para que el entrenamiento sea reproducible (mismo resultado
# en cada corrida). Necesario para la trazabilidad exigida por la rúbrica.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class LSTMForecaster:
    """Trains an LSTM model on a single store-item time series."""

    def __init__(self, window: int = 30, epochs: int = 20, batch_size: int = 16):
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Filter one store-item series and sort it chronologically.

        Args:
            df (pd.DataFrame): Full dataset with columns date, store, item, sales.
            store (int): Store id to filter.
            item (int): Item id to filter.

        Returns:
            pd.DataFrame: Series with columns date, sales, sorted chronologically.
        """
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]]
        serie["date"] = pd.to_datetime(serie["date"])
        return serie.sort_values("date").reset_index(drop=True)

    def scale(self, serie: pd.DataFrame) -> np.ndarray:
        """Scale sales values to [0, 1]. LSTM is sensitive to input scale,
        unlike the tree-based models (XGBoost, Random Forest).

        Args:
            serie (pd.DataFrame): Series with column sales.

        Returns:
            np.ndarray: Scaled values, shape (n, 1).
        """
        self.scaler = MinMaxScaler()
        values = serie["sales"].values.reshape(-1, 1)
        return self.scaler.fit_transform(values)

    def create_windows(self, scaled_values: np.ndarray):
        """Build sliding windows: use `window` past days to predict the next one.

        Args:
            scaled_values (np.ndarray): Scaled series values.

        Returns:
            tuple: (X, y) arrays ready for the LSTM (X shape: samples, timesteps, features).
        """
        X, y = [], []
        for i in range(self.window, len(scaled_values)):
            X.append(scaled_values[i - self.window:i, 0])
            y.append(scaled_values[i, 0])
        X = np.array(X).reshape(-1, self.window, 1)
        y = np.array(y)
        return X, y

    def split_train_test(self, X: np.ndarray, y: np.ndarray, test_days: int = 90):
        """Chronological train/test split (last `test_days` windows go to test).

        Args:
            X (np.ndarray): Input windows.
            y (np.ndarray): Target values.
            test_days (int): Number of trailing windows reserved for testing.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test = X[:-test_days], X[-test_days:]
        y_train, y_test = y[:-test_days], y[-test_days:]
        logger.info(f"train={len(X_train)} windows, test={len(X_test)} windows")
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the LSTM model.

        Args:
            X_train (np.ndarray): Training input windows.
            y_train (np.ndarray): Training targets.
        """
        self.model = Sequential([
            Input(shape=(self.window, 1)),
            LSTM(50, activation="tanh"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self.model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict and inverse-transform back to real sales units.

        Args:
            X_test (np.ndarray): Test input windows.

        Returns:
            np.ndarray: Predicted sales values (real scale).
        """
        pred_scaled = self.model.predict(X_test, verbose=0)
        return self.scaler.inverse_transform(pred_scaled).flatten()

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform the scaled test targets back to real sales units.

        Args:
            y_scaled (np.ndarray): Scaled target values.

        Returns:
            np.ndarray: Real-scale target values.
        """
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

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