"""XGBoost model for demand forecasting.

The input is a series with the columns `date` and `sales` for one store
and one item. This module builds calendar and lag features, splits the
data by date, trains an XGBoost regressor and calculates the metrics.
"""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """Train an XGBoost regressor on the series of one store and item."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Store the hyperparameters for training.

        Args:
            n_estimators (int): Number of boosting rounds (trees). It is K
                in the formula y_hat = sum_{k=1}^{K} f_k(x).
            max_depth (int): Maximum depth of each tree. It controls the
                complexity, together with the penalty Omega(f) in the
                objective.
            learning_rate (float): Step size of each boosting round. A
                small value helps to avoid overfitting.
            random_state (int): Seed for reproducible results.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_columns = None

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
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]].copy()
        serie["date"] = pd.to_datetime(serie["date"])
        return serie.sort_values("date").reset_index(drop=True)

    def build_features(self, serie: pd.DataFrame) -> pd.DataFrame:
        """Create the calendar and lag features for XGBoost.

        XGBoost does not know the order of the data, as ARIMA or Prophet
        do. So past sales and calendar information must be given as
        columns. These columns are the vector x_i in
        y_hat_i = sum_k f_k(x_i).

        Args:
            serie (pd.DataFrame): Series with the columns `date` and
                `sales`.

        Returns:
            pd.DataFrame: Series with the new feature columns. The first
            rows, which do not have enough history for the largest lag,
            are removed.
        """
        features = serie.copy()

        # Calendar features.
        features["day_of_week"] = features["date"].dt.dayofweek
        features["month"] = features["date"].dt.month
        features["day"] = features["date"].dt.day
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Lag features: sales of previous days, the "memory" of the model.
        for lag in (1, 7, 14, 30):
            features[f"lag_{lag}"] = features["sales"].shift(lag)

        # 7-day rolling mean, a signal of the short-term trend.
        features["rolling_mean_7"] = features["sales"].shift(1).rolling(window=7).mean()

        # Remove the first rows, where the lag and rolling values are empty.
        features = features.dropna().reset_index(drop=True)

        self.feature_columns = [
            "day_of_week", "month", "day", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7",
        ]
        return features

    def split_train_test(self, features: pd.DataFrame, test_days: int = 90):
        """Split the data by date: the last `test_days` days go to the test.

        Args:
            features (pd.DataFrame): Series with the feature columns.
            test_days (int): Number of last days kept for the test.

        Returns:
            tuple: `(train_df, test_df, cutoff_date)`.
        """
        cutoff = features["date"].max() - pd.Timedelta(days=test_days)
        train = features[features["date"] <= cutoff]
        test = features[features["date"] > cutoff]
        logger.info(
            f"Chronological cutoff: {cutoff.date()} | "
            f"train={len(train)} rows, test={len(test)} rows"
        )
        return train, test, cutoff

    def fit(self, train: pd.DataFrame) -> XGBRegressor:
        """Train the XGBoost regressor with the training data.

        Args:
            train (pd.DataFrame): Training data with the feature columns
                and `sales`.

        Returns:
            XGBRegressor: The trained model (also saved in `self.model`).
        """
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="reg:squarederror",
        )
        self.model.fit(train[self.feature_columns], train["sales"])
        return self.model

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Predict sales for the rows in the test data.

        Args:
            test (pd.DataFrame): Test data with the same feature columns
                used in `fit()`.

        Returns:
            pd.DataFrame: Predictions with the columns `date` and `yhat`.
        """
        preds = self.model.predict(test[self.feature_columns])
        result = test[["date"]].copy()
        result["yhat"] = preds
        return result

    def get_feature_importance(self) -> pd.Series:
        """Get the importance of each feature in the trained model.

        It shows which variables the model uses most (for example recent
        lags or calendar features).

        Returns:
            pd.Series: Importance score of each feature, from highest to
            lowest.
        """
        importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
        return importances.sort_values(ascending=False)

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