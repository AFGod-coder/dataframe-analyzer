"""
XGBoost model for demand forecasting.

Given a series with columns [date, sales] (already filtered for a
specific store-item combination), this module builds calendar and lag
features, performs a chronological split, trains an XGBoost regressor,
and calculates evaluation metrics.
"""

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """Trains an XGBoost regressor on a single store-item time series."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """Store the hyperparameters used when the model is trained.

        Args:
            n_estimators (int): Number of boosting rounds (trees). Corresponds
                to K in the y_hat = sum_{k=1}^{K} f_k(x) formulation.
            max_depth (int): Maximum depth of each tree (controls complexity,
                related to the Omega(f) penalty term in the objective).
            learning_rate (float): Step size shrinkage used to prevent
                overfitting between boosting rounds.
            random_state (int): Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_columns = None

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Filter one store-item series and sort it chronologically.

        Args:
            df (pd.DataFrame): Full dataset with columns date, store, item, sales.
            store (int): Store id to filter.
            item (int): Item id to filter.

        Returns:
            pd.DataFrame: Series with columns date, sales, sorted chronologically.
        """
        serie = df[(df["store"] == store) & (df["item"] == item)][["date", "sales"]].copy()
        serie["date"] = pd.to_datetime(serie["date"])
        return serie.sort_values("date").reset_index(drop=True)

    def build_features(self, serie: pd.DataFrame) -> pd.DataFrame:
        """Create calendar and lag features required by XGBoost.

        Unlike ARIMA or Prophet, XGBoost does not model the sequence
        implicitly, so past sales values and calendar information must
        be provided explicitly as columns (this is the x_i vector in
        y_hat_i = sum_k f_k(x_i)).

        Args:
            serie (pd.DataFrame): Series with columns date, sales.

        Returns:
            pd.DataFrame: Series with added feature columns. Rows without
                enough history to compute the largest lag are dropped.
        """
        features = serie.copy()

        # Calendar features
        features["day_of_week"] = features["date"].dt.dayofweek
        features["month"] = features["date"].dt.month
        features["day"] = features["date"].dt.day
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Lag features: sales from previous days, used as the model's "memory"
        for lag in (1, 7, 14, 30):
            features[f"lag_{lag}"] = features["sales"].shift(lag)

        # Rolling mean as an additional short-term trend signal
        features["rolling_mean_7"] = features["sales"].shift(1).rolling(window=7).mean()

        # Drop initial rows where lag/rolling features could not be computed
        features = features.dropna().reset_index(drop=True)

        self.feature_columns = [
            "day_of_week", "month", "day", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7",
        ]
        return features

    def split_train_test(self, features: pd.DataFrame, test_days: int = 90):
        """Chronological train/test split (last `test_days` days go to test).

        Args:
            features (pd.DataFrame): Series with engineered feature columns.
            test_days (int): Number of trailing days reserved for testing.

        Returns:
            tuple: (train_df, test_df, cutoff_date)
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
        """Train the XGBoost regressor on the training set.

        Args:
            train (pd.DataFrame): Training data with feature columns and 'sales'.

        Returns:
            XGBRegressor: The fitted model (also stored in self.model).
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
        """Predict sales for the rows present in the test set.

        Args:
            test (pd.DataFrame): Test data with the same feature columns used in fit().

        Returns:
            pd.DataFrame: Predictions with columns date, yhat.
        """
        preds = self.model.predict(test[self.feature_columns])
        result = test[["date"]].copy()
        result["yhat"] = preds
        return result

    def get_feature_importance(self) -> pd.Series:
        """Return feature importance scores from the trained model.

        Useful for the results section (2.3/Modelos Matematico): shows which
        variables (e.g. recent lags vs. calendar features) the model relied
        on the most.

        Returns:
            pd.Series: Feature names mapped to importance scores, sorted descending.
        """
        importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
        return importances.sort_values(ascending=False)

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