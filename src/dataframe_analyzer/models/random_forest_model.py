"""
Random Forest model for demand forecasting.

Given a series with columns [date, sales] (already filtered for a
specific store-item combination), this module builds calendar and lag
features, performs a chronological split, trains a Random Forest
regressor, and calculates evaluation metrics.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class RandomForestForecaster:
    """Trains a Random Forest regressor on a single store-item time series."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        """Store the hyperparameters used when the model is trained.

        Args:
            n_estimators (int): Number of trees (B in f_hat(x) = (1/B) * sum_b T_b(x)).
                Unlike XGBoost, these trees are trained independently and
                in parallel, not sequentially correcting each other's errors.
            max_depth (int): Maximum depth of each individual tree.
            random_state (int): Seed for reproducibility (controls the
                bootstrap sampling of rows and the random subset of
                features considered at each split).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
        """Create calendar and lag features required by Random Forest.

        Same feature set used for XGBoost (xgboost_model.py), so that
        both tree-based models are compared under identical inputs.

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

        # Lag features: sales from previous days
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

    def fit(self, train: pd.DataFrame) -> RandomForestRegressor:
        """Train the Random Forest regressor on the training set.

        Args:
            train (pd.DataFrame): Training data with feature columns and 'sales'.

        Returns:
            RandomForestRegressor: The fitted model (also stored in self.model).
        """
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
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