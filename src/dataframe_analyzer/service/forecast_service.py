"""Forecast service: the layer between the web app and the models.

The web app never uses a model class directly. It always uses
`ForecastService`. Because of this, to add or change a model we only
need to change `MODEL_REGISTRY` (or the default model). The interface
code does not change.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dataframe_analyzer.data.cleaner import DataCleaner
from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.ensemble_model import EnsembleForecaster
from dataframe_analyzer.models.prophet_model import ProphetForecaster
from dataframe_analyzer.service.adapters import LSTMAdapter, RandomForestAdapter, XGBoostAdapter

logger = logging.getLogger(__name__)

# Model registry. The model selector in the web app reads this dictionary.
# "Ensemble (Fusionado)" is the main mathematical model and the default
# model of the app. It combines the 4 base models below with stacking:
# a weighted sum with non-negative weights that add up to 1.
MODEL_REGISTRY = {
    "Ensemble (Fusionado)": EnsembleForecaster,
    "Prophet": ProphetForecaster,
    "Random Forest": RandomForestAdapter,
    "XGBoost": XGBoostAdapter,
    "LSTM": LSTMAdapter,
}

DEFAULT_MODEL = "Ensemble (Fusionado)"
DEFAULT_BACKTEST_DAYS = 90


@dataclass
class ForecastResult:
    """All the data that the interface needs to show one forecast.

    Attributes:
        history (pd.DataFrame): Full history of the chosen store and
            item, with the columns `ds` and `y`.
        backtest_predictions (pd.DataFrame): Model predictions for the
            backtest window, with the columns `ds` and `yhat`.
        metrics (dict): Keys `MAE`, `RMSE` and `MAPE`, calculated on the
            backtest window.
        future_forecast (pd.DataFrame): Predictions for the
            `horizon_days` after the last known date, with the columns
            `ds` and `yhat`.
        model_name (str): Name of the model used (a key of
            `MODEL_REGISTRY`).
    """
    history: pd.DataFrame
    backtest_predictions: pd.DataFrame
    metrics: dict
    future_forecast: pd.DataFrame
    model_name: str


class ForecastService:
    """Load the dataset once and give forecasts for any store and item."""

    def __init__(self, data_path: str):
        """Load and clean the dataset, and keep the original `date` column.

        Only `DataLoader` and `DataCleaner` are used, not the full
        pipeline in `scripts/run_pipeline.py`. That pipeline changes
        `date` into day, month and year columns and removes it. This is
        useful for the EDA, but Prophet needs the original date column.

        Args:
            data_path (str): Path to the raw dataset. It must have the
                columns `date`, `store`, `item` and `sales` (for example
                `data/raw/train.csv`).
        """
        self.data_path = data_path
        df_raw = DataLoader(data_path).load()
        self._df = DataCleaner().clean(df_raw)
        logger.info(f"ForecastService loaded {len(self._df)} rows from {data_path}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ForecastService":
        """Create a service from a dataset in memory instead of a CSV file.

        The upload flow of the app uses it. The module
        `dataframe_analyzer.data.sales_upload` first checks and cleans the
        user's file into the same columns. Then the forecasts use exactly
        the same code as for the project dataset.

        Args:
            df (pd.DataFrame): Columns `date`, `store`, `item` and `sales`,
                with one row for each store, item and day, and no gaps.

        Returns:
            ForecastService: A service that uses `df`.

        Raises:
            ValueError: If a required column is missing.
        """
        missing = {"date", "store", "item", "sales"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for ForecastService: {sorted(missing)}")
        service = cls.__new__(cls)
        service.data_path = None
        service._df = DataCleaner().clean(df[["date", "store", "item", "sales"]])
        logger.info(f"ForecastService built from an in-memory dataset of {len(service._df)} rows")
        return service

    def list_stores(self) -> list:
        """Get the available store ids.

        Returns:
            list: Store ids in ascending order.
        """
        return sorted(self._df["store"].unique().tolist())

    def list_items(self) -> list:
        """Get the available product (item) ids.

        Returns:
            list: Item ids in ascending order.
        """
        return sorted(self._df["item"].unique().tolist())

    def get_forecast(
        self,
        store: int,
        item: int,
        horizon_days: int,
        model_name: str = DEFAULT_MODEL,
        backtest_days: int = DEFAULT_BACKTEST_DAYS,
    ) -> ForecastResult:
        """Create a full forecast: backtest metrics and future predictions.

        Args:
            store (int): Store id.
            item (int): Product id.
            horizon_days (int): Number of days to forecast after the last
                known date.
            model_name (str, optional): Key of `MODEL_REGISTRY`. Defaults
                to `DEFAULT_MODEL` ("Ensemble (Fusionado)").
            backtest_days (int, optional): Size of the test window used to
                calculate MAE, RMSE and MAPE. Defaults to 90.

        Returns:
            ForecastResult: All the data that the interface needs.

        Raises:
            ValueError: If `model_name` is not in the registry, or if there
                is no data for the store and item.
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
            )

        forecaster_cls = MODEL_REGISTRY[model_name]
        forecaster = forecaster_cls()

        series = forecaster.prepare_series(self._df, store=store, item=item)
        if series.empty:
            raise ValueError(f"No data found for store={store}, item={item}.")

        # 1. Backtest: train with all data except the last `backtest_days`,
        #    predict that window and calculate the error metrics.
        train, test, _cutoff = forecaster.split_train_test(series, test_days=backtest_days)
        forecaster.fit(train)
        backtest_predictions = forecaster.predict(test)
        metrics = forecaster.compute_metrics(test["y"].values, backtest_predictions["yhat"].values)
        logger.info(f"Backtest metrics for store={store}, item={item}, model={model_name}: {metrics}")

        # 2. Future forecast: train again with the full series (the most
        #    recent data) and predict `horizon_days` after the last date.
        full_model_forecaster = forecaster_cls()
        full_model_forecaster.fit(series)
        last_date = series["ds"].max()
        future_dates = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
        })
        future_forecast = full_model_forecaster.predict(future_dates)

        return ForecastResult(
            history=series,
            backtest_predictions=backtest_predictions,
            metrics=metrics,
            future_forecast=future_forecast,
            model_name=model_name,
        )
