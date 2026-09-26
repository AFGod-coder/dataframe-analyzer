"""
Forecast service: the layer between the web app (UI) and the forecasting
models.

The web app never talks to a model class directly. It always goes
through `ForecastService`, so that once section 2.3 decides which of
the 4 compared models performs best, only MODEL_REGISTRY below needs to
change (or a new default_model argument) -- no UI code has to be
rewritten.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from dataframe_analyzer.data.cleaner import DataCleaner
from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.prophet_model import ProphetForecaster

logger = logging.getLogger(__name__)

# Pluggable model registry. Add new entries here (e.g. "SARIMA": SarimaForecaster)
# once section 2.3 implements and validates them; the web app's model
# selector reads this dict automatically.
MODEL_REGISTRY = {
    "Prophet": ProphetForecaster,
}

DEFAULT_MODEL = "Prophet"
DEFAULT_BACKTEST_DAYS = 90


@dataclass
class ForecastResult:
    """Container for everything the UI needs to render one forecast.

    Attributes:
        history (pd.DataFrame): Full historical series for the chosen
            store-item combination, columns ds, y.
        backtest_predictions (pd.DataFrame): Model predictions over the
            held-out backtest window, columns ds, yhat.
        metrics (dict): {"MAE": ..., "RMSE": ..., "MAPE": ...} computed on
            the backtest window.
        future_forecast (pd.DataFrame): Predictions for `horizon_days`
            beyond the last historical date, columns ds, yhat.
        model_name (str): Name of the model used (key in MODEL_REGISTRY).
    """
    history: pd.DataFrame
    backtest_predictions: pd.DataFrame
    metrics: dict
    future_forecast: pd.DataFrame
    model_name: str


class ForecastService:
    """Loads the dataset once and serves forecasts for any store-item pair."""

    def __init__(self, data_path: str):
        """Load and clean the dataset, keeping the raw 'date' column.

        Uses DataLoader + DataCleaner only (not the full pipeline in
        scripts/run_pipeline.py), because that pipeline's output
        (data/processed/clean_dataframe.csv) decomposes and drops 'date'
        into day/month/year columns -- useful for the EDA, but Prophet
        needs the original date column to build its own time features.

        Args:
            data_path (str): Path to the raw dataset (must have columns
                date, store, item, sales -- e.g. data/raw/train.csv).
        """
        self.data_path = data_path
        df_raw = DataLoader(data_path).load()
        self._df = DataCleaner().clean(df_raw)
        logger.info(f"ForecastService loaded {len(self._df)} rows from {data_path}")

    def list_stores(self) -> list:
        """Return the sorted list of available store ids."""
        return sorted(self._df["store"].unique().tolist())

    def list_items(self) -> list:
        """Return the sorted list of available product (item) ids."""
        return sorted(self._df["item"].unique().tolist())

    def get_forecast(
        self,
        store: int,
        item: int,
        horizon_days: int,
        model_name: str = DEFAULT_MODEL,
        backtest_days: int = DEFAULT_BACKTEST_DAYS,
    ) -> ForecastResult:
        """Build a full forecast (backtest metrics + future prediction).

        Args:
            store (int): Store id to forecast for.
            item (int): Product id to forecast for.
            horizon_days (int): How many days beyond the last known date
                to forecast.
            model_name (str, optional): Key into MODEL_REGISTRY. Defaults
                to DEFAULT_MODEL ("Prophet").
            backtest_days (int, optional): Size of the held-out window
                used to compute MAE/RMSE/MAPE. Defaults to 90.

        Returns:
            ForecastResult: everything the UI needs to render.

        Raises:
            ValueError: if model_name is not registered, or there is no
                data for the given store-item combination.
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

        # 1. Backtest: train on everything except the last `backtest_days`,
        #    predict that window, and compute error metrics against it.
        train, test, _cutoff = forecaster.split_train_test(series, test_days=backtest_days)
        forecaster.fit(train)
        backtest_predictions = forecaster.predict(test)
        metrics = forecaster.compute_metrics(test["y"].values, backtest_predictions["yhat"].values)
        logger.info(f"Backtest metrics for store={store}, item={item}, model={model_name}: {metrics}")

        # 2. Future forecast: refit on the FULL series (most up-to-date
        #    model) and predict `horizon_days` beyond the last known date.
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
