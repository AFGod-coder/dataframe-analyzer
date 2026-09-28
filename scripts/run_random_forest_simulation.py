"""
Run script: Random Forest simulation for demand forecasting, across one
or many store-item combinations.

TRAZABILIDAD:
    1. Carga de datos              -> DataLoader.load()
    2. Lista de combinaciones      -> df[['store','item']].drop_duplicates()
    3. Por cada combinación:
       3.1 Selección de la serie   -> RandomForestForecaster.prepare_series()
       3.2 Ingeniería de features  -> RandomForestForecaster.build_features()
       3.3 Split cronológico       -> RandomForestForecaster.split_train_test()
       3.4 Entrenamiento           -> RandomForestForecaster.fit()
       3.5 Predicción              -> RandomForestForecaster.predict()
       3.6 Métricas                -> RandomForestForecaster.compute_metrics()
    4. Resultados agregados        -> saved to OUTPUT_PATH + printed summary

NOTE: uses the same DATA_PATH, TEST_DAYS and feature set as
run_xgboost_simulation.py so both tree-based models are compared under
identical conditions.
"""

import logging
import os

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.random_forest_model import RandomForestForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.csv"
TEST_DAYS = 90
OUTPUT_PATH = "reports/results/random_forest_metrics_by_combination.csv"

# Set this to an integer (e.g. 20) to quickly test on a random sample of
# combinations instead of the full 500. Set to None to run on everything.
SAMPLE_SIZE = None
RANDOM_STATE = 42


def run_single_combination(df: pd.DataFrame, store: int, item: int, test_days: int = TEST_DAYS):
    """Run the full Random Forest pipeline for one store-item combination.

    Args:
        df (pd.DataFrame): Full raw dataset with columns date, store, item, sales.
        store (int): Store id.
        item (int): Item id.
        test_days (int): Number of trailing days reserved for testing.

    Returns:
        dict | None: Metrics plus store/item identifiers, or None if the
            combination did not have enough history.
    """
    forecaster = RandomForestForecaster()
    serie = forecaster.prepare_series(df, store=store, item=item)
    features = forecaster.build_features(serie)

    if len(features) < test_days + 30:
        logger.warning(f"Skipping store={store}, item={item}: not enough history.")
        return None

    train, test, _ = forecaster.split_train_test(features, test_days=test_days)
    forecaster.fit(train)
    forecast = forecaster.predict(test)

    y_true = test["sales"].values
    y_pred = forecast["yhat"].values
    metrics = forecaster.compute_metrics(y_true, y_pred)
    metrics.update({"store": store, "item": item})
    return metrics


if __name__ == "__main__":
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    combinations = df[["store", "item"]].drop_duplicates().reset_index(drop=True)
    if SAMPLE_SIZE is not None:
        combinations = combinations.sample(
            n=SAMPLE_SIZE, random_state=RANDOM_STATE
        ).reset_index(drop=True)

    logger.info(f"Running Random Forest simulation on {len(combinations)} store-item combinations...")

    results = []
    for _, row in combinations.iterrows():
        metrics = run_single_combination(df, store=row["store"], item=row["item"])
        if metrics is not None:
            results.append(metrics)

    results_df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Results saved to: {OUTPUT_PATH}")

    print(f"\nEvaluated {len(results_df)} combinations.")
    print("\nOverall average metrics (across all combinations):")
    print(results_df[["MAE", "RMSE", "MAPE"]].mean())

    print("\nTop 5 best-performing combinations (lowest MAPE):")
    print(results_df.sort_values("MAPE").head())

    print("\nTop 5 worst-performing combinations (highest MAPE):")
    print(results_df.sort_values("MAPE", ascending=False).head())