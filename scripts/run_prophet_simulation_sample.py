"""
Run script: Prophet simulation over a representative sample of series.

TRAZABILIDAD:
    1. Carga de datos                 -> DataLoader.load()
    2. Selección de la muestra        -> select_representative_sample()
    3. Por cada serie (store, item):
        3.1 Selección de la serie      -> ProphetForecaster.prepare_series()
        3.2 Split cronológico          -> ProphetForecaster.split_train_test()
        3.3 Entrenamiento              -> ProphetForecaster.fit()
        3.4 Predicción                 -> ProphetForecaster.predict()
        3.5 Cálculo de métricas        -> ProphetForecaster.compute_metrics()
    4. Agregación de resultados       -> pandas.DataFrame + mean/median/std
    5. Guardado de resultados         -> results/prophet_metrics.csv
"""

import logging

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.prophet_model import ProphetForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.csv"
TEST_DAYS = 90
OUTPUT_PATH = "results/prophet_metrics.csv"
SAMPLE_FRACTION = 0.10  # 10% de las 500 series ~= 50 series
RANDOM_SEED = 42


def select_representative_sample(df: pd.DataFrame, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Select a stratified representative sample of (store, item) pairs.

    Stratifies by total sales volume (terciles: low/medium/high) so the
    sample covers slow-, medium- and fast-moving store-item combinations
    instead of only the highest-volume series.

    Args:
        df (pd.DataFrame): Full raw dataframe with columns date, store, item, sales.
        fraction (float): Fraction of the 500 (store, item) series to keep.
        seed (int): Random seed for reproducibility.

    Returns:
        list[tuple[int, int]]: (store, item) pairs selected.
    """
    totals = df.groupby(["store", "item"])["sales"].sum().reset_index()
    totals["volume_tercile"] = pd.qcut(totals["sales"], 3, labels=["low", "medium", "high"])

    sample = (
        totals.groupby("volume_tercile", group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=seed))
    )
    pairs = list(zip(sample["store"], sample["item"]))
    logger.info(f"Selected {len(pairs)} of {len(totals)} series ({fraction:.0%} stratified sample).")
    return pairs


if __name__ == "__main__":
    # 1. Carga de datos (reutiliza DataLoader ya existente en el repo)
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # 2. Selección de la muestra representativa (estratificada por volumen)
    pairs = select_representative_sample(df, SAMPLE_FRACTION, RANDOM_SEED)

    # 3. Simulación serie por serie (reutiliza ProphetForecaster ya existente)
    rows = []
    for store, item in pairs:
        forecaster = ProphetForecaster()
        serie = forecaster.prepare_series(df, store=store, item=item)
        train, test, cutoff = forecaster.split_train_test(serie, test_days=TEST_DAYS)

        forecaster.fit(train)
        forecast = forecaster.predict(test)

        y_true = test["y"].values
        y_pred = forecast["yhat"].values
        metrics = forecaster.compute_metrics(y_true, y_pred)

        rows.append({"store": store, "item": item, "cutoff": cutoff.date(), **metrics})
        logger.info(
            f"store={store} item={item} -> "
            f"MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} MAPE={metrics['MAPE']:.2f}%"
        )

    # 4. Agregación de resultados
    results = pd.DataFrame(rows)
    summary = results[["MAE", "RMSE", "MAPE"]].agg(["mean", "median", "std"])

    # 5. Guardado (crea la carpeta results/ si no existe)
    import os
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== Prophet - resumen sobre {len(results)} series (muestra estratificada) ===")
    print(summary)
    print(f"\nResultados detallados guardados en {OUTPUT_PATH}")