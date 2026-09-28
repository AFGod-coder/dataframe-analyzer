"""
Run script: LSTM simulation over the same representative sample used for Prophet.

TRAZABILIDAD:
    1. Carga de datos                 -> DataLoader.load()
    2. Selección de la muestra        -> select_representative_sample()
                                        (misma lógica, mismo seed=42 y misma
                                        fracción que run_prophet_simulation_sample.py,
                                        para comparar los modelos sobre las mismas series)
    3. Por cada serie (store, item):
        3.1 Selección de la serie      -> LSTMForecaster.prepare_series()
        3.2 Escalado                   -> LSTMForecaster.scale()
        3.3 Construcción de ventanas   -> LSTMForecaster.create_windows()
        3.4 Split cronológico          -> LSTMForecaster.split_train_test()
        3.5 Entrenamiento              -> LSTMForecaster.fit()
        3.6 Predicción y desescalado   -> LSTMForecaster.predict() + inverse_transform_y()
        3.7 Cálculo de métricas        -> LSTMForecaster.compute_metrics()
    4. Agregación de resultados       -> pandas.DataFrame + mean/median/std
    5. Guardado de resultados         -> results/lstm_metrics.csv
"""

import logging
import os

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.lstm_model import LSTMForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.csv"
WINDOW = 30
TEST_DAYS = 90
OUTPUT_PATH = "results/lstm_metrics.csv"
SAMPLE_FRACTION = 0.10  # misma fracción que Prophet (~51 de 500 series)
RANDOM_SEED = 42  # misma semilla que Prophet -> misma muestra exacta de series


def select_representative_sample(df: pd.DataFrame, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Select the same stratified representative sample used for Prophet.

    Identical logic (stratify (store, item) pairs by total sales volume
    tercile, same fraction and seed) as in run_prophet_simulation_sample.py,
    so both models are evaluated on exactly the same store-item series and
    the results are directly comparable in the final report table.

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

    # 2. Selección de la muestra (idéntica a la de Prophet)
    pairs = select_representative_sample(df, SAMPLE_FRACTION, RANDOM_SEED)

    # 3. Simulación serie por serie (reutiliza LSTMForecaster ya existente)
    rows = []
    for store, item in pairs:
        forecaster = LSTMForecaster(window=WINDOW)
        serie = forecaster.prepare_series(df, store=store, item=item)
        scaled = forecaster.scale(serie)
        X, y = forecaster.create_windows(scaled)
        X_train, X_test, y_train, y_test = forecaster.split_train_test(X, y, test_days=TEST_DAYS)

        forecaster.fit(X_train, y_train)
        y_pred = forecaster.predict(X_test)
        y_true = forecaster.inverse_transform_y(y_test)

        metrics = forecaster.compute_metrics(y_true, y_pred)
        rows.append({"store": store, "item": item, **metrics})
        logger.info(
            f"store={store} item={item} -> "
            f"MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} MAPE={metrics['MAPE']:.2f}%"
        )

    # 4. Agregación de resultados
    results = pd.DataFrame(rows)
    summary = results[["MAE", "RMSE", "MAPE"]].agg(["mean", "median", "std"])

    # 5. Guardado (crea la carpeta results/ si no existe)
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== LSTM - resumen sobre {len(results)} series (misma muestra que Prophet) ===")
    print(summary)
    print(f"\nResultados detallados guardados en {OUTPUT_PATH}")