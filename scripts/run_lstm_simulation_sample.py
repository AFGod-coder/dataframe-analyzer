"""
Run script: LSTM simulation over the same representative sample used for Prophet.

Two evaluation protocols are reported for every series:
    - 90d  : recursive 90-day forecast, no real sales seen in the test period.
            Same protocol as Prophet -> directly comparable with it.
    - 1step: one-step-ahead prediction using the previous real days.
            Same protocol as the tree-based models (XGBoost, Random Forest).

TRAZABILIDAD:
    1. Carga de datos                 -> DataLoader.load()
    2. Selección de la muestra        -> select_representative_sample()
                                        (misma lógica, mismo seed=42 y misma
                                        fracción que run_prophet_simulation_sample.py,
                                        para comparar los modelos sobre las mismas series)
    3. Por cada serie (store, item):
        3.1 Selección de la serie      -> LSTMForecaster.prepare_series()
        3.2 Escalado (solo con train)  -> LSTMForecaster.scale()
        3.3 Construcción de ventanas   -> LSTMForecaster.create_windows()
        3.4 Split cronológico          -> LSTMForecaster.split_train_test()
        3.5 Entrenamiento              -> LSTMForecaster.fit()
        3.6 Pronóstico un día vista    -> LSTMForecaster.predict()
        3.7 Pronóstico recursivo 90d   -> LSTMForecaster.predict_recursive()
        3.8 Valores reales (desescala) -> LSTMForecaster.inverse_transform_y()
        3.9 Cálculo de métricas        -> LSTMForecaster.compute_metrics()
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
        scaled = forecaster.scale(serie, test_days=TEST_DAYS)
        X, y = forecaster.create_windows(scaled)
        X_train, X_test, y_train, y_test = forecaster.split_train_test(X, y, test_days=TEST_DAYS)

        forecaster.fit(X_train, y_train)

        y_true = forecaster.inverse_transform_y(y_test)
        y_pred_1step = forecaster.predict(X_test)
        y_pred_90d = forecaster.predict_recursive(scaled, test_days=TEST_DAYS)

        m1 = forecaster.compute_metrics(y_true, y_pred_1step)
        m90 = forecaster.compute_metrics(y_true, y_pred_90d)

        rows.append({
            "store": store,
            "item": item,
            "MAE_90d": m90["MAE"], "RMSE_90d": m90["RMSE"], "MAPE_90d": m90["MAPE"],
            "MAE_1step": m1["MAE"], "RMSE_1step": m1["RMSE"], "MAPE_1step": m1["MAPE"],
        })
        logger.info(
            f"store={store} item={item} | "
            f"90d: MAE={m90['MAE']:.3f} RMSE={m90['RMSE']:.3f} MAPE={m90['MAPE']:.2f}% | "
            f"1step: MAE={m1['MAE']:.3f} RMSE={m1['RMSE']:.3f} MAPE={m1['MAPE']:.2f}%"
        )

    # 4. Agregación de resultados
    results = pd.DataFrame(rows)
    cols_90d = ["MAE_90d", "RMSE_90d", "MAPE_90d"]
    cols_1step = ["MAE_1step", "RMSE_1step", "MAPE_1step"]

    # 5. Guardado (crea la carpeta results/ si no existe)
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    n = len(results)
    print(f"\n=== LSTM - pronóstico recursivo a 90 días ({n} series, comparable con Prophet) ===")
    print(results[cols_90d].agg(["mean", "median", "std"]))
    print(f"\n=== LSTM - un día vista ({n} series, comparable con XGBoost / Random Forest) ===")
    print(results[cols_1step].agg(["mean", "median", "std"]))
    print(f"\nResultados detallados guardados en {OUTPUT_PATH}")