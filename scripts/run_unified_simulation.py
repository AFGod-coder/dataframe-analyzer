"""
Run script: backtest of the 4 BASE models (Prophet, Random Forest, XGBoost,
LSTM) over the 51-series stratified sample used in the article, with the same
90-day test window and through the exact ForecastService.get_forecast() code
path the web app uses (recursive multi-step forecasts for the lag-based
models, via service/adapters.py).

Produces results/unified_metrics.csv (one row per model x series). The
Ensemble's metrics come from scripts/run_ensemble_simulation.py; both files are
merged into results/final_comparison.csv by scripts/compute_article_stats.py.

Resumable and chunkable (each series is flushed to disk as soon as it is done):
    python3 scripts/run_unified_simulation.py
    MODEL_FILTER=LSTM SERIES_START=0 SERIES_END=10 python3 scripts/run_unified_simulation.py
"""

import logging
import os
import time

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.service.forecast_service import ForecastService

logging.basicConfig(level=logging.WARNING)  # keep per-model noise down
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.csv"
TEST_DAYS = 90
OUTPUT_PATH = os.environ.get("UNIFIED_OUTPUT", "results/unified_metrics.csv")
SUMMARY_PATH = os.environ.get("UNIFIED_SUMMARY", "results/unified_metrics_summary.csv")
SAMPLE_FRACTION = 0.10  # 10% de las 500 series ~= 51 series
RANDOM_SEED = 42

# The 4 base models of the article (section 4.3), i.e. the inputs of the Ensemble.
OFFICIAL_MODELS = ["Prophet", "Random Forest", "XGBoost", "LSTM"]  # inputs of the Ensemble


def select_representative_sample(df: pd.DataFrame, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Select a stratified representative sample of (store, item) pairs.

    Identical to the function already used in run_prophet_simulation_sample.py
    and run_lstm_simulation_sample.py (same fraction, same seed) so this
    script evaluates the exact same 51 series already used for Prophet/LSTM,
    instead of a different random subset.

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
    # 1. Carga de datos + muestra (reutiliza DataLoader y la misma funcion
    #    de muestreo estratificado de los scripts anteriores)
    df = DataLoader(DATA_PATH).load()
    pairs = select_representative_sample(df, SAMPLE_FRACTION, RANDOM_SEED)
    print(f"Muestra estratificada: {len(pairs)} series (store, item).")

    # 2. Servicio (carga y limpia el dataset una sola vez)
    service = ForecastService(DATA_PATH)

    # 3. Simulacion: cada modelo registrado, sobre cada serie de la muestra.
    #    MODEL_FILTER (variable de entorno opcional) permite correr un solo
    #    modelo a la vez -- util para ejecutar la simulacion completa en
    #    tandas cortas sin perder progreso si una tanda se interrumpe.
    model_filter = os.environ.get("MODEL_FILTER")
    models_to_run = [model_filter] if model_filter else OFFICIAL_MODELS

    # Optional chunking by series index, so a slow model (many series x many
    # trees/epochs) can be run across several short invocations without
    # losing progress -- each chunk appends to the same per-series CSV.
    start_idx = int(os.environ.get("SERIES_START", 0))
    end_idx = int(os.environ.get("SERIES_END", len(pairs)))
    chunk_pairs = pairs[start_idx:end_idx]
    print(f"Procesando series [{start_idx}:{end_idx}] de {len(pairs)} ({len(chunk_pairs)} series).")

    # Resumable: load whatever is already in OUTPUT_PATH so a call that
    # gets cut off by its own timeout loses nothing -- the next call just
    # skips the (model, store, item) combos already present and keeps
    # going, flushing to disk after every single series instead of only
    # at the very end.
    os.makedirs("results", exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        done_keys = set(map(tuple, done_df[["model", "store", "item"]].values.tolist()))
    else:
        done_df = pd.DataFrame(columns=["model", "store", "item", "MAE", "RMSE", "MAPE"])
        done_keys = set()

    rows = []
    for model_name in models_to_run:
        t0 = time.time()
        n_ok, n_fail, n_skip = 0, 0, 0
        for store, item in chunk_pairs:
            if (model_name, store, item) in done_keys:
                n_skip += 1
                continue
            try:
                result = service.get_forecast(
                    store=store, item=item, horizon_days=14,
                    model_name=model_name, backtest_days=TEST_DAYS,
                )
                row = {
                    "model": model_name,
                    "store": store, "item": item,
                    "MAE": result.metrics["MAE"],
                    "RMSE": result.metrics["RMSE"],
                    "MAPE": result.metrics["MAPE"],
                }
                rows.append(row)
                done_keys.add((model_name, store, item))
                # Flush immediately: append this one row to OUTPUT_PATH so
                # a timeout mid-chunk never loses an already-computed series.
                pd.DataFrame([row]).to_csv(
                    OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False
                )
                n_ok += 1
            except Exception as exc:
                logger.warning(f"[{model_name}] store={store} item={item} FAILED: {exc}")
                n_fail += 1
        elapsed = time.time() - t0
        print(f"[{model_name}] {n_ok} nuevas OK, {n_skip} ya hechas, {n_fail} fallidas, {elapsed:.1f}s")

    # 4. Agregacion de resultados: se relee OUTPUT_PATH completo (cada
    #    serie ya quedo escrita ahi mismo apenas se calculo, en el paso 3).
    results = pd.read_csv(OUTPUT_PATH)
    summary = (
        results.groupby("model")[["MAE", "RMSE", "MAPE"]]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    # 5. Guardado
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH)

    print(f"\n=== Resumen por modelo sobre {len(pairs)} series (muestra estratificada, TEST_DAYS={TEST_DAYS}) ===")
    print(summary)
    print(f"\nDetalle por serie: {OUTPUT_PATH}")
    print(f"Resumen agregado:  {SUMMARY_PATH}")
