"""Backtest of the 4 base models on the stratified sample of 51 series.

The base models are Prophet, Random Forest, XGBoost and LSTM. The script
uses a 90-day test window and the same code as the web app,
`ForecastService.get_forecast()`. The lag models make recursive
forecasts through `service/adapters.py`.

Output: `results/unified_metrics.csv`, with one row for each model and
series. The ensemble metrics come from `scripts/run_ensemble_simulation.py`.
`scripts/compute_article_stats.py` joins both files into
`results/final_comparison.csv`.

The script can continue after a stop, and it can run in parts. Each
series is saved to disk when it is finished:
    python3 scripts/run_unified_simulation.py
    MODEL_FILTER=LSTM SERIES_START=0 SERIES_END=10 python3 scripts/run_unified_simulation.py
"""

import logging
import os
import time

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.service.forecast_service import ForecastService

logging.basicConfig(level=logging.WARNING)  # Show fewer log messages from the models.
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/train.csv"
TEST_DAYS = 90
OUTPUT_PATH = os.environ.get("UNIFIED_OUTPUT", "results/unified_metrics.csv")
SUMMARY_PATH = os.environ.get("UNIFIED_SUMMARY", "results/unified_metrics_summary.csv")
SAMPLE_FRACTION = 0.10  # 10% of the 500 series, about 51 series.
RANDOM_SEED = 42

# The 4 base models, which are the inputs of the ensemble.
OFFICIAL_MODELS = ["Prophet", "Random Forest", "XGBoost", "LSTM"]


def select_representative_sample(df: pd.DataFrame, fraction: float, seed: int) -> list[tuple[int, int]]:
    """Select a stratified sample of (store, item) pairs.

    The series are split into three groups by total sales (low, medium
    and high). The same fraction is taken from each group. With the same
    fraction and seed, the function always gives the same 51 series, so
    all models are evaluated on the same data.

    Args:
        df (pd.DataFrame): Full raw dataset with the columns `date`,
            `store`, `item` and `sales`.
        fraction (float): Fraction of the 500 (store, item) series to
            keep.
        seed (int): Random seed for reproducible results.

    Returns:
        list[tuple[int, int]]: Selected (store, item) pairs.
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
    # 1. Load the data and select the stratified sample.
    df = DataLoader(DATA_PATH).load()
    pairs = select_representative_sample(df, SAMPLE_FRACTION, RANDOM_SEED)
    print(f"Muestra estratificada: {len(pairs)} series (store, item).")

    # 2. Create the service (it loads and cleans the dataset only once).
    service = ForecastService(DATA_PATH)

    # 3. Simulation: run each model on each series of the sample.
    #    MODEL_FILTER (optional environment variable) runs only one model.
    #    This helps to run the full simulation in short parts without
    #    losing progress if one part stops.
    model_filter = os.environ.get("MODEL_FILTER")
    models_to_run = [model_filter] if model_filter else OFFICIAL_MODELS

    # Optional parts by series index. A slow model can run in several
    # short calls without losing progress. Each part adds rows to the same
    # CSV file.
    start_idx = int(os.environ.get("SERIES_START", 0))
    end_idx = int(os.environ.get("SERIES_END", len(pairs)))
    chunk_pairs = pairs[start_idx:end_idx]
    print(f"Procesando series [{start_idx}:{end_idx}] de {len(pairs)} ({len(chunk_pairs)} series).")

    # Continue after a stop: read the rows already in OUTPUT_PATH. The
    # next call skips the (model, store, item) combinations that are
    # already done. Each series is saved to disk when it is finished, not
    # only at the end.
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
                # Save this row now, so a stop in the middle of a part does
                # not lose a series that is already calculated.
                pd.DataFrame([row]).to_csv(
                    OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False
                )
                n_ok += 1
            except Exception as exc:
                logger.warning(f"[{model_name}] store={store} item={item} FAILED: {exc}")
                n_fail += 1
        elapsed = time.time() - t0
        print(f"[{model_name}] {n_ok} nuevas OK, {n_skip} ya hechas, {n_fail} fallidas, {elapsed:.1f}s")

    # 4. Summarize the results. OUTPUT_PATH is read again, because step 3
    #    saved each series there.
    results = pd.read_csv(OUTPUT_PATH)
    summary = (
        results.groupby("model")[["MAE", "RMSE", "MAPE"]]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    # 5. Save the files.
    os.makedirs("results", exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH)

    print(f"\n=== Resumen por modelo sobre {len(pairs)} series (muestra estratificada, TEST_DAYS={TEST_DAYS}) ===")
    print(summary)
    print(f"\nDetalle por serie: {OUTPUT_PATH}")
    print(f"Resumen agregado:  {SUMMARY_PATH}")
