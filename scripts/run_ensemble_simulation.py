"""
Run script: backtest the EnsembleForecaster (section 2.4's single fused
mathematical model) over the same 51-series stratified sample already
used for Prophet/Random Forest/XGBoost/LSTM in results/unified_metrics.csv,
with the same TEST_DAYS=90 backtest window, so the comparison table in
section 2.3 is directly comparable.

Resumable by design: each series' result is flushed to disk immediately,
and a second run skips (store, item) pairs already present in the output
file -- the same pattern used for the LSTM run in Hito 2, needed because
each series takes ~50-90s (training 2 copies of 4 base models) and a
single process call may be cut off.

Usage:
    python3 scripts/run_ensemble_simulation.py            # all 51 series
    SERIES_START=10 SERIES_END=15 python3 scripts/run_ensemble_simulation.py
"""

import logging
import os
import time

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.ensemble_model import EnsembleForecaster

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TEST_DAYS = 90
OUTPUT_PATH = "results/ensemble_metrics.csv"


def select_representative_sample(df: pd.DataFrame, fraction: float, seed: int) -> list:
    """Identical to the function used for every other model's simulation
    (same fraction, same seed) so this evaluates the exact same 51 series."""
    totals = df.groupby(["store", "item"])["sales"].sum().reset_index()
    totals["volume_tercile"] = pd.qcut(totals["sales"], 3, labels=["low", "medium", "high"])
    sample = (
        totals.groupby("volume_tercile", group_keys=False)
        .apply(lambda g: g.sample(frac=fraction, random_state=seed))
    )
    return list(zip(sample["store"], sample["item"]))


if __name__ == "__main__":
    df = DataLoader("data/raw/train.csv").load()
    pairs = select_representative_sample(df, fraction=0.10, seed=42)

    start_idx = int(os.environ.get("SERIES_START", 0))
    end_idx = int(os.environ.get("SERIES_END", len(pairs)))
    chunk_pairs = pairs[start_idx:end_idx]

    os.makedirs("results", exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        done_df = pd.read_csv(OUTPUT_PATH)
        done_keys = set(map(tuple, done_df[["store", "item"]].values.tolist()))
    else:
        done_keys = set()

    for store, item in chunk_pairs:
        if (store, item) in done_keys:
            continue
        t0 = time.time()
        try:
            ens = EnsembleForecaster()
            series = ens.prepare_series(df, store=store, item=item)
            train, test, _cutoff = ens.split_train_test(series, test_days=TEST_DAYS)
            ens.fit(train)
            preds = ens.predict(test[["ds"]])
            metrics = ens.compute_metrics(test["y"].values, preds["yhat"].values)
            row = {
                "model": "Ensemble (Fusionado)", "store": store, "item": item,
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"],
                "w_prophet": ens.weights["Prophet"], "w_rf": ens.weights["Random Forest"],
                "w_xgb": ens.weights["XGBoost"], "w_lstm": ens.weights["LSTM"],
                "intercept": ens.weights["intercept"],
            }
            pd.DataFrame([row]).to_csv(
                OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False
            )
            print(f"store={store} item={item} MAE={metrics['MAE']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            logger.warning(f"Failed store={store} item={item}: {exc}")
