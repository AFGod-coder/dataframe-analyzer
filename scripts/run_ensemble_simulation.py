"""Backtest of the ensemble model on the stratified sample of 51 series.

The script uses the same 51 series and the same 90-day test window as
the base models in `results/unified_metrics.csv`, so the results can be
compared directly.

The script can continue after a stop. The result of each series is
saved to disk at once, and a new run skips the (store, item) pairs that
are already in the output file. This is useful because each series
takes about 50-90 seconds (it trains 2 copies of the 4 base models).

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
    """Select a stratified sample of (store, item) pairs.

    This is the same selection used for the other models (same fraction
    and seed), so all models are evaluated on the same 51 series.

    Args:
        df (pd.DataFrame): Full raw dataset with the columns `date`,
            `store`, `item` and `sales`.
        fraction (float): Fraction of the 500 series to keep from each
            sales group (low, medium and high).
        seed (int): Random seed for reproducible results.

    Returns:
        list: Selected (store, item) pairs.
    """
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
