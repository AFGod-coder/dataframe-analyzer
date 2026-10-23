"""Save the daily backtest predictions of the ensemble model.

The script saves the prediction of each day (not only MAE, RMSE and
MAPE) on the same stratified sample of 51 series. With these values we
can calculate the weekly accuracy of the ensemble, which is the default
model of the app.

The script can continue after a stop: it skips the (store, item) pairs
that are already in the output file.

Usage:
    python3 scripts/run_ensemble_predictions.py
    SERIES_START=10 SERIES_END=15 python3 scripts/run_ensemble_predictions.py
"""
import os
import time

import pandas as pd

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.ensemble_model import EnsembleForecaster

TEST_DAYS = 90
OUTPUT_PATH = "results/ensemble_predictions.csv"


def select_representative_sample(df, fraction, seed):
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
        done_keys = set(map(tuple, done_df[["store", "item"]].drop_duplicates().values.tolist()))
    else:
        done_keys = set()

    for store, item in chunk_pairs:
        if (store, item) in done_keys:
            continue
        t0 = time.time()
        ens = EnsembleForecaster()
        series = ens.prepare_series(df, store=store, item=item)
        train, test, _cutoff = ens.split_train_test(series, test_days=TEST_DAYS)
        ens.fit(train)
        preds = ens.predict(test[["ds"]])
        merged = test.merge(preds, on="ds")
        merged["store"] = store
        merged["item"] = item
        out = merged[["store", "item", "ds", "y", "yhat"]]
        out.to_csv(OUTPUT_PATH, mode="a", header=not os.path.exists(OUTPUT_PATH), index=False)
        print(f"store={store} item={item} rows={len(out)} ({time.time()-t0:.0f}s)", flush=True)
