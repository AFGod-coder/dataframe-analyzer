"""
Computes every number reported in the article's Results section from the
per-series result files, so each table can be regenerated from the repo.

Inputs:
    results/unified_metrics.csv     (4 base models; scripts/run_unified_simulation.py)
    results/ensemble_metrics.csv    (Ensemble + learned weights; scripts/run_ensemble_simulation.py)
    results/ensemble_predictions.csv(daily Ensemble predictions; scripts/run_ensemble_predictions.py)
    data/raw/train.csv              (for the seasonal-naive baseline and the EDA numbers)

Outputs:
    results/final_comparison.csv       5 models x 51 series (Table 4 / Fig. 4)
    results/all_models_per_series.csv  same + seasonal-naive baseline y_{t-7}
    results/article_stats.json         Tables 2, 4, 5, 6 and the weight / R2 / EDA figures

Usage:
    python3 scripts/compute_article_stats.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

TEST_DAYS = 90
BASE = ["Prophet", "Random Forest", "XGBoost", "LSTM"]
ENS = "Ensemble (Fusionado)"
NAIVE = "Ingenuo estacional (t-7)"


def naive_metrics(df, pairs):
    rows, r2 = [], []
    for store, item in pairs:
        x = df[(df.store == store) & (df.item == item)].sort_values("date").set_index("date")["sales"].astype(float)
        test = x[x.index > x.index.max() - pd.Timedelta(days=TEST_DAYS)]
        y, yh = test.values, x.shift(7).loc[test.index].values
        rows.append({"model": NAIVE, "store": store, "item": item,
                     "MAE": np.mean(np.abs(y - yh)), "RMSE": np.sqrt(np.mean((y - yh) ** 2)),
                     "MAPE": np.mean(np.abs((y - yh) / np.where(y == 0, 1, y))) * 100})
        r2.append(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    return pd.DataFrame(rows), r2


def accuracy(preds, tol, weekly):
    within = total = 0
    for _, g in preds.groupby(["store", "item"]):
        g = g.sort_values("ds")
        y, yh = g.y.values, np.clip(g.yhat.values, 0, None)
        if weekly:  # consecutive 7-day blocks from the start of the test window
            n = len(y) // 7
            y, yh = y[: n * 7].reshape(n, 7).sum(1), yh[: n * 7].reshape(n, 7).sum(1)
        within += int((np.abs(y - yh) <= tol * y).sum())
        total += len(y)
    return {"within": within, "total": total, "pct": 100 * within / total}


def main():
    df = pd.read_csv("data/raw/train.csv", parse_dates=["date"])
    base = pd.read_csv("results/unified_metrics.csv")
    base = base[base.model.isin(BASE)][["model", "store", "item", "MAE", "RMSE", "MAPE"]]
    ens = pd.read_csv("results/ensemble_metrics.csv")
    final = pd.concat([base, ens[["model", "store", "item", "MAE", "RMSE", "MAPE"]]], ignore_index=True)
    final.to_csv("results/final_comparison.csv", index=False)

    pairs = list(ens[["store", "item"]].itertuples(index=False, name=None))
    naive, r2_naive = naive_metrics(df, pairs)
    allm = pd.concat([final, naive], ignore_index=True)
    allm.to_csv("results/all_models_per_series.csv", index=False)

    out = {}
    summ = allm.groupby("model")[["MAE", "RMSE", "MAPE"]].agg(["mean", "median", "std"])
    out["table4_summary"] = {m: {k: {s: float(summ.loc[m, (k, s)]) for s in ["mean", "median", "std"]}
                                 for k in ["MAE", "RMSE", "MAPE"]} for m in summ.index}
    piv = allm.pivot_table(index=["store", "item"], columns="model", values="MAE")
    out["table4_best_counts"] = piv[[ENS] + BASE].idxmin(axis=1).value_counts().to_dict()
    out["table5_wilcoxon"] = {}
    for m in BASE + [NAIVE]:
        w, p = stats.wilcoxon(piv[ENS], piv[m])
        d = piv[ENS] - piv[m]
        out["table5_wilcoxon"][m] = {"W": float(w), "p": float(p), "mean_diff": float(d.mean()),
                                     "ensemble_wins": int((d < 0).sum())}

    preds = pd.read_csv("results/ensemble_predictions.csv", parse_dates=["ds"])
    r2 = [1 - np.sum((g.y - g.yhat) ** 2) / np.sum((g.y - g.y.mean()) ** 2) for _, g in preds.groupby(["store", "item"])]
    out["r2"] = {"ensemble_mean": float(np.mean(r2)), "ensemble_median": float(np.median(r2)),
                 "ensemble_min": float(np.min(r2)), "ensemble_max": float(np.max(r2)),
                 "naive_mean": float(np.mean(r2_naive))}
    out["table6_accuracy"] = {f"daily_{t}": accuracy(preds, t, False) for t in (0.05, 0.30)}
    out["table6_accuracy"].update({f"weekly_{t}": accuracy(preds, t, True) for t in (0.05, 0.10, 0.15, 0.20)})

    w = ens[["w_prophet", "w_rf", "w_xgb", "w_lstm"]]
    out["weights"] = {c: {"mean": float(w[c].mean()), "median": float(w[c].median()), "min": float(w[c].min()),
                          "max": float(w[c].max()), "n_below_0.001": int((w[c] < 1e-3).sum())} for c in w}
    rerun = preds.groupby(["store", "item"]).apply(lambda g: np.mean(np.abs(g.y - g.yhat)), include_groups=False)
    m = ens.set_index(["store", "item"]).MAE
    out["reproducibility_identical_series"] = int((np.abs(rerun - m.loc[rerun.index]) < 1e-9).sum())

    by_store = df.groupby("store").sales.sum(); by_item = df.groupby("item").sales.sum()
    out["table2_eda"] = {
        "rows": len(df), "days_per_series": sorted(df.groupby(["store", "item"]).size().unique().tolist()),
        "nulls": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
        "total_units": int(df.sales.sum()), "mean": float(df.sales.mean()), "median": float(df.sales.median()),
        "std": float(df.sales.std()), "skew": float(df.sales.skew()),
        "store_max": [int(by_store.idxmax()), int(by_store.max())], "store_min": [int(by_store.idxmin()), int(by_store.min())],
        "item_max": [int(by_item.idxmax()), int(by_item.max())], "item_min": [int(by_item.idxmin()), int(by_item.min())],
        "year_totals": {int(k): int(v) for k, v in df.groupby(df.date.dt.year).sales.sum().items()},
        "month_mean": [round(v, 1) for v in df.groupby(df.date.dt.month).sales.mean()],
        "weekday_mean": [round(v, 1) for v in df.groupby(df.date.dt.dayofweek).sales.mean()],
    }
    json.dump(out, open("results/article_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ["table4_best_counts", "r2", "reproducibility_identical_series"]}, indent=1))


if __name__ == "__main__":
    main()
