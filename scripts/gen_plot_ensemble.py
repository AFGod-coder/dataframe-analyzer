"""
Generate the ajuste/residuales diagnostic figure for the Ensemble
(Fusionado) model, same reference series (Tienda 2, Producto 28) and same
visual style as the 4 base-model figures in reports/figures/modelos/,
so section 2.4 can show it alongside them.

Uses EnsembleForecaster directly (one fit/predict cycle on the 90-day
backtest window) instead of ForecastService.get_forecast() (which also
computes the 14-day future forecast, doubling the compute for no benefit
to this plot).
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.ensemble_model import EnsembleForecaster

STORE, ITEM = 2, 28
OUT = "reports/figures/modelos"

df = DataLoader("data/raw/train.csv").load()
ens = EnsembleForecaster()
series = ens.prepare_series(df, store=STORE, item=ITEM)
train, test, _cutoff = ens.split_train_test(series, test_days=90)
ens.fit(train)
preds = ens.predict(test[["ds"]])
metrics = ens.compute_metrics(test["y"].values, preds["yhat"].values)

merged = test.merge(preds, on="ds")
y_true = merged["y"].values
y_pred = merged["yhat"].values
residuals = y_true - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(merged["ds"], y_true, label="Real", color="#4C78A8", linewidth=1.3)
axes[0].plot(merged["ds"], y_pred, label="Predicho", color="#F58518", linewidth=1.3, linestyle="--")
axes[0].set_title(f"Ensemble (Fusionado) - Ajuste (backtest 90 dias)\nTienda {STORE}, Producto {ITEM}")
axes[0].set_xlabel("Fecha"); axes[0].set_ylabel("Ventas")
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)

axes[1].scatter(merged["ds"], residuals, s=14, color="#54A24B", alpha=0.7)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Ensemble (Fusionado) - Residuales (real - predicho)")
axes[1].set_xlabel("Fecha"); axes[1].set_ylabel("Residual")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
import os
os.makedirs(OUT, exist_ok=True)
plt.savefig(f"{OUT}/ensemble_ajuste_residuales.png", dpi=130)
plt.close()
print(f"Ensemble: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE']:.2f}")
print(f"Pesos: {ens.weights}")
print(f"Guardado en {OUT}/ensemble_ajuste_residuales.png")
