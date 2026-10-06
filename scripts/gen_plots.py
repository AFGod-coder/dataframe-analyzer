import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataframe_analyzer.service.forecast_service import ForecastService

STORE, ITEM = 2, 28  # alta rotacion (Figura 1/2 del EDA), buena senal visual
svc = ForecastService("data/raw/train.csv")
OUT = "reports/figures/modelos"

for model in ["Prophet", "Random Forest", "XGBoost", "LSTM"]:
    result = svc.get_forecast(store=STORE, item=ITEM, horizon_days=14, model_name=model, backtest_days=90)
    merged = result.history.merge(result.backtest_predictions, on="ds")
    y_true = merged["y"].values
    y_pred = merged["yhat"].values
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(merged["ds"], y_true, label="Real", color="#4C78A8", linewidth=1.3)
    axes[0].plot(merged["ds"], y_pred, label="Predicho", color="#F58518", linewidth=1.3, linestyle="--")
    axes[0].set_title(f"{model} - Ajuste (backtest 90 dias)\nTienda {STORE}, Producto {ITEM}")
    axes[0].set_xlabel("Fecha"); axes[0].set_ylabel("Ventas")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].scatter(merged["ds"], residuals, s=14, color="#54A24B", alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{model} - Residuales (real - predicho)")
    axes[1].set_xlabel("Fecha"); axes[1].set_ylabel("Residual")
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fname = model.lower().replace(" ", "_")
    plt.savefig(f"{OUT}/{fname}_ajuste_residuales.png", dpi=130)
    plt.close()
    print(f"{model}: MAE={result.metrics['MAE']:.2f} guardado en {OUT}/{fname}_ajuste_residuales.png")
