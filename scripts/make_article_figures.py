"""
Publication figures of the article (Figs. 1-7), in Spanish ('es', decimal comma)
or English ('en'). Reads the per-series results produced by
scripts/compute_article_stats.py and results/ref_series_2_28.csv (test-window
predictions of the 5 models for store 2 / item 28, an out-of-sample reference
series: the highest-volume series of the 500).

Usage:
    python3 scripts/make_article_figures.py es
    python3 scripts/make_article_figures.py en
Output: reports/figures/articulo/<lang>/
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.figure
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ART = "results"
REPO = "."

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, GREY, RED = "#2f5d8a", "#d9822b", "#4f8f5b", "#8a8a8a", "#b5473a"

T = {
    "es": dict(
        store="Tienda", units_m="Unidades vendidas (millones)", item="Producto",
        a_store="(a) Ventas totales por tienda, 2013–2017", a_item="(b) Ventas totales por producto (ordenadas)",
        month="Mes", units_month="Unidades por mes (miles)", a_trend="(a) Ventas totales mensuales",
        days=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], avg_day="Unidades por día (promedio)",
        a_week="(b) Promedio por día de la semana",
        train_base="Entrenamiento base\n(modelos base)", val="Validación\n(pesos w)", test="Prueba\n(90 días)",
        stage1="Etapa 1: los 4 modelos base se entrenan solo con el tramo base y predicen la validación (datos no vistos)",
        stage2="Etapa 2: se resuelve min Σ(y − Σ wₖŷₖ)²  s.a. wₖ ≥ 0, Σwₖ = 1  →  pesos congelados",
        stage3="Etapa 3: los modelos base se reentrenan con base + validación; ŷ = Σ wₖŷₖ se evalúa en la prueba",
        time="tiempo →",
        mae_axis="MAE por serie (unidades/día)", naive="Ingenuo\nestacional", ens="Ensemble",
        mae_title="Distribución del MAE en las 51 series (ventana de prueba de 90 días)",
        real="Real", pred="Predicho", date="Fecha", sales="Unidades", resid="Residuo (real − predicho)",
        fit_t="(a) Ensemble: valores reales y predichos", res_t="(b) Ensemble: residuos",
        weight_axis="Peso aprendido wₖ", weights_t="Pesos de fusión aprendidos en las 51 series",
    ),
    "en": dict(
        store="Store", units_m="Units sold (millions)", item="Item",
        a_store="(a) Total sales by store, 2013–2017", a_item="(b) Total sales by item (sorted)",
        month="Month", units_month="Units per month (thousands)", a_trend="(a) Total monthly sales",
        days=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], avg_day="Units per day (mean)",
        a_week="(b) Mean by day of week",
        train_base="Base training\n(base models)", val="Validation\n(weights w)", test="Test\n(90 days)",
        stage1="Stage 1: the 4 base models are trained on the base slice only and predict the validation slice (unseen data)",
        stage2="Stage 2: solve min Σ(y − Σ wₖŷₖ)²  s.t. wₖ ≥ 0, Σwₖ = 1  →  weights frozen",
        stage3="Stage 3: base models are retrained on base + validation; ŷ = Σ wₖŷₖ is evaluated on the test slice",
        time="time →",
        mae_axis="Per-series MAE (units/day)", naive="Seasonal\nnaive", ens="Ensemble",
        mae_title="MAE distribution across the 51 series (90-day test window)",
        real="Actual", pred="Predicted", date="Date", sales="Units", resid="Residual (actual − predicted)",
        fit_t="(a) Ensemble: actual vs. predicted", res_t="(b) Ensemble: residuals",
        weight_axis="Learned weight wₖ", weights_t="Fusion weights learned across the 51 series",
    ),
}


def main(lang):
    t = T[lang]
    from matplotlib.ticker import FuncFormatter
    dec = (lambda v, nd=1: f"{v:.{nd}f}".replace(".", ",")) if lang == "es" else (lambda v, nd=1: f"{v:.{nd}f}")
    if lang == "es":
        plt.rcParams["axes.formatter.use_locale"] = False
        _orig_savefig = matplotlib.figure.Figure.savefig
        def _savefig(self, *a, **k):
            for ax_ in self.axes:
                for axis in (ax_.yaxis, ax_.xaxis):
                    fmt = axis.get_major_formatter()
                    if type(fmt).__name__ == "ScalarFormatter":
                        axis.set_major_formatter(FuncFormatter(lambda v, _p: (f"{v:g}").replace(".", ",")))
            return _orig_savefig(self, *a, **k)
        matplotlib.figure.Figure.savefig = _savefig
    out = f"reports/figures/articulo/{lang}"
    import os
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(f"{REPO}/data/raw/train.csv", parse_dates=["date"])

    # Fig 1: store and item totals
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.6]})
    st = df.groupby("store").sales.sum() / 1e6
    ax[0].bar(st.index.astype(str), st.values, color=BLUE, width=0.7)
    ax[0].set(xlabel=t["store"], ylabel=t["units_m"], title=t["a_store"])
    it = (df.groupby("item").sales.sum() / 1e6).sort_values(ascending=False)
    ax[1].bar(range(len(it)), it.values, color=BLUE, width=0.75)
    ax[1].set_xticks(range(len(it)))
    ax[1].set_xticklabels(it.index.astype(str), rotation=90, fontsize=5.5)
    ax[1].set(xlabel=t["item"], ylabel=t["units_m"], title=t["a_item"])
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_tiendas_productos.png"); plt.close(fig)

    # Fig 2: monthly trend + weekday
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})
    mo = df.set_index("date").sales.resample("MS").sum() / 1e3
    ax[0].plot(mo.index, mo.values, color=BLUE, lw=1.4, marker="o", ms=2.2)
    ax[0].set(xlabel=t["month"], ylabel=t["units_month"], title=t["a_trend"])
    wd = df.groupby(df.date.dt.dayofweek).sales.mean()
    colors = [BLUE] * 5 + [ORANGE] * 2
    ax[1].bar(t["days"], wd.values, color=colors, width=0.7)
    for i, v in enumerate(wd.values):
        ax[1].text(i, v + 0.8, dec(v), ha="center", fontsize=7)
    ax[1].set(ylabel=t["avg_day"], title=t["a_week"], ylim=(0, 70))
    fig.tight_layout(); fig.savefig(f"{out}/fig2_tendencia_semana.png"); plt.close(fig)

    # Fig 3: walk-forward scheme
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    segs = [(2, 60, BLUE, t["train_base"]), (62, 18, ORANGE, t["val"]), (80, 18, GREEN, t["test"])]
    for x, w, c, lab in segs:
        ax.add_patch(FancyBboxPatch((x, 27), w - 1, 9, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=c, ec="none", alpha=0.9))
        ax.text(x + (w - 1) / 2, 31.5, lab, ha="center", va="center", color="white", fontsize=8.5, weight="bold")
    ax.annotate("", xy=(99, 24.5), xytext=(2, 24.5), arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.text(99, 22, t["time"], ha="right", va="top", fontsize=7.5, color=GREY)
    for y, txt in zip([16, 10, 4], [t["stage1"], t["stage2"], t["stage3"]]):
        ax.text(2, y, txt, fontsize=7.6, va="center")
    fig.savefig(f"{out}/fig3_esquema_walk_forward.png"); plt.close(fig)

    # Fig 4: MAE distribution boxplot
    allm = pd.read_csv(f"{ART}/all_models_per_series.csv")
    order = ["Ensemble (Fusionado)", "Prophet", "Random Forest", "XGBoost", "LSTM", "Ingenuo estacional (t-7)"]
    labels = [t["ens"], "Prophet", "Random\nForest", "XGBoost", "LSTM", t["naive"]]
    data = [allm[allm.model == m].MAE.values for m in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bp = ax.boxplot(data, tick_labels=labels, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", ms=4),
                    medianprops=dict(color="black", lw=1.1), flierprops=dict(ms=3, markeredgecolor=GREY))
    for patch, c in zip(bp["boxes"], [ORANGE, BLUE, BLUE, BLUE, BLUE, GREY]):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=5, color="black", alpha=0.35, zorder=3)
    ax.set(ylabel=t["mae_axis"], title=t["mae_title"])
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig4_distribucion_mae.png"); plt.close(fig)

    # Fig 5: ensemble fit and residuals on reference series
    ref = pd.read_csv(f"{ART}/ref_series_2_28.csv", parse_dates=["ds"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(ref.ds, ref.y, color=BLUE, lw=1.1, label=t["real"])
    ax[0].plot(ref.ds, ref.Ensemble, color=ORANGE, lw=1.1, ls="--", label=t["pred"])
    ax[0].set(xlabel=t["date"], ylabel=t["sales"], title=t["fit_t"]); ax[0].legend(frameon=False, loc="upper right")
    resid = ref.y - ref.Ensemble
    ax[1].scatter(ref.ds, resid, s=8, color=GREEN, alpha=0.8)
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set(xlabel=t["date"], ylabel=t["resid"], title=t["res_t"])
    import matplotlib.dates as mdates
    for a in ax:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(f"{out}/fig5_ensemble_ajuste.png"); plt.close(fig)

    # Fig 6: base models fit (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for a, (name, letter) in zip(axs.flat, [("Prophet", "a"), ("Random Forest", "b"), ("XGBoost", "c"), ("LSTM", "d")]):
        mae = (ref.y - ref[name]).abs().mean()
        a.plot(ref.ds, ref.y, color=BLUE, lw=0.9, label=t["real"])
        a.plot(ref.ds, ref[name], color=ORANGE, lw=0.9, ls="--", label=t["pred"])
        a.set_title(f"({letter}) {name}  —  MAE = {dec(mae, 2)}")
        import matplotlib.dates as mdates
        a.xaxis.set_major_locator(mdates.MonthLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.tick_params(axis="x", rotation=30)
    axs[0, 0].legend(frameon=False, loc="upper right")
    for a in axs[:, 0]:
        a.set_ylabel(t["sales"])
    for a in axs[1, :]:
        a.set_xlabel(t["date"])
    fig.tight_layout(); fig.savefig(f"{out}/fig6_modelos_base.png"); plt.close(fig)

    # Fig 7: learned weights
    ens = pd.read_csv(f"{REPO}/results/ensemble_metrics.csv")
    cols = [("w_prophet", "Prophet"), ("w_xgb", "XGBoost"), ("w_rf", "Random Forest"), ("w_lstm", "LSTM")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    data = [ens[c].values for c, _ in cols]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=[n for _, n in cols], widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", ms=4),
                    medianprops=dict(color="black", lw=1.1), flierprops=dict(ms=3, markeredgecolor=GREY))
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE); patch.set_alpha(0.5)
    rng = np.random.default_rng(1)
    for i, d in enumerate(data, start=1):
        ax.scatter(d, i + rng.uniform(-0.12, 0.12, len(d)), s=5, color="black", alpha=0.35, zorder=3)
    ax.invert_yaxis()
    ax.set(xlabel=t["weight_axis"], title=t["weights_t"], xlim=(-0.03, 1.03))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(f"{out}/fig7_pesos.png"); plt.close(fig)
    print("ok", lang)


if __name__ == "__main__":
    main(sys.argv[1])
