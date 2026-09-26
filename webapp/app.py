"""
Demo web app for section 2.5 (Application Implementation).

Architecture: input (store + product + horizon) -> processing (best
model from section 2.3, served through ForecastService) -> output
(forecast chart + error metrics).

Run with:
    streamlit run webapp/app.py

Numbered functionalities (for the rubric's "at least 10 distinguishable
functionalities" requirement -- keep this list in sync with the code):
    1.  Selección de tienda
    2.  Selección de producto
    3.  Selección del horizonte de pronóstico (días)
    4.  Selección de modelo (registro extensible, ver ForecastService)
    5.  Botón para generar el pronóstico
    6.  Gráfico de histórico + pronóstico (Plotly, interactivo)
    7.  Tabla de métricas de error del modelo (MAE, RMSE, MAPE)
    8.  Manejo de errores de entrada (combinación sin datos, fallos del modelo)
    9.  Exportar el resultado del pronóstico a CSV
    10. Historial de consultas realizadas en la sesión actual
    11. Sección de ayuda que interpreta las métricas para un lector no técnico
"""

import logging
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataframe_analyzer.service.forecast_service import (
    DEFAULT_BACKTEST_DAYS,
    MODEL_REGISTRY,
    ForecastService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get("DATAFRAME_ANALYZER_DATA_PATH", "data/raw/train.csv")

st.set_page_config(page_title="Demand Forecasting Demo", layout="wide")


@st.cache_resource
def get_service(data_path: str) -> ForecastService:
    """Load the dataset once per Streamlit server process (functionality 8 support).

    Args:
        data_path (str): Path to the processed dataset.

    Returns:
        ForecastService: the cached service instance.
    """
    return ForecastService(data_path)


def render_help_section() -> None:
    """Functionality 11: explain the metrics in plain language."""
    with st.expander("¿Cómo interpretar estos resultados?"):
        st.markdown(
            """
            - **MAE (Error Absoluto Medio):** en promedio, cuántas unidades
              se equivoca el modelo por día (en la misma unidad que las ventas).
            - **RMSE (Raíz del Error Cuadrático Medio):** similar al MAE, pero
              penaliza más los errores grandes/atípicos.
            - **MAPE (Error Porcentual Absoluto Medio):** el error anterior
              expresado como porcentaje, para comparar entre productos con
              volúmenes de venta muy distintos.

            Estas métricas se calculan sobre una ventana de prueba (los
            últimos días conocidos), no sobre el futuro: es la forma de
            estimar qué tan confiable es el pronóstico antes de usarlo.
            """
        )


def main() -> None:
    """Entry point: renders the full Streamlit page."""
    st.title("Demo: Pronóstico de Demanda por Tienda y Producto")
    st.caption(
        "Sección 2.5 - Application Implementation. "
        "Consume el pipeline y el modelo de forecasting del proyecto."
    )

    try:
        service = get_service(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"No se encontró el dataset procesado en '{DATA_PATH}'. "
            f"Corre primero: python scripts/run_pipeline.py"
        )
        return

    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # functionality 10

    with st.sidebar:
        st.header("Parámetros del pronóstico")
        store = st.selectbox("Tienda", service.list_stores())  # functionality 1
        item = st.selectbox("Producto", service.list_items())  # functionality 2
        horizon_days = st.slider("Horizonte de pronóstico (días)", 7, 180, 30)  # functionality 3
        model_name = st.selectbox("Modelo", list(MODEL_REGISTRY.keys()))  # functionality 4
        run_forecast = st.button("Generar pronóstico", type="primary")  # functionality 5

        st.divider()
        st.subheader("Historial de esta sesión")  # functionality 10
        if st.session_state.query_history:
            st.dataframe(pd.DataFrame(st.session_state.query_history), hide_index=True)
        else:
            st.caption("Aún no has generado ningún pronóstico.")

    if not run_forecast:
        st.info("Elige tienda, producto y horizonte, y presiona 'Generar pronóstico'.")
        render_help_section()
        return

    try:
        result = service.get_forecast(
            store=store,
            item=item,
            horizon_days=horizon_days,
            model_name=model_name,
        )
    except ValueError as exc:
        # functionality 8: input/data error handling, shown to the user
        # instead of a raw traceback.
        st.error(f"No fue posible generar el pronóstico: {exc}")
        logger.exception("Forecast failed for store=%s, item=%s", store, item)
        return

    st.session_state.query_history.append(
        {"tienda": store, "producto": item, "horizonte_dias": horizon_days, "modelo": model_name}
    )

    col_chart, col_metrics = st.columns([3, 1])

    with col_chart:
        # functionality 6: interactive history + forecast chart.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.history["ds"], y=result.history["y"],
            mode="lines", name="Histórico", line=dict(color="#4C78A8"),
        ))
        fig.add_trace(go.Scatter(
            x=result.future_forecast["ds"], y=result.future_forecast["yhat"],
            mode="lines", name=f"Pronóstico ({result.model_name})", line=dict(color="#F58518", dash="dash"),
        ))
        fig.update_layout(
            title=f"Ventas - Tienda {store}, Producto {item}",
            xaxis_title="Fecha", yaxis_title="Ventas",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.subheader(f"Desempeño del modelo (últimos {DEFAULT_BACKTEST_DAYS} días)")
        # functionality 7: error metrics table.
        st.metric("MAE", f"{result.metrics['MAE']:.2f}")
        st.metric("RMSE", f"{result.metrics['RMSE']:.2f}")
        st.metric("MAPE", f"{result.metrics['MAPE']:.1f}%")

        # functionality 9: export the forecast as CSV.
        st.download_button(
            "Descargar pronóstico (CSV)",
            data=result.future_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_store{store}_item{item}.csv",
            mime="text/csv",
        )

    render_help_section()


if __name__ == "__main__":
    main()
