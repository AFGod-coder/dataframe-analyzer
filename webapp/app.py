"""Streamlit web app for demand forecasting and purchase planning.

The app has two tabs. Both use the same mathematical model (the
ensemble, through `ForecastService`):

  1. `Dataset del proyecto`: forecasts for any store and item of the
     Kaggle dataset, with backtest metrics and a future forecast.
  2. `Mis ventas: ¿cuánto comprar?`: the flow for the end user. The user
     uploads a sales history (Excel or CSV, with a fixed template). The
     app checks and cleans the file, shows the products with enough
     history, forecasts the chosen product and recommends how many units
     to buy for a period that ends on a chosen date. The recommendation
     uses the current inventory and a service level.

Run with:
    streamlit run webapp/app.py

Numbered features (keep this list the same as the code):
    Tab 1 - Project dataset
    1.  Store selection
    2.  Product selection
    3.  Forecast horizon selection (days)
    4.  Model selection (ensemble by default; the registry can grow)
    5.  Button to create the forecast
    6.  Interactive chart with history and forecast (Plotly)
    7.  Table with the model error metrics (MAE, RMSE, MAPE)
    8.  Handling of input errors (store and item without data, model errors)
    9.  Export of the forecast to CSV
    10. History of the queries in the current session
    11. Help section that explains the metrics for non-technical readers
    Tab 2 - My sales: how much to buy?
    12. Download of the Excel template with the required structure
    13. Upload of the user's sales history (Excel .xlsx or CSV)
    14. File check and cleaning, with a report of the problems found
    15. Summary of the products found and of the products that can be forecast
    16. Selection of the product and the target date / period to cover
    17. Purchase recommendation (expected demand + safety stock - inventory)
    18. Reliability of the model on the user's data (weekly accuracy)
    19. Export of the recommendation to Excel
    20. Example data to try the flow without preparing a file
"""

import io
import logging
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dataframe_analyzer.data.sales_upload import (
    MIN_HISTORY_DAYS,
    RECOMMENDED_HISTORY_DAYS,
    SalesFileError,
    build_template_excel,
    normalize_sales,
    process_upload,
    project_dataset_to_template,
)
from dataframe_analyzer.service.forecast_service import (
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    ForecastService,
)
from dataframe_analyzer.service.inventory import (
    SERVICE_LEVELS,
    choose_backtest_days,
    recommend_purchase,
    weekly_accuracy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get("DATAFRAME_ANALYZER_DATA_PATH", "data/raw/train.csv")
MAX_HORIZON_DAYS = 180
MODEL_OPTIONS = list(MODEL_REGISTRY.keys())

st.set_page_config(page_title="Demand Forecasting Demo", layout="wide")


# ------------------------------------------------------- shared helpers

@st.cache_resource
def get_service(data_path: str) -> ForecastService:
    """Load the project dataset only once for each Streamlit server process.

    Args:
        data_path (str): Path to the raw dataset.

    Returns:
        ForecastService: The cached service.
    """
    return ForecastService(data_path)


def render_help_section() -> None:
    """Show the help section that explains the metrics in simple words.

    Feature 11.
    """
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


def model_caption(model_name: str) -> str:
    """Get a short note about the selected model and its training time.

    Args:
        model_name (str): Name of the selected model (a key of
            `MODEL_REGISTRY`).

    Returns:
        str: Note in Spanish to show under the model selector.
    """
    if model_name == DEFAULT_MODEL:
        return (
            "Modelo matemático principal: fusiona Prophet, Random Forest, "
            "XGBoost y LSTM con pesos aprendidos. Es el más preciso, pero entrena 4 "
            "modelos: puede tardar 1-3 minutos."
        )
    return "Modelo base individual (insumo del Ensemble). Útil para comparar."


# ------------------------------------------------- tab 1: project dataset

def render_project_tab() -> None:
    """Show tab 1: forecasts on the Kaggle dataset of the project.

    Features 1 to 11.
    """
    st.caption(
        "Pronóstico sobre el dataset del artículo (Kaggle Store Item Demand "
        "Forecasting: 10 tiendas, 50 productos, 2013-2017)."
    )
    try:
        service = get_service(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"No se encontró el dataset en '{DATA_PATH}'. "
            "Descárgalo en data/raw/train.csv o usa la pestaña 'Mis ventas'."
        )
        return

    if "query_history" not in st.session_state:
        st.session_state.query_history = []  # Feature 10

    with st.form("project_form"):
        c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
        store = c1.selectbox("Tienda", service.list_stores())  # Feature 1
        item = c2.selectbox("Producto", service.list_items())  # Feature 2
        horizon_days = c3.slider("Horizonte de pronóstico (días)", 7, MAX_HORIZON_DAYS, 30)  # Feature 3
        model_name = c4.selectbox("Modelo", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))  # Feature 4
        st.caption(model_caption(model_name))
        run_forecast = st.form_submit_button("Generar pronóstico", type="primary")  # Feature 5

    with st.expander("Historial de esta sesión"):  # Feature 10
        if st.session_state.query_history:
            st.dataframe(pd.DataFrame(st.session_state.query_history), hide_index=True)
        else:
            st.caption("Aún no has generado ningún pronóstico.")

    if not run_forecast:
        st.info("Elige tienda, producto, horizonte y modelo, y presiona 'Generar pronóstico'.")
        render_help_section()
        return

    try:
        with st.spinner(f"Entrenando {model_name}..."):
            result = service.get_forecast(
                store=store, item=item, horizon_days=horizon_days, model_name=model_name,
            )
    except ValueError as exc:
        # Feature 8: show input and data errors to the user as a clear
        # message, not as a traceback.
        st.error(f"No fue posible generar el pronóstico: {exc}")
        logger.exception("Forecast failed for store=%s, item=%s", store, item)
        return

    st.session_state.query_history.append(
        {"tienda": store, "producto": item, "horizonte_dias": horizon_days, "modelo": model_name}
    )

    col_chart, col_metrics = st.columns([3, 1])
    with col_chart:
        # Feature 6: interactive chart with history and forecast.
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
            xaxis_title="Fecha", yaxis_title="Ventas", legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.subheader(f"Desempeño del modelo (últimos {DEFAULT_BACKTEST_DAYS} días)")
        # Feature 7: error metrics.
        st.metric("MAE", f"{result.metrics['MAE']:.2f}")
        st.metric("RMSE", f"{result.metrics['RMSE']:.2f}")
        st.metric("MAPE", f"{result.metrics['MAPE']:.1f}%")
        # Feature 9: export the forecast to CSV.
        st.download_button(
            "Descargar pronóstico (CSV)",
            data=result.future_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_store{store}_item{item}.csv",
            mime="text/csv",
        )

    render_help_section()


# --------------------------------------------------- tab 2: user upload flow

@st.cache_data(show_spinner=False)
def cached_process_upload(file_bytes: bytes, filename: str):
    """Check and clean an uploaded file, only once for each different file.

    Feature 14.

    Args:
        file_bytes (bytes): Content of the uploaded file.
        filename (str): Original file name (.xlsx or .csv).

    Returns:
        UploadReport: Clean data, product summary and warnings.

    Raises:
        SalesFileError: If the file cannot be used.
    """
    return process_upload(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_example_upload(data_path: str):
    """Create an example upload from the project dataset.

    Feature 20. It takes 3 products of store 1 and the last 2 years.

    Args:
        data_path (str): Path to the raw project dataset.

    Returns:
        tuple: The `UploadReport` of the example and the example table in
        the template format.
    """
    df_project = pd.read_csv(data_path)
    example = project_dataset_to_template(df_project, store=1, items=[5, 15, 28], last_days=730)
    return normalize_sales(example), example


@st.cache_data(show_spinner=False)
def cached_product_forecast(series: pd.DataFrame, store: str, item: str, model_name: str, backtest_days: int) -> dict:
    """Train the model on one product and forecast `MAX_HORIZON_DAYS` ahead.

    The horizon is always the maximum. So, when the user changes the
    target date, the inventory or the service level, the app only
    calculates the purchase recommendation again (this is fast), and the
    model is not trained again. For every model in the registry, the
    first k days of a long forecast are the same as a k-day forecast
    (Prophet fits a curve and the other models are recursive), so the
    accuracy does not change.

    Args:
        series (pd.DataFrame): Clean daily data with the columns `date`,
            `store`, `item` and `sales`.
        store (str): Store name.
        item (str): Product name.
        model_name (str): Name of the model (a key of `MODEL_REGISTRY`).
        backtest_days (int): Size of the test window in days.

    Returns:
        dict: Keys `history` (full series), `backtest` (real and predicted
        values on the test window), `metrics` (MAE, RMSE, MAPE) and
        `forecast` (future daily forecast).
    """
    service = ForecastService.from_dataframe(series)
    result = service.get_forecast(
        store=store, item=item, horizon_days=MAX_HORIZON_DAYS,
        model_name=model_name, backtest_days=backtest_days,
    )
    backtest = result.history.merge(result.backtest_predictions, on="ds", how="inner")
    return {
        "history": result.history,
        "backtest": backtest,
        "metrics": result.metrics,
        "forecast": result.future_forecast,
    }


def build_report_excel(rec, label: str, model_name: str, metrics: dict, accuracy: dict) -> bytes:
    """Create the Excel report with the recommendation and the daily forecast.

    Feature 19.

    Args:
        rec (PurchaseRecommendation): Purchase recommendation to report.
        label (str): Product and store name shown in the report.
        model_name (str): Name of the model used.
        metrics (dict): Keys `MAE`, `RMSE` and `MAPE` on the test window.
        accuracy (dict): Result of `weekly_accuracy`.

    Returns:
        bytes: An .xlsx file with the sheets `recomendacion` and
        `pronostico_diario`.
    """
    summary = pd.DataFrame({
        "concepto": [
            "Producto", "Modelo", "Periodo cubierto", "Días del periodo",
            "Demanda esperada en el periodo (unidades)",
            f"Demanda esperada el {rec.period_end.date()} (unidades)",
            f"Stock de seguridad (nivel de servicio {rec.service_level:.0%})",
            "Inventario actual (unidades)", "UNIDADES RECOMENDADAS A COMPRAR",
            "Error diario típico del modelo (RMSE)", "MAE en ventana de prueba",
            "MAPE en ventana de prueba (%)", "Precisión semanal ±15% en tus datos (%)",
        ],
        "valor": [
            label, model_name, f"{rec.period_start.date()} a {rec.period_end.date()}", rec.days,
            round(rec.expected_demand, 1), round(rec.demand_on_target_day, 1),
            round(rec.safety_stock, 1), rec.current_inventory, rec.recommended_units,
            round(rec.daily_error, 2), round(metrics["MAE"], 2), round(metrics["MAPE"], 1),
            "N/D" if accuracy["accuracy_pct"] is None else round(accuracy["accuracy_pct"], 1),
        ],
    })
    daily = rec.daily_forecast.rename(columns={"ds": "fecha", "yhat": "demanda_pronosticada"})
    daily["fecha"] = daily["fecha"].dt.strftime("%Y-%m-%d")
    daily["demanda_pronosticada"] = daily["demanda_pronosticada"].round(2)
    daily["demanda_acumulada"] = daily["demanda_pronosticada"].cumsum().round(2)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="recomendacion", index=False)
        daily.to_excel(writer, sheet_name="pronostico_diario", index=False)
        for sheet in writer.sheets.values():
            for column_cells in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(12, width + 2), 60)
    return buffer.getvalue()


def render_file_instructions() -> None:
    """Show step 1: the required file structure and the template download.

    Feature 12.
    """
    st.markdown("#### 1. Prepara tu archivo")
    st.markdown(
        "Sube tu historial de ventas en Excel (.xlsx) o CSV con estas columnas "
        "(una fila por día o una fila por venta; las del mismo día se suman):"
    )
    st.table(pd.DataFrame({
        "columna": ["fecha", "tienda", "producto", "cantidad_vendida"],
        "obligatoria": ["Sí", "No", "Sí", "Sí"],
        "ejemplo": ["2025-01-02", "Centro", "Arroz 1kg", "14"],
    }))
    st.caption(
        f"Mínimo {MIN_HISTORY_DAYS} días de historia por producto; ideal {RECOMMENDED_HISTORY_DAYS} "
        "días o más para que el modelo aprenda la estacionalidad anual. Los días sin registros "
        "se toman como 0 ventas."
    )
    st.download_button(
        "Descargar plantilla Excel",
        data=build_template_excel(),
        file_name="plantilla_ventas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def load_user_data():
    """Show step 2: upload a file or use the example data.

    Features 13, 14 and 20.

    Returns:
        UploadReport | None: The processed data, or None when there is no
        data yet or the file cannot be used.
    """
    st.markdown("#### 2. Sube tus ventas")
    col_up, col_ex = st.columns([3, 1])
    uploaded = col_up.file_uploader("Archivo de ventas", type=["xlsx", "csv"])  # Feature 13
    with col_ex:
        st.write("")
        if st.button("Usar datos de ejemplo", help="3 productos de la Tienda 1 del dataset del proyecto, últimos 2 años"):
            st.session_state.use_example = True
        if st.session_state.get("use_example") and uploaded is None:
            st.caption("Usando datos de ejemplo.")

    try:
        if uploaded is not None:
            st.session_state.use_example = False
            return cached_process_upload(uploaded.getvalue(), uploaded.name)
        if st.session_state.get("use_example"):
            if not os.path.exists(DATA_PATH):
                st.error(f"Los datos de ejemplo requieren el dataset del proyecto en '{DATA_PATH}'.")
                return None
            report, _ = cached_example_upload(DATA_PATH)
            return report
    except SalesFileError as exc:
        # Feature 14: explain to the user the problems that stop the process.
        st.error(f"No pudimos usar tu archivo: {exc}")
        return None
    return None


def render_upload_summary(report) -> None:
    """Show the file check report and the product summary.

    Features 14 and 15.

    Args:
        report (UploadReport): Result of processing the uploaded file.
    """
    data = report.data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas leídas", f"{report.rows_read:,}")
    c2.metric("Productos", f"{report.products.shape[0]}")
    c3.metric("Aptos para pronosticar", f"{int(report.products['apto'].sum())}")
    c4.metric(
        "Rango de fechas", f"{data['date'].min():%Y-%m} → {data['date'].max():%Y-%m}",
        help=f"{data['date'].min():%Y-%m-%d} a {data['date'].max():%Y-%m-%d}",
    )
    for message in report.warnings:
        st.warning(message)
    with st.expander("Productos encontrados", expanded=False):
        shown = report.products.copy()
        shown["fecha_inicio"] = shown["fecha_inicio"].dt.strftime("%Y-%m-%d")
        shown["fecha_fin"] = shown["fecha_fin"].dt.strftime("%Y-%m-%d")
        shown["apto"] = shown["apto"].map({True: "Sí", False: f"No (< {MIN_HISTORY_DAYS} días)"})
        st.dataframe(shown, hide_index=True, use_container_width=True)


def render_recommendation(report) -> None:
    """Show steps 3 to 5: product, model, planning inputs and recommendation.

    Features 16 to 19.

    Args:
        report (UploadReport): Result of processing the uploaded file.
    """
    eligible = report.products[report.products["apto"]]
    labels = {
        f"{row.producto}  —  {row.tienda}": (row.tienda, row.producto)
        for row in eligible.itertuples()
    }

    st.markdown("#### 3. Elige el producto y genera su pronóstico")
    c1, c2 = st.columns([2, 2])
    label = c1.selectbox("Producto", list(labels))  # Feature 16
    model_name = c2.selectbox(
        "Modelo", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL), key="upload_model",
    )
    st.caption(model_caption(model_name))
    store, item = labels[label]
    series = report.data[(report.data["store"] == store) & (report.data["item"] == item)].reset_index(drop=True)
    last_date = series["date"].max()
    backtest_days = choose_backtest_days(len(series))

    key = (label, model_name, len(series), float(series["sales"].sum()))
    if st.button("Generar pronóstico del producto", type="primary"):
        try:
            with st.spinner(f"Entrenando {model_name} con {len(series)} días de historia de '{item}'..."):
                st.session_state.upload_forecast = {
                    "key": key,
                    "result": cached_product_forecast(series, store, item, model_name, backtest_days),
                }
        except Exception as exc:  # Show model errors to the user as a message, not as a traceback.
            logger.exception("Upload forecast failed for %s", label)
            st.error(f"No fue posible entrenar el modelo para este producto: {exc}")
            return

    state = st.session_state.get("upload_forecast")
    if not state or state["key"] != key:
        st.info("Presiona 'Generar pronóstico del producto' para continuar.")
        return
    result = state["result"]

    st.markdown("#### 4. ¿Para qué fecha quieres saber cuánto comprar?")
    first_day = (last_date + pd.Timedelta(days=1)).date()
    last_allowed = (last_date + pd.Timedelta(days=MAX_HORIZON_DAYS)).date()
    c1, c2, c3, c4 = st.columns(4)
    target = c1.date_input(
        "Fecha objetivo", value=(last_date + pd.Timedelta(days=14)).date(),
        min_value=first_day, max_value=last_allowed,
        help=f"Hasta {MAX_HORIZON_DAYS} días después de tu último registro ({last_date:%Y-%m-%d}).",
    )
    start = c2.date_input(
        "Cubrir desde", value=first_day, min_value=first_day, max_value=target,
        help="Primer día que la compra debe cubrir. Por defecto, el día siguiente a tu último registro.",
    )
    inventory = c3.number_input(
        "Inventario disponible al inicio (unidades)", min_value=0.0, value=0.0, step=1.0,
    )
    level_label = c4.select_slider(
        "Nivel de servicio", options=list(SERVICE_LEVELS), value="95%",
        help="Probabilidad deseada de no quedarte sin producto. Más alto = más stock de seguridad.",
    )

    try:
        rec = recommend_purchase(
            forecast=result["forecast"], period_start=start, period_end=target,
            daily_error=result["metrics"]["RMSE"], current_inventory=inventory,
            service_level=SERVICE_LEVELS[level_label],
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    accuracy = weekly_accuracy(result["backtest"]["y"].values, result["backtest"]["yhat"].values)

    st.markdown("#### 5. Recomendación")
    c1, c2, c3, c4 = st.columns(4)  # Feature 17
    c1.metric("Unidades a comprar", f"{rec.recommended_units:,}")
    c2.metric(f"Demanda esperada ({rec.days} días)", f"{rec.expected_demand:,.0f}")
    c3.metric("Stock de seguridad", f"{rec.safety_stock:,.0f}")
    c4.metric(f"Demanda el {rec.period_end:%Y-%m-%d}", f"{rec.demand_on_target_day:,.1f}")
    st.caption(
        f"Cálculo: {rec.expected_demand:,.1f} (demanda esperada del {rec.period_start:%Y-%m-%d} al "
        f"{rec.period_end:%Y-%m-%d}) + {rec.safety_stock:,.1f} (stock de seguridad = {rec.z:.2f} × "
        f"{rec.daily_error:.2f} × √{rec.days}) − {rec.current_inventory:,.0f} (inventario) "
        f"= {rec.recommended_units:,} unidades (redondeado hacia arriba, nunca negativo)."
    )

    fig = go.Figure()
    recent = result["history"][result["history"]["ds"] > last_date - pd.Timedelta(days=180)]
    fig.add_trace(go.Scatter(x=recent["ds"], y=recent["y"], mode="lines", name="Ventas reales",
                             line=dict(color="#4C78A8")))
    fig.add_trace(go.Scatter(x=result["forecast"]["ds"], y=result["forecast"]["yhat"].clip(lower=0),
                             mode="lines", name="Pronóstico", line=dict(color="#F58518", width=2)))
    fig.add_vrect(x0=rec.period_start, x1=rec.period_end, fillcolor="#F58518", opacity=0.12,
                  line_width=0, annotation_text="Periodo a cubrir", annotation_position="top left")
    fig.update_layout(title=f"{item} — {store}", xaxis_title="Fecha", yaxis_title="Unidades",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### ¿Qué tan confiable es este pronóstico para tu producto?")  # Feature 18
    c1, c2, c3, c4 = st.columns(4)
    acc = accuracy["accuracy_pct"]
    c1.metric("Precisión semanal (±15%)", "N/D" if acc is None else f"{acc:.1f}%",
              help=f"{accuracy['weeks_within']} de {accuracy['weeks']} semanas de prueba dentro de ±15%.")
    c2.metric("MAE (unid./día)", f"{result['metrics']['MAE']:.2f}")
    c3.metric("RMSE (unid./día)", f"{result['metrics']['RMSE']:.2f}")
    c4.metric("MAPE", f"{result['metrics']['MAPE']:.1f}%")
    st.caption(
        f"Medido sobre tus últimos {backtest_days} días: el modelo se entrenó sin verlos y luego se "
        "compararon sus predicciones con lo que realmente vendiste. El MAPE puede verse alto en "
        "productos con días de muy pocas ventas; la precisión semanal es más representativa para "
        "decidir compras."
    )
    if acc is not None and acc < 80:
        st.warning(
            "La precisión semanal en tu producto es baja: usa la recomendación como referencia y "
            "considera subir el nivel de servicio o revisar si hubo eventos atípicos en tus datos."
        )

    with st.expander("Ver pronóstico día a día del periodo"):
        daily = rec.daily_forecast.rename(columns={"ds": "fecha", "yhat": "demanda_pronosticada"})
        daily["demanda_acumulada"] = daily["demanda_pronosticada"].cumsum()
        st.dataframe(daily.round(2), hide_index=True, use_container_width=True)

    st.download_button(  # Feature 19
        "Descargar recomendación (Excel)",
        data=build_report_excel(rec, label, model_name, result["metrics"], accuracy),
        file_name=f"recomendacion_compra_{str(item).replace(' ', '_')}_{rec.period_end:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def render_upload_tab() -> None:
    """Show tab 2: the flow for the end user.

    Features 12 to 20.
    """
    st.caption(
        "Sube tu historial de ventas y obtén cuántas unidades comprar de un producto "
        "para cubrir la demanda hasta la fecha que elijas."
    )
    render_file_instructions()
    report = load_user_data()
    if report is None:
        return
    render_upload_summary(report)
    render_recommendation(report)


def main() -> None:
    """Show the full Streamlit page."""
    st.title("Pronóstico de Demanda y Recomendación de Compra")
    tab_upload, tab_project = st.tabs(["Mis ventas: ¿cuánto comprar?", "Dataset del proyecto"])
    with tab_upload:
        render_upload_tab()
    with tab_project:
        render_project_tab()


if __name__ == "__main__":
    main()
