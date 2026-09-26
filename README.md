# dataframe-analyzer

Proyecto final de **Ciencias de la Computación II** (Electiva en Ingeniería Aplicada II / Data Science) — Grupo 6.

**Demand Forecasting to Support Inventory Management in Retail Stores**

Analiza el historial de ventas de un comerciante para construir un modelo que prediga la cantidad de producto que debería adquirir, apoyando así la gestión de inventario en tiendas retail.

## Estructura del proyecto

```
.
├── src/
│   └── dataframe_analyzer/     # Paquete principal (código fuente)
│       ├── pipeline.py          # Orquestador del flujo completo
│       ├── data/                # Carga, limpieza y transformación
│       │   ├── loader.py         # DataLoader
│       │   ├── cleaner.py        # DataCleaner
│       │   └── transform.py      # DataTransform, DatePart
│       ├── visualization/       # Gráficas y utilidades de EDA
│       │   ├── chart_strategy.py
│       │   └── eda_utils.py
│       ├── models/               # Modelos de forecasting
│       │   └── prophet_model.py  # ProphetForecaster
│       ├── service/              # Capa entre la app y los modelos
│       │   └── forecast_service.py # ForecastService (registro de modelos, backtest + pronóstico futuro)
│       └── reporting/           # Generación de reportes
│           └── report_generator.py
├── scripts/                    # Puntos de entrada ejecutables
│   ├── run_pipeline.py          # Corre el pipeline principal (antes main.py)
│   ├── run_prophet_simulation.py # Simulación con Prophet para una combinación tienda-producto
│   └── explore_customer_data.py # EDA sobre customer_shopping_data.csv
├── webapp/
│   └── app.py                   # Demo web (Streamlit) — sección 2.5 Application Implementation
├── data/
│   ├── raw/                     # Datasets de entrada (train.csv, customer_shopping_data.csv)
│   └── processed/               # Salidas generadas (no se versiona)
├── tests/                      # Pruebas (pendiente)
├── pyproject.toml              # Configuración del paquete instalable
└── requirements.txt             # Dependencias del proyecto
```

## Requisitos

- Python 3.10+

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` instala el paquete `dataframe_analyzer` en modo editable, así los scripts pueden importarlo directamente sin configurar `PYTHONPATH`.

## Uso

Desde la raíz del proyecto:

```bash
python scripts/run_pipeline.py
```

Esto carga `data/raw/train.csv`, limpia los datos y genera `data/processed/clean_dataframe.csv` con las columnas de fecha descompuestas (día, mes, año).

Para explorar el dataset de `customer_shopping_data.csv`:

```bash
python scripts/explore_customer_data.py
```

### Demo web (sección 2.5 - Application Implementation)

```bash
streamlit run webapp/app.py
```

Abre una página local donde eliges tienda, producto y horizonte de pronóstico, y ves el resultado gráfico junto con las métricas de error (MAE, RMSE, MAPE). Por ahora usa `ProphetForecaster` (el único modelo listo); cuando la sección 2.3 defina cuál de los 4 modelos comparados gana, basta con registrarlo en `MODEL_REGISTRY` (`src/dataframe_analyzer/service/forecast_service.py`) para que aparezca en el selector de la app, sin tocar el resto del código.

La app lee `data/raw/train.csv` directamente (no el CSV ya procesado por el pipeline), porque Prophet necesita la columna `date` original y el pipeline la descompone/elimina.

## Estado del proyecto

Este proyecto se evalúa mediante 3 hitos:

- **Hito 1** — Propuesta y Estado del Arte
- **Hito 2** — Modelo, Dataset y Demo
- **Hito 3** — Artículo Científico y Sustentación

## Contribuir

Antes de hacer cambios, revisa la [guía de contribución](./CONTRIBUTING.md) para convenciones de commits, ramas y pull requests.
