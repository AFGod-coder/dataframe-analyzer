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
│       │   └── transform.py      # DataTransform, DatePart (día/mes/año, día de la semana, fin de semana; festivo disponible pero opt-in)
│       ├── models/               # Modelos de forecasting
│       │   └── prophet_model.py  # ProphetForecaster (split cronológico, entrenamiento, métricas)
│       ├── visualization/       # Gráficas y utilidades de EDA
│       │   ├── chart_strategy.py
│       │   └── eda_utils.py      # EdaPlotter: histograma, boxplot, barras y línea, con guardado a PNG
│       └── reporting/           # Generación de reportes
│           └── report_generator.py
├── scripts/                    # Puntos de entrada ejecutables
│   ├── run_pipeline.py          # Corre el pipeline principal (antes main.py)
│   ├── eda_analysis.py          # EDA completo: balance del dataset + 4 gráficos (2.1)
│   ├── run_prophet_simulation.py # Simulación con Prophet para una combinación tienda-producto
│   └── explore_customer_data.py # EDA sobre customer_shopping_data.csv
├── data/
│   ├── raw/                     # Datasets de entrada (train.csv, customer_shopping_data.csv)
│   └── processed/               # Salidas generadas (no se versiona)
├── reports/
│   └── figures/eda/             # Gráficos PNG generados por eda_analysis.py (para el artículo)
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

Esto carga `data/raw/train.csv`, limpia los datos y genera `data/processed/clean_dataframe.csv` con las siguientes columnas derivadas de `date`:

- `day_of_month`, `month`, `year`
- `day_of_week` (0=lunes...6=domingo) y `day_of_week_name`
- `is_weekend` (booleano)

`DataTransform.add_holiday_flag(df, 'date', country=...)` también existe, pero **no se llama por defecto**: el dataset (Kaggle "Store Item Demand Forecasting Challenge") no especifica el país/región de las 10 tiendas, así que asumir un calendario de festivos (EE. UU., Colombia, etc.) sería un supuesto no verificable. Si en algún momento se confirma la ubicación real, se puede llamar explícitamente con el código de país correspondiente.

### Análisis exploratorio (EDA)

Una vez generado `clean_dataframe.csv`, corre el EDA completo del punto 2.1 (Dataset Construction):

```bash
python scripts/eda_analysis.py
```

Esto verifica que el dataset esté balanceado (mismas filas por cada combinación tienda-producto) y genera 4 gráficos en `reports/figures/eda/`:

- `sales_by_store.png` — ventas totales por tienda (barras verticales)
- `sales_by_product.png` — ventas totales por producto (barras horizontales, por ser 50 categorías)
- `sales_trend_monthly.png` — tendencia y estacionalidad mensual (línea)
- `sales_by_weekday.png` — ventas promedio por día de la semana

Las gráficas se generan con la clase `EdaPlotter` (`src/dataframe_analyzer/visualization/eda_utils.py`), que centraliza histogramas, boxplots, barras y líneas con un estilo consistente y guardado opcional a PNG (`save_path`).

### Simulación con Prophet

```bash
python scripts/run_prophet_simulation.py
```

Entrena un modelo Prophet para una combinación tienda-producto (configurable en el script), con split cronológico train/test y métricas MAE, RMSE y MAPE.

Para explorar el dataset de `customer_shopping_data.csv`:

```bash
python scripts/explore_customer_data.py
```

## Estado del proyecto

Este proyecto se evalúa mediante 3 hitos:

- **Hito 1** — Propuesta y Estado del Arte
- **Hito 2** — Modelo, Dataset y Demo
- **Hito 3** — Artículo Científico y Sustentación

## Contribuir

Antes de hacer cambios, revisa la [guía de contribución](./CONTRIBUTING.md) para convenciones de commits, ramas y pull requests.
