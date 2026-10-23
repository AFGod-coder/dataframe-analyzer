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
│       ├── data/                # Carga, limpieza, transformación y carga de archivos del usuario
│       │   ├── loader.py         # DataLoader
│       │   ├── cleaner.py        # DataCleaner
│       │   ├── transform.py      # DataTransform, DatePart (día/mes/año, día de la semana, fin de semana; festivo opcional)
│       │   └── sales_upload.py   # Validación y normalización del Excel/CSV de ventas del usuario
│       ├── models/               # Modelos de pronóstico
│       │   ├── prophet_model.py
│       │   ├── random_forest_model.py
│       │   ├── xgboost_model.py
│       │   ├── lstm_model.py
│       │   └── ensemble_model.py # Ensemble (Fusionado): combinación convexa de los 4 modelos base
│       ├── service/              # Capa entre la app y los modelos
│       │   ├── forecast_service.py # ForecastService y MODEL_REGISTRY
│       │   ├── adapters.py       # Interfaz común para Random Forest, XGBoost y LSTM
│       │   └── inventory.py      # Recomendación de compra (stock de seguridad)
│       ├── visualization/       # Gráficas y utilidades de EDA
│       │   ├── chart_strategy.py
│       │   └── eda_utils.py      # EdaPlotter: histograma, boxplot, barras y línea, con guardado a PNG
│       └── reporting/           # Generación de reportes
│           └── report_generator.py
├── scripts/                    # Puntos de entrada ejecutables
│   ├── run_pipeline.py          # Pipeline principal (carga, limpieza, transformación)
│   ├── eda_analysis.py          # EDA: balance del dataset + 4 gráficos
│   ├── run_unified_simulation.py  # Backtest de los 4 modelos base en las 51 series
│   ├── run_ensemble_simulation.py # Backtest del Ensemble en las mismas 51 series
│   ├── run_ensemble_predictions.py # Predicciones diarias del Ensemble (precisión semanal)
│   ├── compute_article_stats.py # Tablas y estadísticas de resultados
│   └── make_article_figures.py  # Figuras del artículo (es/en)
├── webapp/
│   └── app.py                   # Aplicación web (Streamlit)
├── data/
│   ├── raw/                     # Dataset de entrada (train.csv)
│   └── processed/               # Salidas generadas (no se versiona)
├── results/                    # Métricas por serie y estadísticas de resultados
├── reports/
│   └── figures/                 # Gráficos PNG (eda/, articulo/es, articulo/en)
├── tests/                      # Pruebas unitarias (pytest)
├── pyproject.toml              # Configuración del paquete instalable
└── requirements.txt             # Dependencias del proyecto
```

La carpeta `_archive/` (no versionada, ver `.gitignore`) guarda localmente experimentos antiguos y datos que no forman parte del proyecto final.

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

### Simulaciones y reproducibilidad de resultados

Todas las cifras de resultados se pueden regenerar desde el repositorio (muestra estratificada de 51 series, semilla 42, ventana de prueba de 90 días):

```bash
python scripts/run_unified_simulation.py    # 4 modelos base  -> results/unified_metrics.csv
python scripts/run_ensemble_simulation.py   # Ensemble        -> results/ensemble_metrics.csv
python scripts/run_ensemble_predictions.py  # predicciones    -> results/ensemble_predictions.csv
python scripts/compute_article_stats.py     # tablas          -> results/article_stats.json
python scripts/make_article_figures.py es   # figuras         -> reports/figures/articulo/es
```

Las simulaciones se pueden interrumpir y continuar: cada serie se guarda al terminar y las ya calculadas se omiten.

### Aplicación web

```bash
streamlit run webapp/app.py
```

La app tiene dos pestañas y usa por defecto el modelo matemático principal, **Ensemble (Fusionado)**, que combina Prophet, Random Forest, XGBoost y LSTM con pesos aprendidos. Los 4 modelos base siguen disponibles en el selector para comparar (`MODEL_REGISTRY` en `src/dataframe_analyzer/service/forecast_service.py`).

**1. Mis ventas: ¿cuánto comprar?** — flujo para un usuario final:

1. Descarga la plantilla Excel desde la app (o usa el botón *Usar datos de ejemplo*).
2. Sube tu historial de ventas (`.xlsx` o `.csv`) con esta estructura:

   | columna | obligatoria | ejemplo |
   |---|---|---|
   | `fecha` | Sí | `2025-01-02` (también `02/01/2025` o celda de fecha de Excel) |
   | `tienda` | No | `Centro` (si falta, todo se toma como una sola tienda) |
   | `producto` | Sí | `Arroz 1kg` |
   | `cantidad_vendida` | Sí | `14` |

   Se aceptan sinónimos de columnas (`date`, `sku`, `ventas`, `unidades`, `sucursal`...). Puede haber una fila por día o una por venta (las del mismo día se suman); los días sin registros se toman como 0 ventas. Cada producto necesita al menos 120 días de historia (ideal: 2 años).
3. La app valida el archivo, reporta los problemas que encuentre y lista los productos aptos.
4. Elige el producto y genera su pronóstico (con el Ensemble tarda 1-3 minutos).
5. Elige la fecha objetivo, desde cuándo quieres cubrir, tu inventario actual y el nivel de servicio. La app recomienda cuántas unidades comprar:

   `unidades = ⌈ demanda esperada del periodo + z · RMSE · √días − inventario actual ⌉` (nunca negativo)

   donde `RMSE` es el error diario del modelo medido sobre tus propios datos y `z` depende del nivel de servicio (90 % → 1.28, 95 % → 1.645, 99 % → 2.33). También muestra la precisión semanal del modelo en tu producto y permite descargar la recomendación en Excel.

La lógica está en `src/dataframe_analyzer/data/sales_upload.py` (validación y normalización del archivo) y `src/dataframe_analyzer/service/inventory.py` (recomendación de compra).

**2. Dataset del proyecto** — pronósticos sobre cualquier tienda-producto del dataset de Kaggle del artículo, con métricas MAE/RMSE/MAPE y exportación a CSV. Lee `data/raw/train.csv` directamente (no el CSV procesado por el pipeline), porque los modelos necesitan la columna `date` original.

### Pruebas

```bash
pip install pytest
PYTHONPATH=src pytest tests -q
```

## Estado del proyecto

Este proyecto se evalúa mediante 3 hitos:

- **Hito 1** — Propuesta y Estado del Arte
- **Hito 2** — Modelo, Dataset y Demo
- **Hito 3** — Artículo Científico y Sustentación

## Contribuir

Antes de hacer cambios, revisa la [guía de contribución](./CONTRIBUTING.md) para convenciones de commits, ramas y pull requests.
