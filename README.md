# dataframe-analyzer

Proyecto final de **Ciencias de la Computación II** (Electiva en Ingeniería Aplicada II / Data Science) — Grupo 6.

**Demand Forecasting to Support Inventory Management in Retail Stores**

Analiza el historial de ventas de un comerciante para construir un modelo que prediga la cantidad de producto que debería adquirir, apoyando así la gestión de inventario en tiendas retail.

## Estructura del proyecto

```
.
├── main.py            # Punto de entrada: orquesta carga, limpieza y transformación
├── DataLoader.py       # Carga del dataset crudo (train.csv)
├── DataCleaner.py      # Limpieza: normalización de texto y remoción de duplicados
├── DataTransform.py    # Transformaciones de fecha y features derivadas
├── train.csv           # Dataset de entrenamiento
└── requirements.txt     # Dependencias del proyecto
```

## Requisitos

- Python 3.10+
- Dependencias listadas en `requirements.txt`

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Esto carga `train.csv`, limpia los datos y genera `clean_dataframe.csv` con las columnas de fecha descompuestas (día, mes, año).

## Estado del proyecto

Este proyecto se evalúa mediante 3 hitos:

- **Hito 1** — Propuesta y Estado del Arte
- **Hito 2** — Modelo, Dataset y Demo
- **Hito 3** — Artículo Científico y Sustentación

## Contribuir

Antes de hacer cambios, revisa la [guía de contribución](./CONTRIBUTING.md) para convenciones de commits, ramas y pull requests.
