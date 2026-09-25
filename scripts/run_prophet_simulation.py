"""
Run script: Prophet simulation for demand forecasting.

TRAZABILIDAD:
    1. Carga de datos           -> DataLoader.load()
    2. Selección de la serie    -> ProphetForecaster.prepare_series()
    3. Split cronológico        -> ProphetForecaster.split_train_test()
    4. Entrenamiento            -> ProphetForecaster.fit()
    5. Predicción                -> ProphetForecaster.predict()
    6. Cálculo de métricas      -> ProphetForecaster.compute_metrics()
"""

import logging

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.prophet_model import ProphetForecaster

logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/raw/train.csv"
STORE = 1
ITEM = 1
TEST_DAYS = 90

if __name__ == "__main__":
    # 1. Carga de datos (reutiliza DataLoader ya existente en el repo)
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # 2-6. Preparación, split, entrenamiento, predicción y métricas
    forecaster = ProphetForecaster()
    serie = forecaster.prepare_series(df, store=STORE, item=ITEM)
    train, test, cutoff = forecaster.split_train_test(serie, test_days=TEST_DAYS)

    forecaster.fit(train)
    forecast = forecaster.predict(test)

    y_true = test["y"].values
    y_pred = forecast["yhat"].values
    metrics = forecaster.compute_metrics(y_true, y_pred)

    print(f"Prophet metrics (store={STORE}, item={ITEM}): {metrics}")