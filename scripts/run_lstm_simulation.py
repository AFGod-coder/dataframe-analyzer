"""
Run script: LSTM simulation for demand forecasting.

TRAZABILIDAD:
    1. Carga de datos           -> DataLoader.load()
    2. Selección de la serie    -> LSTMForecaster.prepare_series()
    3. Escalado                 -> LSTMForecaster.scale()
    4. Construcción de ventanas -> LSTMForecaster.create_windows()
    5. Split cronológico        -> LSTMForecaster.split_train_test()
    6. Entrenamiento            -> LSTMForecaster.fit()
    7. Predicción y desescalado -> LSTMForecaster.predict()
    8. Cálculo de métricas      -> LSTMForecaster.compute_metrics()
"""

import logging

from dataframe_analyzer.data.loader import DataLoader
from dataframe_analyzer.models.lstm_model import LSTMForecaster

logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/raw/train.csv"
STORE = 1
ITEM = 1
WINDOW = 30
TEST_DAYS = 90

if __name__ == "__main__":
    # 1. Carga de datos (reutiliza DataLoader ya existente en el repo)
    loader = DataLoader(DATA_PATH)
    df = loader.load()

    # 2-8. Preparación, escalado, ventanas, split, entrenamiento, predicción y métricas
    forecaster = LSTMForecaster(window=WINDOW)
    serie = forecaster.prepare_series(df, store=STORE, item=ITEM)
    scaled = forecaster.scale(serie)
    X, y = forecaster.create_windows(scaled)
    X_train, X_test, y_train, y_test = forecaster.split_train_test(X, y, test_days=TEST_DAYS)

    forecaster.fit(X_train, y_train)
    y_pred = forecaster.predict(X_test)
    y_true = forecaster.inverse_transform_y(y_test)

    metrics = forecaster.compute_metrics(y_true, y_pred)
    print(f"LSTM metrics (store={STORE}, item={ITEM}): {metrics}")