"""End-to-end test of the upload path: user file -> ForecastService -> recommendation.

Uses Prophet (fast) on a synthetic product with a clear weekly pattern,
so it runs in seconds. The Ensemble goes through the exact same code
path; it is exercised separately because it trains 8 models (slow).
"""

import numpy as np
import pandas as pd

from dataframe_analyzer.data.sales_upload import normalize_sales
from dataframe_analyzer.service.forecast_service import MODEL_REGISTRY, DEFAULT_MODEL, ForecastService
from dataframe_analyzer.service.inventory import choose_backtest_days, recommend_purchase


def _synthetic_upload(days=400):
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    weekly = np.where(dates.dayofweek >= 5, 30, 15)  # weekend peak
    trend = np.linspace(0, 5, days)
    rng = np.random.default_rng(0)
    sales = np.clip(weekly + trend + rng.normal(0, 1.5, days), 0, None).round()
    return pd.DataFrame({"fecha": dates.strftime("%Y-%m-%d"), "tienda": "Centro",
                         "producto": "Arroz 1kg", "cantidad_vendida": sales})


def test_ensemble_is_registered_as_default():
    assert DEFAULT_MODEL == "Ensemble (Fusionado)"
    assert DEFAULT_MODEL in MODEL_REGISTRY


def test_upload_to_recommendation_with_prophet():
    report = normalize_sales(_synthetic_upload())
    service = ForecastService.from_dataframe(report.data)
    backtest = choose_backtest_days(len(report.data))
    result = service.get_forecast(store="Centro", item="Arroz 1kg", horizon_days=30,
                                  model_name="Prophet", backtest_days=backtest)

    assert len(result.future_forecast) == 30
    assert result.future_forecast["ds"].min() == report.data["date"].max() + pd.Timedelta(days=1)
    # A clear weekly pattern should be learned well.
    assert result.metrics["MAPE"] < 15

    start = result.future_forecast["ds"].min()
    rec = recommend_purchase(result.future_forecast, start, start + pd.Timedelta(days=6),
                             daily_error=result.metrics["RMSE"], current_inventory=50)
    # ~ 5 weekdays * ~18 + 2 weekend days * ~33 = ~156 expected demand in a week
    assert 120 < rec.expected_demand < 200
    assert rec.recommended_units > 0
