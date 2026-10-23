"""Tests for the purchase recommendation (dataframe_analyzer.service.inventory)."""

import math

import numpy as np
import pandas as pd
import pytest

from dataframe_analyzer.service.inventory import (
    choose_backtest_days,
    recommend_purchase,
    weekly_accuracy,
)


def _forecast(values, start="2026-01-01"):
    return pd.DataFrame({"ds": pd.date_range(start, periods=len(values)), "yhat": values})


def test_recommendation_formula():
    fc = _forecast([10.0] * 30)
    rec = recommend_purchase(fc, "2026-01-01", "2026-01-07", daily_error=4.0,
                             current_inventory=20, service_level=0.95)
    expected_ss = 1.6448536 * 4.0 * math.sqrt(7)
    assert rec.days == 7
    assert rec.expected_demand == pytest.approx(70.0)
    assert rec.safety_stock == pytest.approx(expected_ss, rel=1e-5)
    assert rec.recommended_units == math.ceil(70 + expected_ss - 20)
    assert rec.demand_on_target_day == pytest.approx(10.0)


def test_never_recommends_negative_purchase():
    rec = recommend_purchase(_forecast([5.0] * 10), "2026-01-01", "2026-01-03",
                             daily_error=1.0, current_inventory=1000)
    assert rec.recommended_units == 0


def test_negative_forecasts_are_clipped():
    rec = recommend_purchase(_forecast([-3.0, 2.0, 2.0]), "2026-01-01", "2026-01-03", daily_error=0.0)
    assert rec.expected_demand == pytest.approx(4.0)
    assert rec.recommended_units == 4


def test_higher_service_level_means_more_stock():
    fc = _forecast([10.0] * 14)
    low = recommend_purchase(fc, "2026-01-01", "2026-01-14", daily_error=3.0, service_level=0.90)
    high = recommend_purchase(fc, "2026-01-01", "2026-01-14", daily_error=3.0, service_level=0.99)
    assert high.recommended_units > low.recommended_units


def test_period_outside_forecast_or_reversed_is_rejected():
    fc = _forecast([10.0] * 5)
    with pytest.raises(ValueError, match="no cubre"):
        recommend_purchase(fc, "2026-01-01", "2026-01-10", daily_error=1.0)
    with pytest.raises(ValueError, match="posterior"):
        recommend_purchase(fc, "2026-01-03", "2026-01-01", daily_error=1.0)


def test_weekly_accuracy():
    real = np.array([10.0] * 21)
    pred = np.concatenate([[10.0] * 7, [11.0] * 7, [20.0] * 7])  # within, within (+10%), off (+100%)
    acc = weekly_accuracy(real, pred, tolerance=0.15)
    assert acc == {"accuracy_pct": pytest.approx(200 / 3), "weeks": 3, "weeks_within": 2}
    assert weekly_accuracy(real[:5], pred[:5])["accuracy_pct"] is None


def test_choose_backtest_days():
    assert choose_backtest_days(1826) == 90
    assert choose_backtest_days(200) == 40
    assert choose_backtest_days(40) == 14
