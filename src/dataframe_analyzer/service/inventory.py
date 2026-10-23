"""
Purchase recommendation: turns a demand forecast into "how many units
should I buy", which is the actual decision the article's third
research question is about (section 1.1) -- a forecast alone is
information, not a decision.

Recommended purchase for a coverage period [start, end]:

    D       = sum of forecasted daily demand over the period
    SS      = z * sigma * sqrt(n)                  (safety stock)
    Q       = max(0, ceil(D + SS - I))             (units to buy)

where
    n       = number of days in the period,
    sigma   = RMSE of the model on its own backtest window for THIS
              product (the model's typical daily error, measured on
              data it did not train on),
    z       = standard-normal quantile for the chosen service level
              (90% -> 1.28, 95% -> 1.645, 99% -> 2.33),
    I       = inventory the user already has at the start of the period.

The sqrt(n) term is the textbook safety-stock approximation that treats
daily forecast errors as independent: errors partly cancel out over
several days, so the uncertainty of a period total grows with the square
root of its length, not linearly. It is an approximation (consecutive
daily errors are usually somewhat correlated), which is why the app
also reports the model's weekly accuracy on the user's own data, so the
user can judge how much to trust the number.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

SERVICE_LEVELS = {"90%": 0.90, "95%": 0.95, "99%": 0.99}


@dataclass
class PurchaseRecommendation:
    """Result of `recommend_purchase`.

    Attributes:
        period_start (pd.Timestamp): First day covered.
        period_end (pd.Timestamp): Last day covered (the user's target date).
        days (int): Number of days in the period.
        expected_demand (float): Sum of forecasted demand over the period.
        demand_on_target_day (float): Forecasted demand for the target date alone.
        safety_stock (float): Extra units to absorb forecast error.
        current_inventory (float): Units the user already has.
        recommended_units (int): Units to buy (never negative).
        service_level (float): Chosen probability of not running out.
        z (float): Standard-normal quantile used.
        daily_error (float): sigma used (model RMSE on its backtest).
        daily_forecast (pd.DataFrame): ds, yhat for each day of the period.
    """
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    days: int
    expected_demand: float
    demand_on_target_day: float
    safety_stock: float
    current_inventory: float
    recommended_units: int
    service_level: float
    z: float
    daily_error: float
    daily_forecast: pd.DataFrame


def recommend_purchase(
    forecast: pd.DataFrame,
    period_start,
    period_end,
    daily_error: float,
    current_inventory: float = 0.0,
    service_level: float = 0.95,
) -> PurchaseRecommendation:
    """Compute how many units to buy to cover demand over a period.

    Args:
        forecast (pd.DataFrame): Daily forecast with columns ds, yhat,
            covering at least [period_start, period_end].
        period_start: First day the purchase must cover.
        period_end: Last day the purchase must cover (target date).
        daily_error (float): Model RMSE on its backtest window (sigma).
        current_inventory (float): Units already in stock at period_start.
        service_level (float): Desired probability of not running out
            (e.g. 0.95).

    Returns:
        PurchaseRecommendation

    Raises:
        ValueError: If the period is invalid or not covered by the forecast.
    """
    start = pd.Timestamp(period_start).normalize()
    end = pd.Timestamp(period_end).normalize()
    if end < start:
        raise ValueError("La fecha final del periodo debe ser igual o posterior a la fecha inicial.")
    if not 0.5 <= service_level < 1:
        raise ValueError("El nivel de servicio debe estar entre 50% y 99.9%.")
    if current_inventory < 0:
        raise ValueError("El inventario actual no puede ser negativo.")

    fc = forecast.copy()
    fc["ds"] = pd.to_datetime(fc["ds"]).dt.normalize()
    period = fc[(fc["ds"] >= start) & (fc["ds"] <= end)].reset_index(drop=True)
    expected_days = (end - start).days + 1
    if len(period) != expected_days:
        raise ValueError(
            "El pronóstico no cubre todo el periodo solicitado. "
            f"Periodo: {start.date()} a {end.date()}, días pronosticados: {len(period)}."
        )

    # Demand cannot be negative: clip model outputs before summing.
    period["yhat"] = period["yhat"].clip(lower=0)
    expected_demand = float(period["yhat"].sum())
    demand_on_target_day = float(period.loc[period["ds"] == end, "yhat"].iloc[0])

    z = float(norm.ppf(service_level))
    safety_stock = float(z * max(daily_error, 0.0) * math.sqrt(expected_days))
    units = max(0, math.ceil(expected_demand + safety_stock - current_inventory))

    return PurchaseRecommendation(
        period_start=start,
        period_end=end,
        days=expected_days,
        expected_demand=expected_demand,
        demand_on_target_day=demand_on_target_day,
        safety_stock=safety_stock,
        current_inventory=float(current_inventory),
        recommended_units=int(units),
        service_level=service_level,
        z=z,
        daily_error=float(daily_error),
        daily_forecast=period[["ds", "yhat"]],
    )


def weekly_accuracy(y_true, y_pred, tolerance: float = 0.15) -> dict:
    """Share of full weeks whose forecasted total is within +-tolerance of the real total.

    Same definition used for the ">95% accuracy" criterion in section 2.5,
    applied here to the user's own product, on the model's backtest window.

    Args:
        y_true: Real daily values of the backtest window (chronological).
        y_pred: Predicted daily values for the same days.
        tolerance (float): Relative tolerance (0.15 = +-15%).

    Returns:
        dict: {"accuracy_pct": float | None, "weeks": int, "weeks_within": int}.
        accuracy_pct is None if the window has no complete week.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    n_weeks = len(y_true) // 7
    if n_weeks == 0:
        return {"accuracy_pct": None, "weeks": 0, "weeks_within": 0}
    real = y_true[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    pred = y_pred[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    # A week with 0 real sales counts as accurate only if the forecast is ~0 too.
    within = np.where(real > 0, np.abs(real - pred) <= tolerance * real, pred < 1)
    return {
        "accuracy_pct": float(within.mean() * 100),
        "weeks": int(n_weeks),
        "weeks_within": int(within.sum()),
    }


def choose_backtest_days(history_days: int, default: int = 90) -> int:
    """Size of the evaluation window for a series of `history_days` days.

    Keeps the project's standard 90-day window when there is enough
    history, and shrinks it for short uploads so enough data is left to
    train on (roughly 1/5 of the history, never below 2 weeks).

    Args:
        history_days (int): Length of the daily series.
        default (int): Window used for long series.

    Returns:
        int: Number of trailing days to hold out for evaluation.
    """
    return int(min(default, max(14, history_days // 5)))
