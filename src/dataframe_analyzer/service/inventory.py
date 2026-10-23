"""Purchase recommendation based on a demand forecast.

A forecast only gives information. The store manager needs a decision:
"how many units should I buy?". This module changes the forecast into
that number.

Recommended purchase for a period [start, end]:

    D       = sum of forecasted daily demand over the period
    SS      = z * sigma * sqrt(n)                  (safety stock)
    Q       = max(0, ceil(D + SS - I))             (units to buy)

where
    n       = number of days in the period,
    sigma   = RMSE of the model on its backtest window for this product
              (the usual daily error, measured on data that the model
              did not use for training),
    z       = standard normal quantile for the chosen service level
              (90% -> 1.28, 95% -> 1.645, 99% -> 2.33),
    I       = units that the user already has at the start of the period.

The sqrt(n) term is the standard safety stock formula. It supposes that
daily errors are independent, so part of the errors cancel over several
days. Because of this, the uncertainty of the total grows with the
square root of the number of days, not in a linear way. Daily errors are
often a little correlated, so this is only an approximation. For this
reason the app also shows the weekly accuracy of the model on the user's
own data, so the user can decide how much to trust the number.
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
        period_start (pd.Timestamp): First day of the period.
        period_end (pd.Timestamp): Last day of the period (the target
            date of the user).
        days (int): Number of days in the period.
        expected_demand (float): Total forecast demand in the period.
        demand_on_target_day (float): Forecast demand on the target date
            only.
        safety_stock (float): Extra units to cover forecast errors.
        current_inventory (float): Units that the user already has.
        recommended_units (int): Units to buy (never negative).
        service_level (float): Chosen probability of not running out of
            stock.
        z (float): Standard normal quantile used.
        daily_error (float): Value of sigma (model RMSE on its backtest).
        daily_forecast (pd.DataFrame): Columns `ds` and `yhat` for each
            day of the period.
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
    """Calculate how many units to buy to cover the demand of a period.

    Args:
        forecast (pd.DataFrame): Daily forecast with the columns `ds` and
            `yhat`. It must include every day from `period_start` to
            `period_end`.
        period_start: First day that the purchase must cover.
        period_end: Last day that the purchase must cover (target date).
        daily_error (float): Model RMSE on its backtest window (sigma).
        current_inventory (float): Units in stock on `period_start`.
        service_level (float): Wanted probability of not running out of
            stock (for example 0.95).

    Returns:
        PurchaseRecommendation: Units to buy and the values used to
        calculate them.

    Raises:
        ValueError: If the period is not valid, if the service level or
            the inventory is out of range, or if the forecast does not
            cover the whole period.
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

    # Demand cannot be negative, so negative predictions become 0 first.
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
    """Calculate the share of full weeks with a good forecast total.

    A week is good when its forecast total is within +-tolerance of the
    real total. This is the same weekly accuracy used in the study, here
    calculated for the user's product on the backtest window.

    Args:
        y_true: Real daily values of the backtest window, in date order.
        y_pred: Predicted daily values for the same days.
        tolerance (float): Relative tolerance (0.15 means +-15%).

    Returns:
        dict: Keys `accuracy_pct` (float, or None when the window has no
        full week), `weeks` (number of full weeks) and `weeks_within`
        (number of good weeks).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    n_weeks = len(y_true) // 7
    if n_weeks == 0:
        return {"accuracy_pct": None, "weeks": 0, "weeks_within": 0}
    real = y_true[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    pred = y_pred[: n_weeks * 7].reshape(n_weeks, 7).sum(axis=1)
    # A week with 0 real sales is good only if the forecast is also close to 0.
    within = np.where(real > 0, np.abs(real - pred) <= tolerance * real, pred < 1)
    return {
        "accuracy_pct": float(within.mean() * 100),
        "weeks": int(n_weeks),
        "weeks_within": int(within.sum()),
    }


def choose_backtest_days(history_days: int, default: int = 90) -> int:
    """Choose the size of the evaluation window for a daily series.

    The standard window is 90 days. For short series the window is
    smaller (about 1/5 of the history, but never less than 2 weeks), so
    enough data stays for training.

    Args:
        history_days (int): Number of days in the series.
        default (int): Window size for long series.

    Returns:
        int: Number of last days to keep for the evaluation.
    """
    return int(min(default, max(14, history_days // 5)))
