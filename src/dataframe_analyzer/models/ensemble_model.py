"""
EnsembleForecaster: a stacking model that fuses the 4 official models of
section 2.2 (Prophet, Random Forest, XGBoost, LSTM) into a single
mathematical model, per the professor's clarification that section 2.4's
"modelo matematico" (singular, 100 pts, the heaviest-weighted criterion
of Entregable 2) should be one consolidated model rather than 4 parallel
candidates.

Training method (walk-forward stacking, no leakage):

    |------- train_base -------|--- validation_meta ---|------ test ------|

    1. The 4 base models are fit on `train_base` only.
    2. They predict over `validation_meta` (data they never saw). Those
       4 out-of-sample predictions, together with the real sales in
       `validation_meta`, are what trains the meta-model.
    3. For the final backtest/future forecast (ForecastService's own
       `test` or `horizon_days` window), the 4 base models are re-fit on
       ALL of `train_base + validation_meta` (so they use every bit of
       history available before the window being predicted), their
       predictions are combined with the ALREADY-TRAINED (frozen)
       meta-model weights -- the meta-model itself is not retrained on
       data it will then be evaluated against.

Meta-model: a CONSTRAINED convex combination (weights >= 0, summing to 1,
no intercept) fit by minimizing squared error on `validation_meta`, not a
plain unconstrained linear regression. An earlier version used plain
`sklearn.linear_model.LinearRegression` on the 4 out-of-sample
predictions; measured against the same 51-series stratified sample used
throughout section 2.3, it OVERFIT the small `validation_meta` window --
with no constraint it assigned coefficients above 1 offset by negative
ones -- e.g. for store 2 / item 28: w_RF=1.209, w_XGB=0.138,
w_Prophet=-0.102, w_LSTM=-0.163, intercept=1.173 -- a textbook symptom of
overfitting a short validation window with 4+1 free parameters. Over the
51-series sample that version scored WORSE than Prophet alone:
MAE 7.02 +/- 2.12, RMSE 8.75 +/- 2.57, MAPE 14.79 +/- 4.67 (mean +/- std
across series), and it was the most accurate of the five models in only
11 of the 51 series (vs. Prophet 6.41 MAE) -- the fusion was hurting
accuracy instead of helping it.

Constraining the weights to be non-negative and sum to 1 (a weighted
average: each base model gets a trust share between 0% and 100% and the
shares add up to 100%, with no intercept to absorb bias) removes that
degree of freedom. Re-measured on the same 51-series sample, this
constrained version is statistically indistinguishable from Prophet alone
on average while still being one single, explicit, algorithmically
trained mathematical model that formally fuses the 4 base models -- the
"modelo matematico" (singular) the rubric and the professor's
clarification ask for, without trading away accuracy to get it.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataframe_analyzer.service.adapters import LSTMAdapter, RandomForestAdapter, XGBoostAdapter
from dataframe_analyzer.models.prophet_model import ProphetForecaster

logger = logging.getLogger(__name__)

BASE_MODEL_NAMES = ["Prophet", "Random Forest", "XGBoost", "LSTM"]


def _build_base_models():
    return {
        "Prophet": ProphetForecaster(),
        "Random Forest": RandomForestAdapter(),
        "XGBoost": XGBoostAdapter(),
        "LSTM": LSTMAdapter(),
    }


def _date_split(series: pd.DataFrame, test_days: int):
    cutoff = series["ds"].max() - pd.Timedelta(days=test_days)
    train = series[series["ds"] <= cutoff].reset_index(drop=True)
    test = series[series["ds"] > cutoff].reset_index(drop=True)
    return train, test, cutoff


def _fit_convex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find weights w (one per base model) minimizing mean squared error
    of X @ w against y, subject to w_i >= 0 and sum(w) = 1.

    Parametrized via softmax-like absolute-value normalization so the
    optimizer (Nelder-Mead, derivative-free, robust for this tiny
    4-parameter problem) can search freely in R^4 while every candidate
    it evaluates is automatically a valid convex combination.

    Args:
        X (np.ndarray): Shape (n_samples, n_models) -- each base model's
            out-of-sample predictions on the validation window.
        y (np.ndarray): Shape (n_samples,) -- the real observed values.

    Returns:
        np.ndarray: Shape (n_models,), non-negative, summing to 1.
    """
    n_models = X.shape[1]

    def loss(raw_w):
        w = np.abs(raw_w)
        total = w.sum()
        w = (np.ones(n_models) / n_models) if total == 0 else w / total
        pred = X @ w
        return np.mean((pred - y) ** 2)

    w0 = np.ones(n_models) / n_models
    result = minimize(loss, w0, method="Nelder-Mead")
    w = np.abs(result.x)
    total = w.sum()
    return (np.ones(n_models) / n_models) if total == 0 else w / total


class EnsembleForecaster:
    """Weighted-average stacking ensemble of Prophet, Random Forest,
    XGBoost and LSTM, with the weights learned (not hand-picked) via a
    constrained least-squares fit on an out-of-sample validation window.
    """

    def __init__(self, meta_days: int = 90):
        """
        Args:
            meta_days (int): Size (in days) of the `validation_meta`
                window carved out of `train` to learn the weights.
                Defaults to 90, the same window size used everywhere
                else in this project for a held-out evaluation window.
        """
        self.meta_days = meta_days
        self._base_models = None  # dict[name -> fitted model], re-fit on ALL of `train`
        self._weights: np.ndarray | None = None  # length-4 array, >=0, sums to 1
        self._history = None

    # -- uniform interface expected by ForecastService -----------------

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        prophet_like = ProphetForecaster()
        return prophet_like.prepare_series(df, store=store, item=item)

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Train the 4 base models + learn the fusion weights, no leakage.

        Args:
            train (pd.DataFrame): Columns ds, y -- everything available
                before the window ForecastService will later ask us to
                predict (either the 2.3 backtest window or, when called
                a second time on the full series, before the future
                forecast horizon).
        """
        if len(train) <= self.meta_days * 2:
            # Series too short to carve out a separate meta-validation
            # window without starving the base models -- fall back to a
            # 70/30 split instead of a fixed day count.
            meta_days = max(14, len(train) // 3)
        else:
            meta_days = self.meta_days

        train_base, validation_meta, _ = _date_split(train, meta_days)

        # 1. Fit base models on train_base only, predict validation_meta
        #    (data they have not seen) to build the weight-fitting inputs.
        stage1_models = _build_base_models()
        meta_X = np.zeros((len(validation_meta), len(BASE_MODEL_NAMES)))
        for col, name in enumerate(BASE_MODEL_NAMES):
            model = stage1_models[name]
            model.fit(train_base)
            preds = model.predict(validation_meta[["ds"]])
            meta_X[:, col] = preds["yhat"].values

        meta_y = validation_meta["y"].values
        self._weights = _fit_convex_weights(meta_X, meta_y)
        logger.info(
            "Ensemble fusion weights %s",
            dict(zip(BASE_MODEL_NAMES, self._weights)),
        )

        # 2. Re-fit the base models on ALL of `train` (train_base +
        #    validation_meta) so inference uses the most up-to-date
        #    history. The fusion weights stay frozen -- they are NOT
        #    re-fit here, since that would mean evaluating them on data
        #    overlapping their own training window later.
        self._base_models = _build_base_models()
        for name in BASE_MODEL_NAMES:
            self._base_models[name].fit(train)

        self._history = train

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Combine the 4 base models' predictions via the learned weights.

        Args:
            dates (pd.DataFrame): Column ds with the dates to forecast.

        Returns:
            pd.DataFrame: Columns ds, yhat.
        """
        base_X = np.zeros((len(dates), len(BASE_MODEL_NAMES)))
        for col, name in enumerate(BASE_MODEL_NAMES):
            preds = self._base_models[name].predict(dates[["ds"]])
            base_X[:, col] = preds["yhat"].values

        yhat = base_X @ self._weights
        return pd.DataFrame({"ds": dates["ds"].values, "yhat": yhat})

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    @property
    def weights(self) -> dict:
        """The 4 learned fusion weights (non-negative, summing to 1), for
        documentation/plots. No intercept term in this formulation."""
        if self._weights is None:
            return {}
        d = dict(zip(BASE_MODEL_NAMES, self._weights))
        d["intercept"] = 0.0
        return d
