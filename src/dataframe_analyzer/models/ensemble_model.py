"""EnsembleForecaster: one model that combines the 4 base models.

The base models are Prophet, Random Forest, XGBoost and LSTM. The
ensemble joins them into one mathematical model with stacking.

Training method (walk-forward stacking, without data leakage):

    |------- train_base -------|--- validation_meta ---|------ test ------|

    1. The 4 base models are trained only with `train_base`.
    2. They predict `validation_meta`, which is data that they did not
       see. These 4 predictions and the real sales in `validation_meta`
       are used to learn the weights.
    3. For the final forecast (the `test` window or the future
       `horizon_days`), the 4 base models are trained again with all of
       `train_base + validation_meta`, so they use all the history before
       the window to predict. Their predictions are combined with the
       weights from step 2. The weights are frozen: they are not learned
       again with data that is later used to evaluate them.

Weights: a convex combination. Each weight is >= 0, the weights add up
to 1 and there is no intercept. The weights minimize the squared error
on `validation_meta`.

Why the weights have these limits: an earlier version used a normal
`sklearn.linear_model.LinearRegression` without limits. On the same
sample of 51 series, it overfit the short `validation_meta` window. It
gave some models weights above 1 and others negative weights. For
example, for store 2 and item 28: w_RF=1.209, w_XGB=0.138,
w_Prophet=-0.102, w_LSTM=-0.163, intercept=1.173. This is a typical
sign of overfitting with 4+1 free parameters and a short window. On the
51 series that version was worse than Prophet alone: MAE 7.02 +/- 2.12,
RMSE 8.75 +/- 2.57, MAPE 14.79 +/- 4.67 (mean +/- standard deviation),
compared with a Prophet MAE of 6.41. It was the best of the five models
in only 11 of the 51 series.

With the limits, the result is a weighted average: each base model gets
a share between 0% and 100%, the shares add up to 100%, and there is no
intercept. On the same 51 series, this version has no significant
difference from Prophet on average, and it is still one single,
explicit mathematical model that combines the 4 base models.
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
    """Create new, untrained copies of the 4 base models.

    Returns:
        dict: Map from each name in `BASE_MODEL_NAMES` to a new model.
    """
    return {
        "Prophet": ProphetForecaster(),
        "Random Forest": RandomForestAdapter(),
        "XGBoost": XGBoostAdapter(),
        "LSTM": LSTMAdapter(),
    }


def _date_split(series: pd.DataFrame, test_days: int):
    """Split a series by date into a first part and a last part.

    Args:
        series (pd.DataFrame): Columns `ds` and `y`.
        test_days (int): Number of last days in the second part.

    Returns:
        tuple: `(train, test, cutoff)`, where `cutoff` is the last date of
        the first part.
    """
    cutoff = series["ds"].max() - pd.Timedelta(days=test_days)
    train = series[series["ds"] <= cutoff].reset_index(drop=True)
    test = series[series["ds"] > cutoff].reset_index(drop=True)
    return train, test, cutoff


def _fit_convex_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find the convex weights that minimize the mean squared error.

    The function looks for one weight w_i for each base model that
    minimizes the mean squared error of `X @ w` against `y`, with
    w_i >= 0 and sum(w) = 1.

    The optimizer works with free values theta, and each weight is
    w_i = |theta_i| / sum(|theta_j|). In this way every value that the
    optimizer tests is a valid convex combination. Nelder-Mead does not
    need derivatives and works well for this small problem with 4
    parameters.

    Args:
        X (np.ndarray): Shape (n_samples, n_models). Predictions of each
            base model on the validation window (data not used for
            training).
        y (np.ndarray): Shape (n_samples,). Real values.

    Returns:
        np.ndarray: Shape (n_models,). Weights that are >= 0 and add up
        to 1.
    """
    n_models = X.shape[1]

    def loss(raw_w):
        """Calculate the mean squared error for the free values `raw_w`.

        Args:
            raw_w (np.ndarray): Free values that become weights after
                normalization.

        Returns:
            float: Mean squared error of the weighted prediction.
        """
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
    """Stacking ensemble of Prophet, Random Forest, XGBoost and LSTM.

    The forecast is a weighted average of the 4 base models. The weights
    are learned from data (not chosen by hand) with least squares and
    convex limits, on a validation window that the base models did not
    use for training.
    """

    def __init__(self, meta_days: int = 90):
        """Set the size of the weight validation window.

        Args:
            meta_days (int): Size in days of the `validation_meta` window
                taken from `train` to learn the weights. Defaults to 90,
                the same size as the test window in the rest of the
                project.
        """
        self.meta_days = meta_days
        self._base_models = None  # Name -> model trained with all of `train`.
        self._weights: np.ndarray | None = None  # 4 values, >= 0, sum = 1.
        self._history = None

    # -- Common interface used by ForecastService ----------------------

    def prepare_series(self, df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
        """Get the daily sales series of one store and item.

        Args:
            df (pd.DataFrame): Clean dataset with the columns `date`,
                `store`, `item` and `sales`.
            store (int): Store id.
            item (int): Item id.

        Returns:
            pd.DataFrame: Series with the columns `ds` and `y`.
        """
        prophet_like = ProphetForecaster()
        return prophet_like.prepare_series(df, store=store, item=item)

    def split_train_test(self, series: pd.DataFrame, test_days: int):
        """Split the series into a training part and a test part by date.

        Args:
            series (pd.DataFrame): Columns `ds` and `y`.
            test_days (int): Number of last days kept for the test.

        Returns:
            tuple: `(train, test, cutoff)`.
        """
        return _date_split(series, test_days)

    def fit(self, train: pd.DataFrame) -> None:
        """Train the 4 base models and learn the weights, without leakage.

        Args:
            train (pd.DataFrame): Columns `ds` and `y`. All the data
                before the window that will be predicted later (the
                backtest window or, for the full series, the future
                horizon).
        """
        if len(train) <= self.meta_days * 2:
            # The series is too short to keep a full validation window
            # and still train the base models well. In this case we use
            # about the last third of the data for validation.
            meta_days = max(14, len(train) // 3)
        else:
            meta_days = self.meta_days

        train_base, validation_meta, _ = _date_split(train, meta_days)

        # 1. Train the base models only with train_base and predict
        #    validation_meta (new data for them) to learn the weights.
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

        # 2. Train the base models again with all of `train` (train_base
        #    and validation_meta), so the forecast uses the most recent
        #    history. The weights stay frozen. They are not learned again,
        #    because then they would be evaluated on their own data.
        self._base_models = _build_base_models()
        for name in BASE_MODEL_NAMES:
            self._base_models[name].fit(train)

        self._history = train

    def predict(self, dates: pd.DataFrame) -> pd.DataFrame:
        """Combine the predictions of the 4 base models with the weights.

        Args:
            dates (pd.DataFrame): Column `ds` with the dates to forecast.

        Returns:
            pd.DataFrame: Columns `ds` and `yhat`.
        """
        base_X = np.zeros((len(dates), len(BASE_MODEL_NAMES)))
        for col, name in enumerate(BASE_MODEL_NAMES):
            preds = self._base_models[name].predict(dates[["ds"]])
            base_X[:, col] = preds["yhat"].values

        yhat = base_X @ self._weights
        return pd.DataFrame({"ds": dates["ds"].values, "yhat": yhat})

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate MAE, RMSE and MAPE.

        In MAPE, days with 0 real sales are divided by 1 instead of 0.

        Args:
            y_true (np.ndarray): Real values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            dict: Keys `MAE`, `RMSE` and `MAPE` (MAPE in percent).
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    @property
    def weights(self) -> dict:
        """Get the 4 learned weights, for reports and charts.

        Returns:
            dict: Weight of each base model (>= 0, sum = 1) and the key
            `intercept`, which is always 0.0. The dictionary is empty if
            the model is not trained yet.
        """
        if self._weights is None:
            return {}
        d = dict(zip(BASE_MODEL_NAMES, self._weights))
        d["intercept"] = 0.0
        return d
