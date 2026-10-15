"""Shared, leakage-safe components for the final regression steps."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lasso,
    QuantileRegressor, Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


from .regression_data import CLINICAL_FEATURES, TARGET, NON_RADIOMICS, load_regression_data
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
LASSO_ALPHAS = tuple(np.logspace(-3, 1, 16))
ABLATION_STRATEGIES = ("all", "pca", "lasso", "pca_lasso")
ADDITIONAL_MODELS = (
    "ridge", "elastic_net", "bayesian_ridge", "ard", "pls", "huber",
    "quantile", "svr_rbf", "gradient_boosting", "extra_trees",
)


class NonEmptyLassoSelector(BaseEstimator, TransformerMixin):
    """Select non-zero LASSO features and retain one if all are zero."""

    def __init__(self, alpha=0.1, max_iter=500_000, tol=1e-3):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        values = np.asarray(x, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.estimator_ = Lasso(
                alpha=self.alpha, max_iter=self.max_iter, tol=self.tol
            ).fit(values, np.asarray(y, dtype=float))
        coefficients = np.abs(self.estimator_.coef_)
        self.raw_nonzero_count_ = int((coefficients > 1e-10).sum())
        self.fallback_ = self.raw_nonzero_count_ == 0
        self.support_ = coefficients > 1e-10
        if self.fallback_:
            self.support_[int(np.argmax(coefficients))] = True
        self.n_features_in_ = int(values.shape[1])
        self.n_features_out_ = int(self.support_.sum())
        return self

    def transform(self, x):
        return np.asarray(x, dtype=float)[:, self.support_]

    def get_support(self):
        return self.support_


def radiomics_pipeline(strategy, pca_variance=0.95):
    """Create one fold-fitted radiomics feature strategy."""
    if strategy not in ABLATION_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    steps = [
        ("variance", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
    ]
    if strategy in {"pca", "pca_lasso"}:
        steps.append(("pca", PCA(pca_variance, svd_solver="full")))
    if strategy in {"lasso", "pca_lasso"}:
        steps.append(("selector", NonEmptyLassoSelector()))
    return Pipeline(steps)


def strategy_pipeline_and_grid(strategy, radiomics_count, pca_variance=0.95):
    """Build Clinical Ridge or one early-fusion Ridge ablation pipeline."""
    if strategy == "clinical":
        return Pipeline([
            ("preprocess", StandardScaler()), ("model", Ridge()),
        ]), {"model__alpha": RIDGE_ALPHAS}
    clinical_indices = list(range(len(CLINICAL_FEATURES)))
    radio_indices = list(range(
        len(CLINICAL_FEATURES), len(CLINICAL_FEATURES) + radiomics_count
    ))
    preprocess = ColumnTransformer([
        ("clinical", StandardScaler(), clinical_indices),
        ("radiomics", radiomics_pipeline(strategy, pca_variance),
         radio_indices),
    ])
    grid = {"model__alpha": RIDGE_ALPHAS}
    if strategy in {"lasso", "pca_lasso"}:
        grid["preprocess__radiomics__selector__alpha"] = LASSO_ALPHAS
    return Pipeline([
        ("preprocess", preprocess), ("model", Ridge()),
    ]), grid


def additional_preprocessor(modality, radiomics_count):
    """Preprocessing used for additional-regressor screening."""
    if modality == "clinical":
        return StandardScaler()
    clinical_indices = list(range(len(CLINICAL_FEATURES)))
    radio_indices = list(range(
        len(CLINICAL_FEATURES), len(CLINICAL_FEATURES) + radiomics_count
    ))
    radiomics = Pipeline([
        ("variance", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("clinical", StandardScaler(), clinical_indices),
        ("radiomics", radiomics, radio_indices),
    ])


def additional_model_and_grid(name, modality, seed):
    """Return one pre-specified additional regressor and tuning grid."""
    if name == "ridge":
        return Ridge(), {"model__alpha": RIDGE_ALPHAS}
    if name == "elastic_net":
        return ElasticNet(
            max_iter=200_000, tol=1e-3, selection="cyclic",
            random_state=seed,
        ), {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
        }
    if name == "bayesian_ridge":
        return BayesianRidge(max_iter=2_000, tol=1e-4), {}
    if name == "ard":
        return ARDRegression(max_iter=2_000, tol=1e-3), {
            "model__threshold_lambda": [1_000.0, 10_000.0, 100_000.0],
        }
    if name == "pls":
        components = (
            [1, 2, 3, 4, 5, 6]
            if modality == "clinical" else [2, 3, 5, 8, 10]
        )
        return PLSRegression(
            scale=False, max_iter=2_000, tol=1e-7
        ), {"model__n_components": components}
    if name == "huber":
        return HuberRegressor(max_iter=5_000, tol=1e-5), {
            "model__epsilon": [1.1, 1.35, 1.75],
            "model__alpha": [0.01, 0.1, 1.0],
        }
    if name == "quantile":
        return QuantileRegressor(
            quantile=0.5, solver="highs"
        ), {"model__alpha": [0.01, 0.1, 1.0, 10.0]}
    if name == "svr_rbf":
        return SVR(kernel="rbf", cache_size=1_000), {
            "model__C": [1.0, 10.0, 100.0],
            "model__epsilon": [0.1, 1.0],
            "model__gamma": ["scale", 0.01],
        }
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            loss="huber", random_state=seed
        ), {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2],
            "model__min_samples_leaf": [5, 10],
        }
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300, random_state=seed, n_jobs=1
        ), {
            "model__max_features": [0.5, 1.0],
            "model__min_samples_leaf": [2, 5],
            "model__max_depth": [None, 5],
        }
    raise ValueError(f"Unknown model: {name}")


def regression_metrics(observed, predicted):
    """Return MSE, MAE, RMSE and R-squared."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mse = float(mean_squared_error(observed, predicted))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(observed, predicted)),
    }


def safe_r2(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    if len(observed) < 2 or np.var(observed) <= np.finfo(float).eps:
        return np.nan
    return float(r2_score(observed, predicted))


def convergence_audit(model):
    """Return iterations, final convergence and non-zero coefficient count."""
    n_iter = getattr(model, "n_iter_", np.nan)
    maximum = getattr(model, "max_iter", np.nan)
    if isinstance(n_iter, (list, tuple, np.ndarray)):
        values = np.asarray(n_iter, dtype=float)
        converged = bool(np.all(values < float(maximum)))
        recorded = json.dumps(values.astype(int).tolist())
    elif np.isscalar(n_iter) and not pd.isna(n_iter):
        recorded = int(n_iter)
        unlimited = (
            maximum is None or pd.isna(maximum) or float(maximum) < 0
        )
        converged = True if unlimited else int(n_iter) < int(maximum)
    else:
        recorded = np.nan
        converged = True
    coefficient = getattr(model, "coef_", None)
    nonzero = np.nan
    if coefficient is not None:
        values = np.asarray(coefficient, dtype=float).ravel()
        nonzero = int(np.sum(np.abs(values) > 1e-10))
    return recorded, converged, nonzero


def bootstrap_draws(patient_count, random_state, final_attempt):
    """Reproduce project draws, requiring at least 20 unique OOB patients."""
    random = np.random.default_rng(random_state + 100_000)
    draws = {}
    for replicate in range(1, final_attempt + 1):
        while True:
            inbag = random.integers(0, patient_count, size=patient_count)
            oob = np.setdiff1d(
                np.arange(patient_count), np.unique(inbag)
            )
            if len(oob) >= 20:
                break
        draws[replicate] = inbag, oob
    return draws


def paired_mae_bootstrap(observed, predictions, resamples, random_state):
    """Patient-level paired bootstrap and all pairwise MAE differences."""
    observed = np.asarray(observed, dtype=float)
    names = list(predictions)
    point = {
        name: regression_metrics(observed, predictions[name])
        for name in names
    }
    random = np.random.default_rng(random_state)
    indices = random.integers(
        0, len(observed), size=(resamples, len(observed))
    )
    distributions = {}
    for name in names:
        error = np.abs(np.asarray(predictions[name], float) - observed)
        distributions[name] = error[indices].mean(axis=1)
    ci_rows = []
    for name in names:
        values = distributions[name]
        ci_rows.append({
            "configuration": name,
            **point[name],
            "mae_ci95_low": float(np.quantile(values, 0.025)),
            "mae_ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_resamples": int(resamples),
        })
    delta_rows = []
    for first_index, reference in enumerate(names):
        for candidate in names[first_index + 1:]:
            delta = distributions[candidate] - distributions[reference]
            delta_rows.append({
                "reference": reference,
                "candidate": candidate,
                "delta_mae_candidate_minus_reference": float(
                    point[candidate]["mae"] - point[reference]["mae"]
                ),
                "delta_mae_ci95_low": float(np.quantile(delta, 0.025)),
                "delta_mae_ci95_high": float(np.quantile(delta, 0.975)),
                "probability_candidate_lower_mae": float(np.mean(delta < 0)),
                "bootstrap_resamples": int(resamples),
            })
    return pd.DataFrame(ci_rows), pd.DataFrame(delta_rows)
