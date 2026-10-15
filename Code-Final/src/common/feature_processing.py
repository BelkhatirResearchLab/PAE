"""Leakage-safe feature selectors used inside scikit-learn pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold


class SafeLassoSelector(BaseEstimator, TransformerMixin):
    """Select PCA components with non-zero Lasso coefficients."""

    def __init__(
        self,
        cv_splits: int = 3,
        random_state: int = 42,
        alphas: tuple[float, ...] = tuple(np.logspace(-3, 1, 16)),
        max_iter: int = 100_000,
        n_jobs: int = 1,
    ) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.alphas = alphas
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, x, y):
        """Fit Lasso only on the samples supplied by the enclosing fold."""
        values = np.asarray(x, dtype=float)
        outcome = np.asarray(y, dtype=float)
        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.estimator_ = LassoCV(
            alphas=np.asarray(self.alphas, dtype=float),
            cv=cv,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            tol=1e-3,
        ).fit(values, outcome)
        if int(self.estimator_.n_iter_) >= self.max_iter:
            raise RuntimeError("Selected baseline LASSO did not converge")
        coefficients = np.abs(self.estimator_.coef_)
        self.support_ = coefficients > 1e-10
        if not self.support_.any():
            self.support_[int(np.argmax(coefficients))] = True
        self.selected_alpha_ = float(self.estimator_.alpha_)
        self.n_features_in_ = int(values.shape[1])
        self.n_features_out_ = int(self.support_.sum())
        return self

    def transform(self, x):
        """Keep the components selected during fit."""
        return np.asarray(x, dtype=float)[:, self.support_]

    def get_support(self):
        """Return the fitted Boolean support mask."""
        return self.support_


class SafeL1LogisticSelector(BaseEstimator, TransformerMixin):
    """Select PCA components with an L1-regularised logistic model."""

    def __init__(
        self,
        cv_splits: int = 3,
        random_state: int = 42,
        cs: tuple[float, ...] = tuple(np.logspace(-3, 1, 9)),
        max_iter: int = 20_000,
        n_jobs: int = 1,
    ) -> None:
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.cs = cs
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, x, y):
        """Fit an L1 selector only on the samples supplied by the fold."""
        values = np.asarray(x, dtype=float)
        outcome = np.asarray(y, dtype=int)
        cv = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.estimator_ = LogisticRegressionCV(
            Cs=np.asarray(self.cs, dtype=float),
            cv=cv,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            refit=True,
        ).fit(values, outcome)
        coefficients = np.abs(self.estimator_.coef_[0])
        self.support_ = coefficients > 1e-10
        if not self.support_.any():
            self.support_[int(np.argmax(coefficients))] = True
        self.selected_c_ = float(self.estimator_.C_[0])
        self.n_features_in_ = int(values.shape[1])
        self.n_features_out_ = int(self.support_.sum())
        return self

    def transform(self, x):
        """Keep the components selected during fit."""
        return np.asarray(x, dtype=float)[:, self.support_]

    def get_support(self):
        """Return the fitted Boolean support mask."""
        return self.support_
