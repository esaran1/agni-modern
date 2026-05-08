"""Logistic regression and random-forest baseline wrappers.

LogReg uses Pipeline: Imputer → Winsorizer → StandardScaler → LogReg.
RF uses Pipeline: Imputer → RandomForest.
All pipelines are persisted as a single artifact so preprocessing
is always consistent between training and inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from agni_modern.models.base import ModelWrapper


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip features to fitted percentile bounds to suppress extreme outliers.

    Fitted on training data only; applied identically at inference.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.array(X, dtype=np.float64)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


class LogisticRegressionWrapper(ModelWrapper):
    """Wrapper for sklearn logistic regression with winsorization, scaling, and class balancing."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.params.setdefault("class_weight", "balanced")
        self.params.setdefault("max_iter", 2000)
        self.params.setdefault("solver", "liblinear")
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", QuantileTransformer(
                output_distribution="normal", n_quantiles=500, random_state=0,
            )),
            ("clf", LogisticRegression(**self.params)),
        ])

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict[str, object]) -> None:
        _ = val_df
        feature_cols = config["feature_cols"]
        target_col = config["target_col"]
        self.model.fit(train_df[feature_cols], train_df[target_col])

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionWrapper":
        instance = cls.__new__(cls)
        with path.open("rb") as handle:
            instance.model = pickle.load(handle)
        instance.params = {}
        return instance


class RandomForestWrapper(ModelWrapper):
    """Wrapper for sklearn random forest with imputation and class balancing."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.params.setdefault("class_weight", "balanced")
        self.params.setdefault("n_estimators", 200)
        self.params.setdefault("random_state", 42)
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**self.params)),
        ])

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict[str, object]) -> None:
        _ = val_df
        feature_cols = config["feature_cols"]
        target_col = config["target_col"]
        self.model.fit(train_df[feature_cols], train_df[target_col])

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: Path) -> "RandomForestWrapper":
        instance = cls.__new__(cls)
        with path.open("rb") as handle:
            instance.model = pickle.load(handle)
        instance.params = {}
        return instance
