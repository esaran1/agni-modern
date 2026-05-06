"""XGBoost wrappers for occurrence and severity tasks.

All wrappers enforce early stopping via the validation set so the model
does not train for more rounds than the data supports.  Default
``early_stopping_rounds=20`` can be overridden via params.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from agni_modern.models.base import ModelWrapper

_DEFAULT_EARLY_STOPPING = 20


class XGBoostOccurrenceWrapper(ModelWrapper):
    """Binary occurrence model wrapper with early stopping."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.early_stopping_rounds = int(self.params.pop("early_stopping_rounds", _DEFAULT_EARLY_STOPPING))
        self.model = xgb.XGBClassifier(
            early_stopping_rounds=self.early_stopping_rounds, **self.params,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict[str, object]) -> None:
        feature_cols = config["feature_cols"]
        target_col = config["target_col"]
        self.model.fit(
            train_df[feature_cols],
            train_df[target_col],
            eval_set=[(val_df[feature_cols], val_df[target_col])],
            verbose=False,
        )

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: Path) -> "XGBoostOccurrenceWrapper":
        instance = cls.__new__(cls)
        with path.open("rb") as handle:
            instance.model = pickle.load(handle)
        instance.params = {}
        instance.early_stopping_rounds = _DEFAULT_EARLY_STOPPING
        return instance


class XGBoostSeverityClassifierWrapper(ModelWrapper):
    """Multi-class severity classifier with early stopping."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.early_stopping_rounds = int(self.params.pop("early_stopping_rounds", _DEFAULT_EARLY_STOPPING))
        self.model = xgb.XGBClassifier(
            early_stopping_rounds=self.early_stopping_rounds, **self.params,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict[str, object]) -> None:
        feature_cols = config["feature_cols"]
        target_col = config["target_col"]
        train_mask = train_df["y_sev_available"] == 1
        val_mask = val_df["y_sev_available"] == 1
        self.model.fit(
            train_df.loc[train_mask, feature_cols],
            train_df.loc[train_mask, target_col],
            eval_set=[(val_df.loc[val_mask, feature_cols], val_df.loc[val_mask, target_col])],
            verbose=False,
        )

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: Path) -> "XGBoostSeverityClassifierWrapper":
        instance = cls.__new__(cls)
        with path.open("rb") as handle:
            instance.model = pickle.load(handle)
        instance.params = {}
        instance.early_stopping_rounds = _DEFAULT_EARLY_STOPPING
        return instance


class XGBoostSeverityRegressorWrapper(ModelWrapper):
    """Severity regression wrapper with early stopping."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = dict(params or {})
        self.early_stopping_rounds = int(self.params.pop("early_stopping_rounds", _DEFAULT_EARLY_STOPPING))
        self.model = xgb.XGBRegressor(
            early_stopping_rounds=self.early_stopping_rounds, **self.params,
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict[str, object]) -> None:
        feature_cols = config["feature_cols"]
        target_col = config["target_col"]
        train_mask = train_df["y_sev_available"] == 1
        val_mask = val_df["y_sev_available"] == 1
        self.model.fit(
            train_df.loc[train_mask, feature_cols],
            train_df.loc[train_mask, target_col],
            eval_set=[(val_df.loc[val_mask, feature_cols], val_df.loc[val_mask, target_col])],
            verbose=False,
        )

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)

    def predict_proba(self, df: pd.DataFrame):
        return self.predict(df)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)

    @classmethod
    def load(cls, path: Path) -> "XGBoostSeverityRegressorWrapper":
        instance = cls.__new__(cls)
        with path.open("rb") as handle:
            instance.model = pickle.load(handle)
        instance.params = {}
        instance.early_stopping_rounds = _DEFAULT_EARLY_STOPPING
        return instance
