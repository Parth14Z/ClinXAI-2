"""Training-only preprocessing and input validation.

Pipeline:  ZeroAsMissing  ->  SimpleImputer(median)
Both steps are fitted on the training split only (and, inside cross-validation,
on each training fold only).  Feature names and order are preserved, so the
model, LIME and SHAP all see the same eight columns in the same order.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from . import config


class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Replace physiologically impossible zeros with NaN in selected columns."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check(X).copy()
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"Feature order mismatch: {list(X.columns)}")
        cols = self.columns if self.columns is not None else config.ZERO_AS_MISSING
        X[cols] = X[cols].mask(X[cols] == 0)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

    @staticmethod
    def _check(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X, dtype=float), columns=config.FEATURES)
        return X


def build_preprocessor() -> Pipeline:
    """Unfitted preprocessing pipeline that outputs named pandas columns."""
    return Pipeline([
        ("zero_to_nan", ZeroAsMissing(columns=list(config.ZERO_AS_MISSING))),
        ("impute", SimpleImputer(strategy="median")),
    ]).set_output(transform="pandas")


def fitted_medians(preprocessor: Pipeline) -> dict[str, float]:
    """Training medians learned by the imputer (report evidence)."""
    imputer = preprocessor.named_steps["impute"]
    return {f: float(v) for f, v in zip(config.FEATURES, imputer.statistics_)}


def imputed_mask(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame marking which raw values will be imputed."""
    mask = pd.DataFrame(False, index=X_raw.index, columns=config.FEATURES)
    for c in config.ZERO_AS_MISSING:
        mask[c] = X_raw[c] == 0
    return mask


def preprocessing_summary(preprocessor: Pipeline, X_train: pd.DataFrame,
                          X_test: pd.DataFrame) -> dict:
    return {
        "zero_as_missing_columns": config.ZERO_AS_MISSING,
        "imputation": "median, fitted on training split only",
        "train_medians": fitted_medians(preprocessor),
        "train_missing_counts": {c: int((X_train[c] == 0).sum()) for c in config.ZERO_AS_MISSING},
        "test_missing_counts": {c: int((X_test[c] == 0).sum()) for c in config.ZERO_AS_MISSING},
        "feature_order": config.FEATURES,
        "scaling": "none for XGBoost (tree splits are scale-invariant); "
                   "StandardScaler for the logistic-regression baseline only",
    }


# --------------------------------------------------------------------------- #
# Input validation for user-edited profiles
# --------------------------------------------------------------------------- #
def validate_patient_input(values: Mapping[str, object]) -> pd.DataFrame:
    """Validate a single patient profile and return a 1-row frame in model order.

    Raises ValueError with a readable message for missing/extra fields,
    non-numeric, non-finite, or out-of-bounds values.
    """
    if not isinstance(values, Mapping):
        raise ValueError("Patient input must be a mapping of feature name to value.")
    missing = [f for f in config.FEATURES if f not in values]
    extra = [k for k in values if k not in config.FEATURES]
    if missing:
        raise ValueError(f"Missing feature(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unknown feature(s): {', '.join(map(str, extra))}")

    clean: dict[str, float] = {}
    for f in config.FEATURES:
        v = values[f]
        if isinstance(v, bool):
            raise ValueError(f"{f} must be numeric, got a boolean.")
        try:
            num = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{f} must be numeric, got {v!r}.") from None
        if not math.isfinite(num):
            raise ValueError(f"{f} must be a finite number.")
        lo, hi = config.INPUT_BOUNDS[f]
        if not lo <= num <= hi:
            raise ValueError(f"{f}={num:g} is outside the accepted range [{lo}, {hi}].")
        clean[f] = num
    return pd.DataFrame([clean], columns=config.FEATURES)
