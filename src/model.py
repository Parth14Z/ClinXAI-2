"""XGBoost and logistic-regression pipelines: training, persistence, inference.

Model selection uses stratified 5-fold cross-validation on the training split
only.  The held-out test split is touched exactly once, in evaluate.py.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_preprocessor

CV_SCORING = ["roc_auc", "accuracy", "f1", "precision", "recall"]


def build_xgb_pipeline(seed: int = config.RANDOM_SEED, **params) -> Pipeline:
    model = XGBClassifier(random_state=seed, **{**config.XGB_FIXED_PARAMS, **params})
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_baseline_pipeline(seed: int = config.RANDOM_SEED) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("model", LogisticRegression(max_iter=2000, random_state=seed)),
    ])


def cv_splitter(seed: int = config.RANDOM_SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)


def tune_xgb(X_train: pd.DataFrame, y_train: pd.Series, seed: int = config.RANDOM_SEED,
             param_grid: dict | None = None) -> GridSearchCV:
    """Grid search over a small, pre-declared grid using training-fold CV AUC."""
    search = GridSearchCV(
        build_xgb_pipeline(seed),
        param_grid=param_grid or config.XGB_PARAM_GRID,
        scoring="roc_auc",
        cv=cv_splitter(seed),
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def cross_validate_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
                            seed: int = config.RANDOM_SEED) -> dict:
    """Training-split CV summary (mean and std per metric)."""
    res = cross_validate(pipe, X, y, cv=cv_splitter(seed), scoring=CV_SCORING)
    return {m: {"mean": float(np.mean(res[f"test_{m}"])), "std": float(np.std(res[f"test_{m}"]))}
            for m in CV_SCORING}


def predict_positive_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """P(diabetes-positive) for raw (un-imputed) feature rows."""
    X = ensure_feature_frame(X)
    return pipe.predict_proba(X)[:, 1]


def predict_margin(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Raw XGBoost margin (log-odds of the positive class)."""
    Xt = transform_features(pipe, X)
    return pipe.named_steps["model"].predict(Xt, output_margin=True)


def transform_features(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessing step only (zeros -> NaN -> train medians)."""
    return pipe.named_steps["preprocess"].transform(ensure_feature_frame(X))


def ensure_feature_frame(X) -> pd.DataFrame:
    """Coerce arrays to a DataFrame and enforce canonical feature order."""
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    if isinstance(X, pd.DataFrame):
        missing = [f for f in config.FEATURES if f not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return X[config.FEATURES].astype(float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(config.FEATURES):
        raise ValueError(f"Expected {len(config.FEATURES)} features, got {arr.shape[1]}.")
    return pd.DataFrame(arr, columns=config.FEATURES)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_pipeline(pipe: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load_pipeline(path: Path) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python train_model.py` first.")
    pipe = joblib.load(path)
    names = list(pipe.named_steps["preprocess"].named_steps["zero_to_nan"].feature_names_in_)
    if names != config.FEATURES:
        raise ValueError(f"Saved model feature order {names} != {config.FEATURES}")
    return pipe


def library_versions() -> dict[str, str]:
    import lime  # noqa: F401  (imported for version only)
    import shap
    import sklearn
    import xgboost
    from importlib.metadata import version
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "lime": version("lime"),
    }


def write_metadata(search: GridSearchCV, cv_xgb: dict, cv_baseline: dict, n_train: int,
                   n_test: int, data_md5: str, path: Path = config.MODEL_METADATA_PATH,
                   train_seconds: float | None = None) -> dict:
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "XGBClassifier inside sklearn Pipeline (ZeroAsMissing -> median imputer -> XGB)",
        "baseline": "LogisticRegression (ZeroAsMissing -> median imputer -> StandardScaler -> LR)",
        "feature_order": config.FEATURES,
        "target": config.TARGET,
        "positive_class": "1 = tested_positive (diabetes-positive)",
        "decision_threshold": config.DECISION_THRESHOLD,
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "n_train": n_train,
        "n_test": n_test,
        "data_md5": data_md5,
        "param_grid": config.XGB_PARAM_GRID,
        "fixed_params": config.XGB_FIXED_PARAMS,
        "best_params": best_params,
        "best_cv_roc_auc": float(search.best_score_),
        "cv_xgb_train_split": cv_xgb,
        "cv_baseline_train_split": cv_baseline,
        "calibration": "none - probabilities are uncalibrated model scores",
        "train_seconds": train_seconds,
        "library_versions": library_versions(),
    }
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def load_metadata(path: Path = config.MODEL_METADATA_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python train_model.py` first.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self.start, 2)
