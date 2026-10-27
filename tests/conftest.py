"""Shared fixtures.

Fixtures train a small model from the authentic dataset so that unit tests do
not depend on artefacts produced by train_model.py.  Only the dashboard test
requires those saved artefacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import load_dataset, split_dataset  # noqa: E402
from src.model import build_xgb_pipeline, transform_features  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_dataset()


@pytest.fixture(scope="session")
def split(df):
    return split_dataset(df)


@pytest.fixture(scope="session")
def fitted(split):
    pipe = build_xgb_pipeline(n_estimators=60, max_depth=3, learning_rate=0.1)
    pipe.fit(split.X_train, split.y_train)
    return pipe


@pytest.fixture(scope="session")
def X_train_t(fitted, split):
    return transform_features(fitted, split.X_train)


@pytest.fixture(scope="session")
def lime_expl(X_train_t):
    from src.lime_explainer import build_lime_explainer
    return build_lime_explainer(X_train_t, config.LIME_PRIMARY_SEED)
