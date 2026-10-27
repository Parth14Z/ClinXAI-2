"""Predictions, probability ranges, persistence and feature ordering."""
import numpy as np
import pytest

from src import config
from src.metrics import classification_metrics
from src.model import (build_baseline_pipeline, ensure_feature_frame, load_pipeline, predict_margin,
                       predict_positive_proba, save_pipeline)


def test_probabilities_in_range_and_sum_to_one(fitted, split):
    proba = fitted.predict_proba(split.X_test)
    assert proba.shape == (len(split.X_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_positive_class_is_column_one(fitted):
    assert list(fitted.classes_) == [0, 1]


def test_margin_sigmoid_equals_probability(fitted, split):
    margin = predict_margin(fitted, split.X_test)
    np.testing.assert_allclose(1 / (1 + np.exp(-margin)), predict_positive_proba(fitted, split.X_test),
                               atol=1e-5)


def test_model_beats_chance_on_holdout(fitted, split):
    m = classification_metrics(split.y_test, predict_positive_proba(fitted, split.X_test))
    assert 0.70 < m["roc_auc"] < 0.95  # sanity band, not a performance claim


def test_save_load_roundtrip(tmp_path, fitted, split):
    path = tmp_path / "m.joblib"
    save_pipeline(fitted, path)
    loaded = load_pipeline(path)
    np.testing.assert_array_equal(loaded.predict_proba(split.X_test), fitted.predict_proba(split.X_test))


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "absent.joblib")


def test_model_feature_names_match_config(fitted):
    booster = fitted.named_steps["model"].get_booster()
    assert booster.feature_names == config.FEATURES


def test_reordered_columns_are_realigned(fitted, split):
    shuffled = split.X_test[list(reversed(config.FEATURES))]
    np.testing.assert_array_equal(predict_positive_proba(fitted, shuffled),
                                  predict_positive_proba(fitted, split.X_test))


def test_wrong_width_array_rejected():
    with pytest.raises(ValueError):
        ensure_feature_frame(np.zeros((1, 5)))


def test_baseline_trains_and_predicts(split):
    base = build_baseline_pipeline().fit(split.X_train, split.y_train)
    p = base.predict_proba(split.X_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.2, 1.4])


def test_metrics_confusion_matrix_counts():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.3])
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == 0.5
