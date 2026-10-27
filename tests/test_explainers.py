"""LIME and TreeSHAP generation, output spaces and SHAP additivity."""
import numpy as np
import pytest

from src import config
from src import lime_explainer as lx
from src import shap_explainer as sx
from src.model import predict_margin, transform_features


# ------------------------------- LIME ------------------------------------- #
def test_lime_explains_positive_class_for_all_features(fitted, split, lime_expl):
    x_t = transform_features(fitted, split.X_test.iloc[[0]])
    res = lx.explain_patient(fitted, lime_expl, x_t, num_samples=1000, seed=config.LIME_PRIMARY_SEED)
    assert set(res.weights) == set(config.FEATURES) == set(res.rules)
    assert res.model_prob == pytest.approx(fitted.predict_proba(split.X_test.iloc[[0]])[0, 1])
    assert np.isfinite(res.r2) and res.r2 <= 1.0
    assert 0 <= res.fidelity_gap <= 1


def test_lime_predict_fn_is_the_pipeline_positive_probability(fitted, split):
    fn = lx.RecordingPredictFn(fitted)
    x_t = transform_features(fitted, split.X_test.iloc[:5]).to_numpy()
    out = fn(x_t)
    np.testing.assert_allclose(out[:, 1], fitted.predict_proba(split.X_test.iloc[:5])[:, 1], atol=1e-6)
    assert fn.last_batch.shape == (5, 8)


def test_lime_neighbourhood_is_recorded(fitted, split, lime_expl):
    x_t = transform_features(fitted, split.X_test.iloc[[3]])
    res = lx.explain_patient(fitted, lime_expl, x_t, num_samples=800, keep_neighborhood=True)
    assert res.neighborhood_X.shape == (800, 8)
    np.testing.assert_allclose(res.neighborhood_X[0], x_t.to_numpy()[0])  # first row = the patient
    assert 0 <= res.neighborhood_summary()["share_predicted_positive"] <= 1


def test_lime_same_seed_is_deterministic(fitted, split, X_train_t):
    x_t = transform_features(fitted, split.X_test.iloc[[1]])
    a = lx.explain_patient(fitted, lx.build_lime_explainer(X_train_t, 7), x_t, 1000)
    b = lx.explain_patient(fitted, lx.build_lime_explainer(X_train_t, 7), x_t, 1000)
    assert a.weights == b.weights


def test_lime_stability_metrics(fitted, split, X_train_t):
    x_t = transform_features(fitted, split.X_test.iloc[[2]])
    stab = lx.lime_stability(fitted, X_train_t, x_t, seeds=(0, 1, 2), num_samples=800)
    assert stab.weights.shape == (3, 8) and list(stab.weights.columns) == config.FEATURES
    m = stab.metrics
    for k in ("stability_jaccard_top3", "sign_consistency", "top1_agreement"):
        assert 0 <= m[k] <= 1
    assert set(m["material_sign_flip_features"]) <= set(m["sign_flip_features"])


def test_lime_rejects_bad_inputs(fitted, split, lime_expl, X_train_t):
    with pytest.raises(ValueError):
        lx.explain_patient(fitted, lime_expl, transform_features(fitted, split.X_test.iloc[:2]), 1000)
    with pytest.raises(ValueError):
        lx.explain_patient(fitted, lime_expl, transform_features(fitted, split.X_test.iloc[[0]]), 10)
    with pytest.raises(ValueError):
        lx.build_lime_explainer(split.X_train.replace(0, np.nan))


# ------------------------------- SHAP ------------------------------------- #
def test_shap_shapes_and_feature_names(fitted, split):
    tree = sx.build_tree_explainer(fitted)
    exp = sx.explain(tree, fitted, split.X_test)
    assert exp.values.shape == (len(split.X_test), 8)
    assert list(exp.feature_names) == config.FEATURES


def test_shap_additivity_in_logodds(fitted, split):
    tree = sx.build_tree_explainer(fitted)
    exp = sx.explain(tree, fitted, split.X_test)
    margin = predict_margin(fitted, split.X_test)
    np.testing.assert_allclose(exp.base_values + exp.values.sum(axis=1), margin, atol=1e-4)
    checks = sx.verify_additivity(exp, fitted, split.X_test)
    assert checks["additive_in_logodds"] and checks["probability_recovered_via_sigmoid"]


def test_shap_values_are_not_probability_contributions(fitted, split):
    """Adding log-odds SHAP values to a base probability must NOT reproduce P."""
    tree = sx.build_tree_explainer(fitted)
    checks = sx.verify_additivity(sx.explain(tree, fitted, split.X_test), fitted, split.X_test)
    assert checks["max_abs_error_naive_probability_sum"] > 0.1


def test_shap_patient_explanation(fitted, split):
    tree = sx.build_tree_explainer(fitted)
    sp = sx.explain_patient(tree, fitted, split.X_test.iloc[[0]])
    assert sp.additivity_error < 1e-4
    assert sp.probability == pytest.approx(fitted.predict_proba(split.X_test.iloc[[0]])[0, 1], abs=1e-5)
    assert list(sp.table()["feature"].sort_values()) == sorted(config.FEATURES)


def test_shap_marks_imputed_values(fitted, split):
    tree = sx.build_tree_explainer(fitted)
    i = int(np.flatnonzero((split.X_test["Insulin"] == 0).to_numpy())[0])
    exp = sx.explain(tree, fitted, split.X_test.iloc[[i]])
    assert "(imputed)" in exp.display_data[0][config.FEATURES.index("Insulin")]


def test_lime_and_shap_explain_the_same_input(fitted, split):
    """Both explainers receive the identical preprocessed vector."""
    x_raw = split.X_test.iloc[[5]]
    tree = sx.build_tree_explainer(fitted)
    exp = sx.explain(tree, fitted, x_raw)
    np.testing.assert_allclose(exp.data[0], transform_features(fitted, x_raw).to_numpy()[0])
