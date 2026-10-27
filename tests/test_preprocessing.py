"""Missing-value handling, training-only imputation, feature order, input validation."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import ZeroAsMissing, build_preprocessor, fitted_medians, validate_patient_input


def test_zeros_become_nan_only_in_selected_columns(split):
    out = ZeroAsMissing(columns=config.ZERO_AS_MISSING).fit(split.X_train).transform(split.X_train)
    for c in config.ZERO_AS_MISSING:
        assert out[c].isna().sum() == (split.X_train[c] == 0).sum()
        assert not (out[c] == 0).any()
    # Zero pregnancies is a valid value and must be kept.
    assert (out["Pregnancies"] == 0).sum() == (split.X_train["Pregnancies"] == 0).sum()
    assert out["Pregnancies"].isna().sum() == 0


def test_imputer_uses_training_medians_only(split):
    pre = build_preprocessor().fit(split.X_train)
    medians = fitted_medians(pre)
    for c in config.ZERO_AS_MISSING:
        expected = split.X_train[c].replace(0, np.nan).median()
        assert medians[c] == pytest.approx(expected)


def test_test_rows_imputed_with_training_statistics(split):
    pre = build_preprocessor().fit(split.X_train)
    out = pre.transform(split.X_test)
    medians = fitted_medians(pre)
    missing = split.X_test["Insulin"] == 0
    assert missing.any()
    assert np.allclose(out.loc[missing, "Insulin"], medians["Insulin"])
    # Fitting on test data instead would give a different median.
    test_median = split.X_test["Insulin"].replace(0, np.nan).median()
    assert test_median != pytest.approx(medians["Insulin"])


def test_no_nan_after_preprocessing_and_order_preserved(split):
    out = build_preprocessor().fit(split.X_train).transform(split.X_test)
    assert list(out.columns) == config.FEATURES
    assert not out.isna().any().any()


def test_preprocessing_is_idempotent(split):
    pre = build_preprocessor().fit(split.X_train)
    once = pre.transform(split.X_test)
    twice = pre.transform(once)
    pd.testing.assert_frame_equal(once, twice)


def test_feature_order_mismatch_rejected(split):
    pre = build_preprocessor().fit(split.X_train)
    with pytest.raises(ValueError):
        pre.transform(split.X_test[list(reversed(config.FEATURES))])


VALID = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80,
         "BMI": 30.5, "DiabetesPedigreeFunction": 0.4, "Age": 35}


def test_valid_input_returns_ordered_frame():
    out = validate_patient_input(dict(reversed(list(VALID.items()))))
    assert list(out.columns) == config.FEATURES and out.shape == (1, 8)


@pytest.mark.parametrize("patch, msg", [
    ({"Glucose": "high"}, "numeric"),
    ({"Glucose": float("nan")}, "finite"),
    ({"BMI": float("inf")}, "finite"),
    ({"Age": 5}, "outside"),
    ({"Insulin": -3}, "outside"),
    ({"Glucose": None}, "numeric"),
    ({"Pregnancies": True}, "boolean"),
])
def test_invalid_values_rejected(patch, msg):
    with pytest.raises(ValueError, match=msg):
        validate_patient_input({**VALID, **patch})


def test_missing_and_unknown_fields_rejected():
    partial = dict(VALID)
    partial.pop("Age")
    with pytest.raises(ValueError, match="Missing"):
        validate_patient_input(partial)
    with pytest.raises(ValueError, match="Unknown"):
        validate_patient_input({**VALID, "Cholesterol": 5})
    with pytest.raises(ValueError):
        validate_patient_input([1, 2, 3])
