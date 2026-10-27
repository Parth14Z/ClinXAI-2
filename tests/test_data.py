"""Dataset loading, provenance, label encoding and train/test separation."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (DatasetError, encode_labels, file_md5, load_dataset, parse_arff, split_dataset,
                      zero_placeholder_counts)


def test_raw_file_matches_openml_checksum():
    assert file_md5(config.RAW_ARFF_PATH) == config.OPENML_MD5


def test_expected_columns_and_order(df):
    assert list(df.columns) == [config.ID_COLUMN, *config.FEATURES, config.TARGET]
    assert config.FEATURES == ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                               "BMI", "DiabetesPedigreeFunction", "Age"]


def test_shape_and_class_counts(df):
    assert len(df) == config.EXPECTED_ROWS
    assert df[config.TARGET].value_counts().to_dict() == config.EXPECTED_CLASS_COUNTS


def test_label_encoding_maps_named_classes():
    s = pd.Series(["tested_positive", "tested_negative", "tested_positive"])
    assert encode_labels(s).tolist() == [1, 0, 1]


def test_label_encoding_rejects_unknown_label():
    with pytest.raises(DatasetError):
        encode_labels(pd.Series(["tested_positive", "maybe"]))


def test_first_record_matches_source(df):
    first = df.iloc[0]
    assert first["Glucose"] == 148 and first["BMI"] == pytest.approx(33.6) and first[config.TARGET] == 1


def test_zero_placeholders_are_counted(df):
    counts = zero_placeholder_counts(df)
    assert counts == {"Glucose": 5, "BloodPressure": 35, "SkinThickness": 227, "Insulin": 374, "BMI": 11}


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "nope.arff")


def test_malformed_arff_raises(tmp_path):
    bad = tmp_path / "bad.arff"
    bad.write_text("@relation x\n@attribute 'a' real\n@attribute 'b' real\n@data\n1,2\n3\n")
    with pytest.raises(DatasetError):
        parse_arff(bad)


def test_wrong_schema_raises(tmp_path):
    bad = tmp_path / "schema.arff"
    bad.write_text("@relation x\n@attribute 'a' real\n@attribute 'class' {p,n}\n@data\n1,p\n")
    with pytest.raises(DatasetError):
        load_dataset(bad, verify_checksum=False)


def test_split_sizes_and_disjoint_ids(split):
    assert len(split.X_train) == 614 and len(split.X_test) == 154
    assert set(split.ids_train).isdisjoint(split.ids_test)
    assert len(set(split.ids_train) | set(split.ids_test)) == config.EXPECTED_ROWS


def test_split_is_stratified(split, df):
    overall = df[config.TARGET].mean()
    assert abs(split.y_train.mean() - overall) < 0.01
    assert abs(split.y_test.mean() - overall) < 0.01


def test_split_is_reproducible(df, split):
    again = split_dataset(df)
    np.testing.assert_array_equal(again.ids_test, split.ids_test)


def test_split_rows_match_source_by_id(df, split):
    src = df.set_index(config.ID_COLUMN)
    for i in (0, 10, 100):
        pid = split.ids_test[i]
        pd.testing.assert_series_equal(split.X_test.iloc[i], src.loc[pid, config.FEATURES],
                                       check_names=False)
