"""Dataset acquisition, provenance, parsing, validation and splitting.

The Pima Indians Diabetes Database is obtained from OpenML (dataset 37), which
re-hosts the UCI copy donated by the Johns Hopkins APL on behalf of the
National Institute of Diabetes and Digestive and Kidney Diseases.  The file is
verified against OpenML's published MD5; if it cannot be obtained or verified,
loading fails loudly.  No synthetic substitute is ever generated.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class DatasetError(RuntimeError):
    """Raised when the authentic dataset is missing, corrupted or malformed."""


# --------------------------------------------------------------------------- #
# Acquisition and provenance
# --------------------------------------------------------------------------- #
def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_dataset(dest: Path = config.RAW_ARFF_PATH, timeout: int = 60) -> Path:
    """Download the ARFF from OpenML and verify its checksum.

    Raises DatasetError instead of falling back to any other data.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(config.OPENML_ARFF_URL, timeout=timeout) as resp:
            payload = resp.read()
    except Exception as exc:  # network failure, HTTP error, ...
        raise DatasetError(
            f"Could not download the Pima dataset from {config.OPENML_ARFF_URL}: {exc}. "
            "Training is stopped; no synthetic data will be substituted."
        ) from exc
    digest = hashlib.md5(payload).hexdigest()
    if digest != config.OPENML_MD5:
        raise DatasetError(f"Checksum mismatch (got {digest}, expected {config.OPENML_MD5}).")
    dest.write_bytes(payload)
    return dest


def ensure_dataset(path: Path = config.RAW_ARFF_PATH) -> Path:
    """Return a verified local copy of the dataset, downloading it if absent."""
    if not path.exists():
        download_dataset(path)
    digest = file_md5(path)
    if digest != config.OPENML_MD5:
        raise DatasetError(
            f"{path} does not match OpenML's published MD5 ({digest} != {config.OPENML_MD5})."
        )
    return path


def write_provenance(path: Path = config.RAW_ARFF_PATH,
                     out: Path = config.PROVENANCE_PATH) -> dict:
    """Write a machine-readable provenance record next to the raw file."""
    record = {
        "dataset": "Pima Indians Diabetes Database",
        "source": "OpenML dataset 37 (version 1)",
        "download_url": config.OPENML_ARFF_URL,
        "metadata_url": config.OPENML_META_URL,
        "original_source": "UCI Machine Learning Repository (Pima Indians Diabetes Database); "
                           "the dataset is no longer listed on the current UCI site",
        "original_owner": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
        "donor": "Vincent Sigillito, Applied Physics Laboratory, The Johns Hopkins University (1990)",
        "openml_licence_field": "Public",
        "citation": "Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. "
                    "(1988). Using the ADAP learning algorithm to forecast the onset of diabetes "
                    "mellitus. Proc. Symposium on Computer Applications and Medical Care, 261-265.",
        "population": "Females aged >= 21 years of Pima Indian heritage living near Phoenix, "
                      "Arizona, USA",
        "label_definition": "tested_positive = signs of diabetes by WHO criteria (2-h post-load "
                            "plasma glucose >= 200 mg/dL at a survey examination, or found during "
                            "routine medical care)",
        "md5_expected": config.OPENML_MD5,
        "md5_actual": file_md5(path),
        "sha256": file_sha256(path),
        "retrieved_on": date.today().isoformat(),
    }
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #
def parse_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader for this numeric + one-nominal-class file."""
    attributes: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for raw in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        lower = line.lower()
        if lower.startswith("@attribute"):
            name = line.split(None, 2)[1].strip("'\"")
            attributes.append(name)
        elif lower.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append([v.strip() for v in line.split(",")])
    if not attributes or not rows:
        raise DatasetError(f"{path} is not a valid ARFF file (no attributes or no data).")
    bad = [i for i, r in enumerate(rows) if len(r) != len(attributes)]
    if bad:
        raise DatasetError(f"Rows with wrong field count at data lines {bad[:5]}.")
    return pd.DataFrame(rows, columns=attributes)


def encode_labels(labels: pd.Series) -> pd.Series:
    """Map 'tested_positive' -> 1 and 'tested_negative' -> 0, rejecting anything else."""
    unknown = sorted(set(labels.unique()) - set(config.LABEL_MAP))
    if unknown:
        raise DatasetError(f"Unexpected class labels: {unknown}")
    return labels.map(config.LABEL_MAP).astype(int)


def load_dataset(path: Path | None = None, verify_checksum: bool = True) -> pd.DataFrame:
    """Load, rename, type-check and validate the dataset.

    Returns a DataFrame with columns [patient_id, *FEATURES, Outcome].  The
    patient_id is the zero-based row position in the original file so that
    every explained patient can be traced back to the source record.
    """
    path = Path(path) if path is not None else config.RAW_ARFF_PATH
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}. Run `python train_model.py`.")
    if verify_checksum and path == config.RAW_ARFF_PATH:
        ensure_dataset(path)

    raw = parse_arff(path)
    expected = list(config.ARFF_TO_FEATURE) + ["class"]
    if list(raw.columns) != expected:
        raise DatasetError(f"Unexpected columns {list(raw.columns)}; expected {expected}.")

    df = raw.rename(columns=config.ARFF_TO_FEATURE)
    try:
        df[config.FEATURES] = df[config.FEATURES].astype(float)
    except ValueError as exc:
        raise DatasetError(f"Non-numeric feature values found: {exc}") from exc
    df[config.TARGET] = encode_labels(df.pop("class"))
    df.insert(0, config.ID_COLUMN, np.arange(len(df)))
    validate_dataframe(df)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Structural checks that guard every downstream step."""
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise DatasetError(f"Missing columns: {missing}")
    if df[config.FEATURES].isna().any().any():
        raise DatasetError("Raw data unexpectedly contains NaN values.")
    if (df[config.FEATURES] < 0).any().any():
        raise DatasetError("Negative feature values found.")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise DatasetError("Outcome must be binary 0/1.")


def zero_placeholder_counts(df: pd.DataFrame) -> dict[str, int]:
    """Number of zero placeholders per feature treated as missing."""
    return {c: int((df[c] == 0).sum()) for c in config.ZERO_AS_MISSING}


def dataset_summary(df: pd.DataFrame) -> dict:
    """Descriptive summary saved as report evidence."""
    desc = df[config.FEATURES].describe().T.round(3)
    counts = df[config.TARGET].value_counts().sort_index()
    return {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURES),
        "features": config.FEATURES,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "zero_placeholders": zero_placeholder_counts(df),
        "zero_placeholder_pct": {c: round(100 * n / len(df), 2)
                                 for c, n in zero_placeholder_counts(df).items()},
        "describe": desc.to_dict(orient="index"),
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: np.ndarray
    ids_test: np.ndarray


def split_dataset(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                  seed: int = config.RANDOM_SEED) -> DataSplit:
    """Stratified, reproducible train/test split that keeps patient ids."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=seed
    )
    return DataSplit(
        X_train=train_df[config.FEATURES].reset_index(drop=True),
        X_test=test_df[config.FEATURES].reset_index(drop=True),
        y_train=train_df[config.TARGET].reset_index(drop=True),
        y_test=test_df[config.TARGET].reset_index(drop=True),
        ids_train=train_df[config.ID_COLUMN].to_numpy(),
        ids_test=test_df[config.ID_COLUMN].to_numpy(),
    )


def save_split(split: DataSplit) -> None:
    for X, y, ids, path in ((split.X_train, split.y_train, split.ids_train, config.TRAIN_CSV),
                            (split.X_test, split.y_test, split.ids_test, config.TEST_CSV)):
        out = X.copy()
        out.insert(0, config.ID_COLUMN, ids)
        out[config.TARGET] = y.to_numpy()
        out.to_csv(path, index=False)


def load_split_csv(path: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Read a saved split; returns (X, y, patient_ids) with canonical feature order."""
    if not Path(path).exists():
        raise DatasetError(f"{path} not found. Run `python train_model.py` first.")
    df = pd.read_csv(path)
    return (df[config.FEATURES].astype(float), df[config.TARGET].astype(int),
            df[config.ID_COLUMN].to_numpy())


def format_patient_id(pid: int) -> str:
    return f"PIMA-{int(pid):04d}"
