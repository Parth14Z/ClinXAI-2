"""Central, reproducible configuration for ClinXAI (diabetes edition).

Every script and module imports paths, seeds and feature definitions from here
so that training, evaluation, testing and the dashboard always agree on the
feature order and the experimental settings.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
CASE_STUDIES_DIR = OUTPUTS_DIR / "case_studies"

RAW_ARFF_PATH = RAW_DIR / "diabetes_openml37.arff"
PROVENANCE_PATH = RAW_DIR / "provenance.json"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
FULL_CSV = PROCESSED_DIR / "pima_diabetes_clean_names.csv"

XGB_MODEL_PATH = MODELS_DIR / "xgb_pipeline.joblib"
BASELINE_MODEL_PATH = MODELS_DIR / "logreg_baseline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------- #
# Dataset provenance (OpenML dataset 37 = Pima Indians Diabetes Database)
# --------------------------------------------------------------------------- #
OPENML_DATASET_ID = 37
OPENML_ARFF_URL = "https://openml.org/data/v1/download/37/diabetes.arff"
OPENML_META_URL = "https://www.openml.org/api/v1/json/data/37"
# MD5 published by OpenML for file_id 37; a mismatch aborts training.
OPENML_MD5 = "3cbaa3e54586aa88cf6aacb4033e4470"
EXPECTED_ROWS = 768
EXPECTED_CLASS_COUNTS = {0: 500, 1: 268}

# ARFF short names -> descriptive names used everywhere else (order matters).
ARFF_TO_FEATURE = {
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
}
FEATURES: list[str] = list(ARFF_TO_FEATURE.values())
TARGET = "Outcome"
ID_COLUMN = "patient_id"
LABEL_MAP = {"tested_negative": 0, "tested_positive": 1}
CLASS_NAMES = ["Diabetes-negative (0)", "Diabetes-positive (1)"]

# Zero is physiologically impossible for these measurements, so the source's
# zeros are treated as missing.  A zero pregnancy count is valid and is kept.
ZERO_AS_MISSING: list[str] = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Original definitions from the dataset documentation.  Units for glucose are
# not stated in the source; mg/dL is the conventional unit for a 2-h OGTT.
FEATURE_DEFINITIONS = {
    "Pregnancies": ("Number of times pregnant", "count"),
    "Glucose": ("Plasma glucose concentration 2 hours into an oral glucose tolerance test",
                "mg/dL (conventional; unit not stated in source)"),
    "BloodPressure": ("Diastolic blood pressure", "mm Hg"),
    "SkinThickness": ("Triceps skin fold thickness", "mm"),
    "Insulin": ("2-hour serum insulin", "µU/mL"),
    "BMI": ("Body mass index (weight in kg / (height in m)²)", "kg/m²"),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function (family-history-based score)", "unitless"),
    "Age": ("Age", "years"),
}

# Plausible input bounds for the dashboard's what-if editor.  These are
# sanity limits for rejecting typing errors, not clinical reference ranges.
INPUT_BOUNDS = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 1000),
    "BMI": (0, 80),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (18, 100),
}

# --------------------------------------------------------------------------- #
# Experimental settings
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.5  # fixed a priori; never tuned on the test set

# Small grid searched with stratified CV on the training split only.
XGB_PARAM_GRID = {
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.03, 0.1],
    "model__n_estimators": [100, 250],
    "model__min_child_weight": [1, 5],
}
XGB_FIXED_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 2,
}

LIME_NUM_SAMPLES = 5000
LIME_PRIMARY_SEED = RANDOM_SEED
LIME_STABILITY_SEEDS: tuple[int, ...] = tuple(range(10))

# Confidence bands on P(diabetes-positive)
HIGH_CONFIDENCE_POS = 0.85
HIGH_CONFIDENCE_NEG = 0.15
BORDERLINE_LOW, BORDERLINE_HIGH = 0.40, 0.60

# Minimum magnitudes before a sign difference counts as a real disagreement
LIME_SIGN_TOL = 0.01   # probability units of the LIME surrogate
SHAP_SIGN_TOL = 0.05   # log-odds units of TreeSHAP

# --------------------------------------------------------------------------- #
# Visual identity (teal / white / muted purple)
# --------------------------------------------------------------------------- #
# teal/purple pair validated for colour-vision deficiency separation
# (worst-case deutan dE 12.3, normal-vision dE 21.7, both >= 3:1 on white).
PALETTE = {
    "teal": "#008F86",
    "teal_dark": "#0B5F5A",
    "teal_light": "#7FCBC4",
    "teal_pale": "#E8F5F4",
    "purple": "#9A78DC",
    "purple_dark": "#5E4A91",
    "purple_light": "#C9B8EE",
    "purple_pale": "#F3EFFB",
    "slate": "#273444",
    "grey": "#8C98A8",
    "grid": "#E3E8EE",
    "white": "#FFFFFF",
}
# Contributions pushing toward diabetes-positive are purple; toward negative, teal.
RISK_UP_COLOR = PALETTE["purple"]
RISK_DOWN_COLOR = PALETTE["teal"]


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
              EXPERIMENTS_DIR, CASE_STUDIES_DIR):
        d.mkdir(parents=True, exist_ok=True)
