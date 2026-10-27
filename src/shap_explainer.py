"""TreeSHAP explanations for the XGBoost pipeline.

Output space
------------
shap.TreeExplainer on an XGBClassifier (tree_path_dependent perturbation)
explains the *raw margin*, i.e. the log-odds of the diabetes-positive class:

    base_value + sum_j phi_j  ==  margin(x)          (exact, additive)
    P(diabetes-positive | x)  ==  sigmoid(margin(x))   (non-linear link)

SHAP values are therefore in log-odds units.  They are additive on the
log-odds scale only and must NOT be added to a probability.  All functions
here keep that distinction explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from . import config
from .model import ensure_feature_frame, predict_margin, transform_features
from .preprocessing import imputed_mask

OUTPUT_UNITS = "log-odds of diabetes-positive (raw XGBoost margin)"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def build_tree_explainer(pipe: Pipeline) -> shap.TreeExplainer:
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    if explainer.model.model_output != "raw":
        raise RuntimeError(f"Unexpected TreeSHAP output space: {explainer.model.model_output}")
    return explainer


def base_value(explainer: shap.TreeExplainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def explain(explainer: shap.TreeExplainer, pipe: Pipeline, X_raw: pd.DataFrame) -> shap.Explanation:
    """SHAP Explanation on the model's input space (after imputation).

    ``display_data`` marks imputed values so plots never pass off a training
    median as a measured value.
    """
    X_raw = ensure_feature_frame(X_raw).reset_index(drop=True)
    X_t = transform_features(pipe, X_raw)
    exp = explainer(X_t)
    mask = imputed_mask(X_raw).to_numpy()
    display = np.empty(X_t.shape, dtype=object)
    for i in range(X_t.shape[0]):
        for j in range(X_t.shape[1]):
            v = X_t.iat[i, j]
            txt = f"{v:.3f}" if config.FEATURES[j] == "DiabetesPedigreeFunction" else f"{v:.4g}"
            display[i, j] = f"{txt} (imputed)" if mask[i, j] else txt
    return shap.Explanation(
        values=np.asarray(exp.values, dtype=float),
        base_values=np.asarray(exp.base_values, dtype=float).reshape(-1),
        data=X_t.to_numpy(),
        display_data=display,
        feature_names=list(config.FEATURES),
    )


def verify_additivity(exp: shap.Explanation, pipe: Pipeline, X_raw: pd.DataFrame,
                      tol: float = 1e-4) -> dict:
    """Check base + sum(phi) == margin, and sigmoid(margin) == predict_proba."""
    X_raw = ensure_feature_frame(X_raw)
    margin = predict_margin(pipe, X_raw)
    recon = exp.base_values + exp.values.sum(axis=1)
    prob = pipe.predict_proba(X_raw)[:, 1]
    logodds_err = np.abs(recon - margin)
    prob_err = np.abs(sigmoid(recon) - prob)
    # Demonstrates the unit trap: naively adding log-odds SHAP values to the
    # base probability does not reproduce the model probability.
    naive_prob = sigmoid(exp.base_values) + exp.values.sum(axis=1)
    return {
        "n": int(len(margin)),
        "max_abs_error_logodds": float(logodds_err.max()),
        "max_abs_error_probability_via_sigmoid": float(prob_err.max()),
        "max_abs_error_naive_probability_sum": float(np.abs(naive_prob - prob).max()),
        "tolerance": tol,
        "additive_in_logodds": bool(logodds_err.max() < tol),
        "probability_recovered_via_sigmoid": bool(prob_err.max() < tol),
    }


@dataclass
class ShapPatient:
    base_value: float
    margin: float
    probability: float
    values: dict[str, float]          # log-odds contributions
    model_input: dict[str, float]     # imputed values the model actually saw
    imputed: dict[str, bool]

    @property
    def additivity_error(self) -> float:
        return abs(self.base_value + sum(self.values.values()) - self.margin)

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": config.FEATURES,
            "model_input_value": [self.model_input[f] for f in config.FEATURES],
            "imputed": [self.imputed[f] for f in config.FEATURES],
            "shap_logodds": [self.values[f] for f in config.FEATURES],
        })
        df["direction"] = np.where(df["shap_logodds"] >= 0, "toward positive", "toward negative")
        df["abs"] = df["shap_logodds"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def explain_patient(explainer: shap.TreeExplainer, pipe: Pipeline, x_raw: pd.DataFrame) -> ShapPatient:
    x_raw = ensure_feature_frame(x_raw)
    if len(x_raw) != 1:
        raise ValueError("explain_patient expects exactly one row.")
    exp = explain(explainer, pipe, x_raw)
    margin = float(predict_margin(pipe, x_raw)[0])
    mask = imputed_mask(x_raw).iloc[0]
    return ShapPatient(
        base_value=float(exp.base_values[0]),
        margin=margin,
        probability=float(sigmoid(margin)),
        values={f: float(v) for f, v in zip(config.FEATURES, exp.values[0])},
        model_input={f: float(v) for f, v in zip(config.FEATURES, exp.data[0])},
        imputed={f: bool(mask[f]) for f in config.FEATURES},
    )


def global_importance(exp: shap.Explanation) -> pd.Series:
    """Mean |SHAP| (log-odds) per feature, descending."""
    return pd.Series(np.abs(exp.values).mean(axis=0), index=config.FEATURES).sort_values(ascending=False)
