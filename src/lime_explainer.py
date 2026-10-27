"""LIME (LimeTabularExplainer) explanations for the XGBoost pipeline.

Design choices
--------------
* LIME's reference distribution is the *preprocessed* training split (zeros
  already imputed with training medians), so perturbations are sampled from
  plausible values rather than from impossible zero placeholders.
* The explained instance is the patient's preprocessed vector - the same
  vector TreeSHAP explains - so both methods explain the same function at the
  same point.
* Every perturbation batch goes through the full fitted pipeline's
  ``predict_proba``; column 1 is P(diabetes-positive), the label LIME
  explains (``labels=(1,)``).
* ``discretize_continuous=True`` with quartile bins: weights describe how
  falling in the patient's quartile bin changes the surrogate's predicted
  probability, so LIME weights are in probability units of a local linear
  surrogate, whereas SHAP values are log-odds.  Ranks and signs are
  comparable; raw magnitudes are not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline

from . import config
from .comparison import jaccard_topk, rank_spearman, top_k


class RecordingPredictFn:
    """predict_fn for LIME that also keeps the last perturbation batch."""

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.last_batch: np.ndarray | None = None
        self.last_probs: np.ndarray | None = None

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        X = pd.DataFrame(np.asarray(arr, dtype=float), columns=config.FEATURES)
        proba = self.pipe.predict_proba(X)
        self.last_batch = np.asarray(arr, dtype=float)
        self.last_probs = proba[:, 1]
        return proba


def build_lime_explainer(X_train_t: pd.DataFrame, seed: int = config.LIME_PRIMARY_SEED
                         ) -> LimeTabularExplainer:
    if list(X_train_t.columns) != config.FEATURES:
        raise ValueError("LIME training data must use the canonical feature order.")
    if X_train_t.isna().any().any():
        raise ValueError("LIME training data must be preprocessed (no NaN).")
    return LimeTabularExplainer(
        training_data=X_train_t.to_numpy(),
        feature_names=config.FEATURES,
        class_names=config.CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        discretizer="quartile",
        random_state=seed,
    )


@dataclass
class LimeResult:
    weights: dict[str, float]       # surrogate coefficients for label 1
    rules: dict[str, str]           # e.g. "Glucose > 141.00"
    intercept: float
    local_pred: float               # surrogate prediction at the patient
    model_prob: float               # XGBoost P(diabetes-positive) at the patient
    r2: float                       # weighted R^2 of the surrogate on the neighbourhood
    seed: int
    num_samples: int
    kernel_width: float
    neighborhood_X: np.ndarray | None = field(default=None, repr=False)
    neighborhood_probs: np.ndarray | None = field(default=None, repr=False)

    @property
    def fidelity_gap(self) -> float:
        return abs(self.local_pred - self.model_prob)

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "feature": config.FEATURES,
            "lime_rule": [self.rules[f] for f in config.FEATURES],
            "lime_weight": [self.weights[f] for f in config.FEATURES],
        })
        df["direction"] = np.where(df["lime_weight"] >= 0, "toward positive", "toward negative")
        df["abs"] = df["lime_weight"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)

    def neighborhood_summary(self) -> dict:
        if self.neighborhood_probs is None:
            return {}
        p = self.neighborhood_probs
        return {
            "n_perturbations": int(len(p)),
            "mean_prob": float(p.mean()),
            "std_prob": float(p.std()),
            "share_predicted_positive": float((p >= config.DECISION_THRESHOLD).mean()),
        }


def explain_patient(pipe: Pipeline, explainer: LimeTabularExplainer, x_t: pd.DataFrame,
                    num_samples: int = config.LIME_NUM_SAMPLES, seed: int | None = None,
                    keep_neighborhood: bool = False) -> LimeResult:
    """Explain P(diabetes-positive) for one preprocessed patient row."""
    if len(x_t) != 1 or list(x_t.columns) != config.FEATURES:
        raise ValueError("x_t must be a single preprocessed row in canonical feature order.")
    if num_samples < 100:
        raise ValueError("num_samples must be at least 100 for a meaningful neighbourhood.")
    predict_fn = RecordingPredictFn(pipe)
    row = x_t.to_numpy()[0]
    exp = explainer.explain_instance(row, predict_fn, labels=(1,),
                                     num_features=len(config.FEATURES), num_samples=num_samples)
    weights = {config.FEATURES[i]: float(w) for i, w in exp.as_map()[1]}
    rules = {}
    for (i, _), (rule, _) in zip(exp.as_map()[1], exp.as_list(label=1)):
        rules[config.FEATURES[i]] = rule
    model_prob = float(pipe.predict_proba(x_t)[0, 1])
    return LimeResult(
        weights=weights,
        rules=rules,
        intercept=float(exp.intercept[1]),
        local_pred=float(np.ravel(exp.local_pred)[0]),
        model_prob=model_prob,
        r2=float(exp.score if np.isscalar(exp.score) else np.ravel(exp.score)[0]),
        seed=int(seed if seed is not None else -1),
        num_samples=num_samples,
        kernel_width=float(np.sqrt(len(config.FEATURES)) * 0.75),
        neighborhood_X=predict_fn.last_batch if keep_neighborhood else None,
        neighborhood_probs=predict_fn.last_probs if keep_neighborhood else None,
    )


# --------------------------------------------------------------------------- #
# Stability across LIME random seeds
# --------------------------------------------------------------------------- #
@dataclass
class StabilityResult:
    weights: pd.DataFrame        # rows = seeds, columns = features
    r2: pd.Series
    local_pred: pd.Series

    @property
    def metrics(self) -> dict:
        rows = [dict(r) for _, r in self.weights.iterrows()]
        pairs = list(combinations(range(len(rows)), 2))
        jac3 = [jaccard_topk(rows[a], rows[b], 3) for a, b in pairs]
        jac5 = [jaccard_topk(rows[a], rows[b], 5) for a, b in pairs]
        rho = [rank_spearman(rows[a], rows[b]) for a, b in pairs]
        signs = np.sign(self.weights.to_numpy())
        sign_consistent = (signs == signs[0]).all(axis=0)
        mean_abs = self.weights.abs().mean()
        top5 = list(mean_abs.sort_values(ascending=False).index[:5])
        cv = (self.weights[top5].std(ddof=0) / self.weights[top5].abs().mean()).mean()
        # A flip is "material" only if some seed gives >= +tol and another <= -tol.
        w = self.weights
        material = [f for f in config.FEATURES
                    if (w[f] >= config.LIME_SIGN_TOL).any() and (w[f] <= -config.LIME_SIGN_TOL).any()]
        top1 = [top_k(r, 1)[0] for r in rows]
        return {
            "n_seeds": len(rows),
            "stability_jaccard_top3": float(np.mean(jac3)) if jac3 else 1.0,
            "stability_jaccard_top5": float(np.mean(jac5)) if jac5 else 1.0,
            "stability_spearman": float(np.nanmean(rho)) if rho else 1.0,
            "sign_consistency": float(sign_consistent.mean()),
            "sign_flip_features": [f for f, ok in zip(config.FEATURES, sign_consistent) if not ok],
            "material_sign_flip_features": material,
            "top1_agreement": float(pd.Series(top1).value_counts().iloc[0] / len(top1)),
            "weight_cv_top5": float(cv),
            "r2_mean": float(self.r2.mean()),
            "r2_std": float(self.r2.std(ddof=0)),
        }


def lime_stability(pipe: Pipeline, X_train_t: pd.DataFrame, x_t: pd.DataFrame,
                   seeds=config.LIME_STABILITY_SEEDS,
                   num_samples: int = config.LIME_NUM_SAMPLES,
                   explainers: dict[int, LimeTabularExplainer] | None = None) -> StabilityResult:
    """Re-run LIME with each seed (fresh explainer per seed) and collect weights."""
    weights, r2, lp = [], [], []
    for s in seeds:
        expl = explainers[s] if explainers and s in explainers else build_lime_explainer(X_train_t, s)
        res = explain_patient(pipe, expl, x_t, num_samples=num_samples, seed=s)
        weights.append(res.weights)
        r2.append(res.r2)
        lp.append(res.local_pred)
    idx = pd.Index(list(seeds), name="seed")
    return StabilityResult(
        weights=pd.DataFrame(weights, index=idx)[config.FEATURES],
        r2=pd.Series(r2, index=idx),
        local_pred=pd.Series(lp, index=idx),
    )
