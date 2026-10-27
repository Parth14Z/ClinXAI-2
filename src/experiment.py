"""Explanation-consistency study over every held-out patient.

Research question: how consistently do LIME and SHAP explain XGBoost diabetes
predictions across patient profiles and prediction-confidence levels?

For every test patient we record the model output, one primary LIME
explanation, the TreeSHAP explanation, agreement metrics between them, and
LIME stability across a fixed set of seeds.  Relationship analysis uses
Spearman correlations and reports effect sizes and p-values as they are,
together with a Bonferroni-adjusted threshold.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from . import config
from . import lime_explainer as lx
from . import shap_explainer as sx
from .comparison import compare
from .data import format_patient_id
from .model import transform_features


def confidence_band(p: float) -> str:
    if p >= config.HIGH_CONFIDENCE_POS:
        return "high-confidence positive"
    if p <= config.HIGH_CONFIDENCE_NEG:
        return "high-confidence negative"
    if config.BORDERLINE_LOW <= p <= config.BORDERLINE_HIGH:
        return "borderline"
    return "moderate"


BAND_ORDER = ["high-confidence negative", "moderate", "borderline", "high-confidence positive"]


def run_study(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series,
              ids_test: np.ndarray, seeds=config.LIME_STABILITY_SEEDS,
              num_samples: int = config.LIME_NUM_SAMPLES,
              progress: Callable[[int, int], None] | None = None,
              explanations: dict | None = None) -> dict[str, pd.DataFrame]:
    """Explain every test patient with SHAP and LIME and tabulate agreement and stability.

    The primary LIME explainer's random state advances from patient to patient, so a single
    patient's draw can only be reproduced by replaying the whole run. If `explanations` is given,
    it is filled with patient_id -> (LimeResult with neighbourhood, StabilityResult), so that
    downstream case studies reuse exactly the draws behind the study table.
    """
    X_train_t = transform_features(pipe, X_train)
    X_test_t = transform_features(pipe, X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    shap_tree = sx.build_tree_explainer(pipe)
    shap_exp = sx.explain(shap_tree, pipe, X_test)
    primary = lx.build_lime_explainer(X_train_t, config.LIME_PRIMARY_SEED)
    seed_explainers = {s: lx.build_lime_explainer(X_train_t, s) for s in seeds}

    rows, lime_long, shap_long, stab_long = [], [], [], []
    n = len(X_test)
    for i in range(n):
        pid = int(ids_test[i])
        x_t = X_test_t.iloc[[i]]
        lime_res = lx.explain_patient(pipe, primary, x_t, num_samples, seed=config.LIME_PRIMARY_SEED,
                                      keep_neighborhood=explanations is not None)
        shap_v = {f: float(v) for f, v in zip(config.FEATURES, shap_exp.values[i])}
        cmp = compare(lime_res.weights, shap_v)
        stab = lx.lime_stability(pipe, X_train_t, x_t, seeds, num_samples, seed_explainers)
        sm = stab.metrics
        if explanations is not None:
            explanations[pid] = (lime_res, stab)
        p = float(probs[i])
        dis = cmp["direction_disagreement_features"]
        rows.append({
            "patient_id": pid,
            "patient": format_patient_id(pid),
            "y_true": int(y_test.iloc[i]),
            "prob_positive": p,
            "y_pred": int(p >= config.DECISION_THRESHOLD),
            "correct": int((p >= config.DECISION_THRESHOLD) == bool(y_test.iloc[i])),
            "confidence": max(p, 1 - p),
            "confidence_band": confidence_band(p),
            "shap_margin": float(shap_exp.base_values[i] + shap_exp.values[i].sum()),
            "lime_r2": lime_res.r2,
            "lime_local_pred": lime_res.local_pred,
            "lime_fidelity_gap": lime_res.fidelity_gap,
            **{k: cmp[k] for k in ("top1_match", "top3_overlap", "top5_overlap", "jaccard_top3",
                                   "jaccard_top5", "rank_spearman", "sign_agreement",
                                   "n_direction_disagreements")},
            "direction_disagreement_features": ";".join(dis),
            "direction_disagreement_shap_mass": float(sum(abs(shap_v[f]) for f in dis)),
            "lime_top3": ";".join(cmp["lime_top3"]),
            "shap_top3": ";".join(cmp["shap_top3"]),
            **{k: sm[k] for k in ("stability_jaccard_top3", "stability_jaccard_top5",
                                  "stability_spearman", "sign_consistency", "top1_agreement",
                                  "weight_cv_top5", "r2_mean", "r2_std")},
            "sign_flip_features": ";".join(sm["sign_flip_features"]),
            "material_sign_flip_features": ";".join(sm["material_sign_flip_features"]),
            "n_material_sign_flips": len(sm["material_sign_flip_features"]),
            "n_imputed": int(sum((X_test.iloc[i][c] == 0) for c in config.ZERO_AS_MISSING)),
        })
        for f in config.FEATURES:
            lime_long.append({"patient_id": pid, "feature": f, "lime_weight": lime_res.weights[f],
                              "lime_rule": lime_res.rules[f]})
            shap_long.append({"patient_id": pid, "feature": f, "shap_logodds": shap_v[f],
                              "model_input_value": float(x_t.iloc[0][f])})
        for s, wrow in stab.weights.iterrows():
            for f in config.FEATURES:
                stab_long.append({"patient_id": pid, "seed": int(s), "feature": f,
                                  "lime_weight": float(wrow[f]), "r2": float(stab.r2[s])})
        if progress:
            progress(i + 1, n)

    return {
        "study": pd.DataFrame(rows),
        "lime_weights": pd.DataFrame(lime_long),
        "shap_values": pd.DataFrame(shap_long),
        "stability_weights": pd.DataFrame(stab_long),
    }


RELATIONSHIPS = [
    ("confidence", "lime_r2", "Confidence vs LIME surrogate R²"),
    ("confidence", "lime_fidelity_gap", "Confidence vs LIME local-prediction gap"),
    ("confidence", "rank_spearman", "Confidence vs LIME-SHAP rank correlation"),
    ("confidence", "jaccard_top3", "Confidence vs LIME-SHAP top-3 Jaccard"),
    ("confidence", "stability_jaccard_top3", "Confidence vs LIME seed stability (top-3 Jaccard)"),
    ("lime_r2", "rank_spearman", "LIME R² vs LIME-SHAP rank correlation"),
    ("lime_r2", "stability_jaccard_top3", "LIME R² vs LIME seed stability"),
    ("stability_jaccard_top3", "rank_spearman", "LIME seed stability vs LIME-SHAP rank correlation"),
]


def strength_label(rho: float) -> str:
    a = abs(rho)
    if np.isnan(a):
        return "undefined"
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


def analyse_relationships(study: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    bonf = alpha / len(RELATIONSHIPS)
    out = []
    for x, y, label in RELATIONSHIPS:
        sub = study[[x, y]].dropna()
        if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
            rho, pval = float("nan"), float("nan")
        else:
            res = spearmanr(sub[x], sub[y])
            rho, pval = float(res.statistic), float(res.pvalue)
        out.append({
            "relationship": label, "x": x, "y": y, "n": len(sub),
            "spearman_rho": rho, "p_value": pval, "strength": strength_label(rho),
            "significant_alpha_0.05": bool(pval < alpha) if not np.isnan(pval) else False,
            "significant_bonferroni": bool(pval < bonf) if not np.isnan(pval) else False,
            "bonferroni_threshold": bonf,
        })
    return pd.DataFrame(out)


BAND_METRICS = ["prob_positive", "lime_r2", "lime_fidelity_gap", "top3_overlap", "jaccard_top3",
                "jaccard_top5", "rank_spearman", "sign_agreement", "n_direction_disagreements",
                "stability_jaccard_top3", "stability_spearman", "sign_consistency"]


def band_summary(study: pd.DataFrame) -> pd.DataFrame:
    g = study.groupby("confidence_band")
    summary = g[BAND_METRICS].mean()
    summary.insert(0, "n_patients", g.size())
    order = [b for b in BAND_ORDER if b in summary.index]
    return summary.loc[order].round(3)


def overall_summary(study: pd.DataFrame) -> dict:
    s = study
    return {
        "n_patients": int(len(s)),
        "top1_match_rate": float(s["top1_match"].mean()),
        "mean_top3_overlap": float(s["top3_overlap"].mean()),
        "mean_top5_overlap": float(s["top5_overlap"].mean()),
        "mean_jaccard_top3": float(s["jaccard_top3"].mean()),
        "mean_jaccard_top5": float(s["jaccard_top5"].mean()),
        "median_rank_spearman": float(s["rank_spearman"].median()),
        "mean_rank_spearman": float(s["rank_spearman"].mean()),
        "share_with_direction_disagreement": float((s["n_direction_disagreements"] > 0).mean()),
        "mean_sign_agreement": float(s["sign_agreement"].mean()),
        "mean_lime_r2": float(s["lime_r2"].mean()),
        "median_lime_r2": float(s["lime_r2"].median()),
        "mean_lime_fidelity_gap": float(s["lime_fidelity_gap"].mean()),
        "mean_stability_jaccard_top3": float(s["stability_jaccard_top3"].mean()),
        "mean_stability_spearman": float(s["stability_spearman"].mean()),
        "mean_sign_consistency": float(s["sign_consistency"].mean()),
        "share_any_sign_flip_across_seeds": float((s["sign_consistency"] < 1).mean()),
        "share_material_sign_flip_across_seeds": float((s["n_material_sign_flips"] > 0).mean()),
        "band_counts": s["confidence_band"].value_counts().to_dict(),
    }
