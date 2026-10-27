"""Agreement metrics between two feature-attribution vectors.

Inputs are mappings ``feature -> signed attribution`` for the same patient and
the same target class (diabetes-positive).  Rankings use absolute magnitude;
directions use the sign.  The two methods use different units (LIME: local
surrogate probability weights; TreeSHAP: log-odds), so magnitudes are never
compared directly - only ranks, set overlaps and signs.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.stats import spearmanr

from . import config


def rank_features(attr: Mapping[str, float]) -> list[str]:
    """Features sorted by |attribution| (desc); ties broken by canonical order."""
    order = {f: i for i, f in enumerate(config.FEATURES)}
    return sorted(attr, key=lambda f: (-abs(attr[f]), order.get(f, len(order))))


def top_k(attr: Mapping[str, float], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return rank_features(attr)[:k]


def topk_overlap(a: Mapping[str, float], b: Mapping[str, float], k: int) -> float:
    """|top_k(a) & top_k(b)| / k."""
    return len(set(top_k(a, k)) & set(top_k(b, k))) / k


def jaccard_topk(a: Mapping[str, float], b: Mapping[str, float], k: int) -> float:
    """Jaccard similarity of the two top-k feature sets."""
    sa, sb = set(top_k(a, k)), set(top_k(b, k))
    return len(sa & sb) / len(sa | sb)


def rank_spearman(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Spearman correlation of |attribution| over the shared features.

    Returns NaN when either vector is constant (correlation undefined).
    """
    feats = [f for f in config.FEATURES if f in a and f in b] or sorted(set(a) & set(b))
    x = np.abs([a[f] for f in feats])
    y = np.abs([b[f] for f in feats])
    if len(feats) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def direction_disagreements(lime_w: Mapping[str, float], shap_v: Mapping[str, float],
                            lime_tol: float = config.LIME_SIGN_TOL,
                            shap_tol: float = config.SHAP_SIGN_TOL) -> list[str]:
    """Features whose signs differ while both attributions are non-negligible."""
    out = []
    for f in config.FEATURES:
        if f not in lime_w or f not in shap_v:
            continue
        lw, sv = lime_w[f], shap_v[f]
        if abs(lw) >= lime_tol and abs(sv) >= shap_tol and np.sign(lw) != np.sign(sv):
            out.append(f)
    return out


def sign_agreement_rate(lime_w: Mapping[str, float], shap_v: Mapping[str, float]) -> float:
    """Fraction of all features with the same sign (no magnitude filter)."""
    feats = [f for f in config.FEATURES if f in lime_w and f in shap_v]
    same = [np.sign(lime_w[f]) == np.sign(shap_v[f]) for f in feats]
    return float(np.mean(same))


def compare(lime_w: Mapping[str, float], shap_v: Mapping[str, float]) -> dict:
    """All agreement metrics for one patient."""
    dis = direction_disagreements(lime_w, shap_v)
    return {
        "lime_top3": top_k(lime_w, 3),
        "shap_top3": top_k(shap_v, 3),
        "lime_top5": top_k(lime_w, 5),
        "shap_top5": top_k(shap_v, 5),
        "top1_match": top_k(lime_w, 1) == top_k(shap_v, 1),
        "top3_overlap": topk_overlap(lime_w, shap_v, 3),
        "top5_overlap": topk_overlap(lime_w, shap_v, 5),
        "jaccard_top3": jaccard_topk(lime_w, shap_v, 3),
        "jaccard_top5": jaccard_topk(lime_w, shap_v, 5),
        "rank_spearman": rank_spearman(lime_w, shap_v),
        "sign_agreement": sign_agreement_rate(lime_w, shap_v),
        "n_direction_disagreements": len(dis),
        "direction_disagreement_features": dis,
    }
