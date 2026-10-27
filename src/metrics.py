"""Held-out classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from . import config


def classification_metrics(y_true, prob, threshold: float = config.DECISION_THRESHOLD) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError("Probabilities must lie in [0, 1].")
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def roc_points(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    return fpr, tpr, float(roc_auc_score(y_true, prob))


def pr_points(y_true, prob):
    prec, rec, _ = precision_recall_curve(y_true, prob)
    return rec, prec, float(average_precision_score(y_true, prob))


def bootstrap_auc_ci(y_true, prob, n_boot: int = 2000, seed: int = config.RANDOM_SEED,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for test-set ROC-AUC (conveys small-sample uncertainty)."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n_boot: int = 2000,
                              seed: int = config.RANDOM_SEED) -> dict:
    """Bootstrap distribution of AUC(a) - AUC(b) on the same resampled patients."""
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "observed_diff": float(roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "share_a_better": float((diffs > 0).mean()),
    }
