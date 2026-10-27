"""Matplotlib figures for the dashboard and the report evidence.

Visual rules used throughout:
* Signed contributions use a diverging pair: purple = pushes toward
  diabetes-positive, teal = pushes toward diabetes-negative.
* LIME (surrogate probability weights) and SHAP (log-odds) are never drawn on
  a shared value axis; they get separate panels with their own units.
* Text is drawn in ink colours, never in the series colour.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config  # noqa: E402
from .comparison import rank_features  # noqa: E402

P = config.PALETTE
INK = P["slate"]
MUTED = "#5B6675"
SEQ_TEAL = LinearSegmentedColormap.from_list("clinx_teal", ["#EEF8F7", P["teal"], "#04443F"])
SEQ_PURPLE = LinearSegmentedColormap.from_list("clinx_purple", [P["purple_light"], "#3B2A70"])

# pyplot's "current figure" is process-global and SHAP's plots draw on it, so concurrent dashboard
# sessions (one thread each) must not build figures at the same time.
PLOT_LOCK = threading.Lock()


@contextmanager
def exclusive_figure(make_fig, *args, **kwargs) -> Iterator[plt.Figure]:
    """Build a figure while holding PLOT_LOCK; it stays locked until the caller has rendered it."""
    with PLOT_LOCK:
        plt.close("all")  # leftovers from an interrupted run could otherwise become "current"
        fig = None
        try:
            fig = make_fig(*args, **kwargs)
            yield fig
        finally:
            if fig is not None:
                plt.close(fig)


LIME_UNITS = "LIME weight (local surrogate, P(positive) units)"
SHAP_UNITS = "SHAP value (log-odds)"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#C7CFD8",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": P["grid"],
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


apply_style()


def save(fig: plt.Figure, path: Path, dpi: int = 160) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _sign_colors(values) -> list[str]:
    return [config.RISK_UP_COLOR if v >= 0 else config.RISK_DOWN_COLOR for v in values]


def _short(name: str) -> str:
    return {"DiabetesPedigreeFunction": "PedigreeFunction"}.get(name, name)


# --------------------------------------------------------------------------- #
# Dataset and model evaluation
# --------------------------------------------------------------------------- #
def plot_dataset_overview(summary: dict) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    counts = summary["class_counts"]
    labels = ["Negative (0)", "Positive (1)"]
    vals = [counts.get(0, counts.get("0", 0)), counts.get(1, counts.get("1", 0))]
    a1.bar(labels, vals, color=[P["teal"], P["purple"]], width=0.55)
    for i, v in enumerate(vals):
        a1.text(i, v + 8, f"{v} ({100 * v / sum(vals):.1f}%)", ha="center", color=INK, fontsize=9)
    a1.set_title("Outcome classes")
    a1.set_ylabel("Patients")
    a1.set_ylim(0, max(vals) * 1.18)
    a1.grid(axis="x", visible=False)

    zp = summary["zero_placeholder_pct"]
    feats = sorted(zp, key=zp.get)
    a2.barh(feats, [zp[f] for f in feats], color=P["teal"], height=0.55)
    for i, f in enumerate(feats):
        a2.text(zp[f] + 0.8, i, f"{zp[f]:.1f}%  (n={summary['zero_placeholders'][f]})",
                va="center", color=INK, fontsize=9)
    a2.set_title("Zero placeholders treated as missing")
    a2.set_xlabel("% of all 768 records")
    a2.set_xlim(0, max(zp.values()) * 1.35)
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion matrix (held-out test set)") -> plt.Figure:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.imshow(cm, cmap=SEQ_TEAL, vmin=0)
    ax.grid(False)
    labels = ["Negative (0)", "Positive (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Recorded outcome")
    thresh = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{100 * cm[i, j] / cm[i].sum():.0f}% of row", ha="center",
                    va="center", fontsize=10, color="white" if cm[i, j] > thresh else INK)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_roc_curves(curves: Mapping[str, tuple], title: str = "ROC curve (held-out test set)") -> plt.Figure:
    """curves: name -> (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    colors = [P["teal"], P["purple"]]
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=P["grey"], label="Chance")
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), colors):
        ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False-positive rate (1 - specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_pr_curves(curves: Mapping[str, tuple], prevalence: float,
                   title: str = "Precision-recall curve (held-out test set)") -> plt.Figure:
    """curves: name -> (recall, precision, average_precision)."""
    fig, ax = plt.subplots(figsize=(5, 4.4))
    colors = [P["teal"], P["purple"]]
    ax.axhline(prevalence, ls="--", lw=1, color=P["grey"], label=f"Prevalence ({prevalence:.2f})")
    for (name, (rec, prec, ap)), c in zip(curves.items(), colors):
        ax.plot(rec, prec, lw=2, color=c, label=f"{name} (AP {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Dot plot; metrics: index = metric names, columns = model names."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = list(metrics.columns)
    colors = [P["teal"], P["purple"]]
    y = np.arange(len(metrics.index))
    for row_i, m in enumerate(metrics.index):
        vals = metrics.loc[m].to_numpy(dtype=float)
        ax.plot([vals.min(), vals.max()], [row_i, row_i], color=P["grid"], lw=3, zorder=1)
    for k, (name, c) in enumerate(zip(names, colors)):
        ax.scatter(metrics[name], y, s=70, color=c, edgecolor="white", linewidth=2,
                   zorder=3, label=name)
    for row_i, m in enumerate(metrics.index):
        vals = metrics.loc[m]
        ax.text(vals.max() + 0.012, row_i, " / ".join(f"{v:.3f}" for v in vals), va="center",
                fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [m.replace("_", " ").upper() if m == "roc_auc" else m.replace("_", " ").title()
                      for m in metrics.index])
    ax.invert_yaxis()
    ax.set_xlim(max(0, metrics.min().min() - 0.08), min(1.0, metrics.max().max() + 0.12))
    ax.set_xlabel("Score on held-out test set")
    ax.set_title("XGBoost vs logistic-regression baseline")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, -0.38))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def plot_shap_importance(importance: pd.Series,
                         title: str = "Global SHAP importance (mean |SHAP|, test set)") -> plt.Figure:
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh([_short(f) for f in imp.index], imp.values, color=P["teal"], height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.015, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_xlim(0, imp.max() * 1.15)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(exp: shap.Explanation, title: str = "SHAP beeswarm (test set)") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    short = shap.Explanation(values=exp.values, base_values=exp.base_values, data=exp.data,
                             feature_names=[_short(f) for f in exp.feature_names])
    np.random.seed(config.RANDOM_SEED)  # beeswarm shuffles and jitters dots with numpy's global RNG
    shap.plots.beeswarm(short, max_display=len(config.FEATURES), show=False, color=SEQ_PURPLE,
                        plot_size=None, ax=ax)
    ax.set_xlabel("SHAP value (log-odds of diabetes-positive)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_shap_waterfall(exp_row: shap.Explanation, title: str | None = None) -> plt.Figure:
    plt.figure(figsize=(7.5, 4.8))
    shap.plots.waterfall(exp_row, max_display=len(config.FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(7.5, 4.8)
    if title:
        fig.axes[0].set_title(title, pad=24)
    fig.text(0.01, -0.045, "Units: log-odds (raw XGBoost margin). f(x) = E[f(X)] + Σ SHAP; "
             "P(positive) = sigmoid(f(x)).", fontsize=8, color=MUTED)
    return fig


def plot_shap_force(base_value: float, values: np.ndarray, display_values,
                    title: str | None = None) -> plt.Figure:
    names = [_short(f) for f in config.FEATURES]
    feats = pd.Series(list(display_values), index=names)
    fig = shap.plots.force(base_value, np.asarray(values, dtype=float), feats, matplotlib=True,
                           show=False, text_rotation=12, contribution_threshold=0.04,
                           figsize=(14, 3.2))
    if title:
        fig.suptitle(title, y=1.12, fontsize=12, color=INK)
    fig.text(0.0, -0.32, "Units: log-odds. Red bars increase and blue bars decrease the margin "
             "(native SHAP colours); probability = sigmoid(margin).", fontsize=9, color=MUTED,
             transform=fig.axes[0].transAxes)
    return fig


# --------------------------------------------------------------------------- #
# Signed contribution bars (LIME, SHAP, side-by-side)
# --------------------------------------------------------------------------- #
def _contribution_axis(ax, attr: Mapping[str, float], order: list[str], xlabel: str,
                       labels: Mapping[str, str] | None = None, title: str = "") -> None:
    vals = [attr[f] for f in order]
    ylabels = [labels[f] if labels else _short(f) for f in order]
    ax.barh(range(len(order)), vals, color=_sign_colors(vals), height=0.62)
    ax.axvline(0, color=INK, lw=0.8)
    span = max(abs(v) for v in vals) or 1.0
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.03 if v >= 0 else -span * 0.03), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(order)), ylabels)
    ax.invert_yaxis()
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def _direction_legend(fig) -> None:
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=config.RISK_UP_COLOR, label="Pushes toward diabetes-positive"),
                        Patch(color=config.RISK_DOWN_COLOR, label="Pushes toward diabetes-negative")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9)


def plot_lime_explanation(weights: Mapping[str, float], rules: Mapping[str, str],
                          title: str = "LIME local explanation") -> plt.Figure:
    order = rank_features(weights)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _contribution_axis(ax, weights, order, LIME_UNITS, labels=rules, title=title)
    _direction_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_shap_contributions(values: Mapping[str, float],
                            title: str = "SHAP patient contributions") -> plt.Figure:
    order = rank_features(values)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    _contribution_axis(ax, values, order, SHAP_UNITS, title=title)
    _direction_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_side_by_side(lime_w: Mapping[str, float], shap_v: Mapping[str, float],
                      title: str = "LIME vs SHAP for the same patient",
                      disagree: list[str] | None = None) -> plt.Figure:
    """Two panels with a shared feature order (SHAP ranking) and separate units."""
    order = rank_features(shap_v)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    _contribution_axis(a1, shap_v, order, SHAP_UNITS, title="TreeSHAP (log-odds)")
    _contribution_axis(a2, lime_w, order, LIME_UNITS, title="LIME (local surrogate)")
    for f in disagree or []:
        i = order.index(f)
        for ax in (a1, a2):
            ax.axhspan(i - 0.45, i + 0.45, color="#FFF3D6", zorder=0)
    fig.suptitle(title, fontsize=13, color=INK, y=1.0)
    _direction_legend(fig)
    if disagree:
        fig.text(0.99, -0.02, "Shaded rows: opposite directions", ha="right", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def plot_rank_slope(lime_w: Mapping[str, float], shap_v: Mapping[str, float],
                    title: str = "Feature rank: SHAP vs LIME") -> plt.Figure:
    sr = {f: i + 1 for i, f in enumerate(rank_features(shap_v))}
    lr = {f: i + 1 for i, f in enumerate(rank_features(lime_w))}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for f in config.FEATURES:
        moved = abs(sr[f] - lr[f]) >= 2
        c = P["purple"] if moved else P["grey"]
        ax.plot([0, 1], [sr[f], lr[f]], color=c, lw=2 if moved else 1.2, marker="o", ms=7,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(-0.04, sr[f], f"{_short(f)}  #{sr[f]}", ha="right", va="center", fontsize=9, color=INK)
        ax.text(1.04, lr[f], f"#{lr[f]}  {_short(f)}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(len(config.FEATURES) + 0.6, 0.4)
    ax.set_xticks([0, 1], ["SHAP rank", "LIME rank"])
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title)
    fig.text(0.5, 0.0, "Highlighted lines move two or more rank positions", ha="center",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# LIME neighbourhood and stability
# --------------------------------------------------------------------------- #
def plot_neighborhood(probs: np.ndarray, patient_prob: float,
                      title: str = "XGBoost scores on LIME's perturbed neighbourhood") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.hist(probs, bins=40, range=(0, 1), color=P["teal_light"], edgecolor="white", linewidth=1)
    ax.axvline(config.DECISION_THRESHOLD, color=P["grey"], ls="--", lw=1)
    ax.axvline(patient_prob, color=P["purple_dark"], lw=2)
    ymax = ax.get_ylim()[1]
    ax.text(patient_prob, ymax * 0.95, f"  patient {patient_prob:.2f}", color=INK, fontsize=9, va="top")
    ax.text(config.DECISION_THRESHOLD, ymax * 0.8, "  threshold 0.5", color=MUTED, fontsize=8.5, va="top")
    ax.set_xlabel("P(diabetes-positive) for each perturbed sample")
    ax.set_ylabel("Perturbed samples")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_stability(weights: pd.DataFrame, title: str = "LIME weights across random seeds") -> plt.Figure:
    """weights: rows = seeds, columns = features."""
    order = list(weights.abs().mean().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        v = weights[f].to_numpy()
        ax.plot([v.min(), v.max()], [i, i], color="#C7CFD8", lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), c=_sign_colors(v), s=26,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(order)), [_short(f) for f in order])
    ax.invert_yaxis()
    ax.set_xlabel(LIME_UNITS)
    ax.set_title(f"{title} (n = {len(weights)} seeds)")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.0, "Grey bar = range across seeds; a bar crossing zero means the sign flipped.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Study-level figures
# --------------------------------------------------------------------------- #
METRIC_LABELS = {
    "confidence": "Prediction confidence max(p, 1-p)",
    "lime_r2": "LIME surrogate R²",
    "lime_fidelity_gap": "|LIME local pred - model p|",
    "rank_spearman": "LIME-SHAP rank Spearman ρ",
    "jaccard_top3": "LIME-SHAP top-3 Jaccard",
    "stability_jaccard_top3": "LIME seed stability (top-3 Jaccard)",
}


def plot_relationships(study: pd.DataFrame, rel: pd.DataFrame) -> plt.Figure:
    pairs = [("confidence", "lime_r2"), ("confidence", "rank_spearman"),
             ("confidence", "stability_jaccard_top3"), ("lime_r2", "rank_spearman")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for ax, (x, y) in zip(axes.ravel(), pairs):
        jitter = np.random.default_rng(1).uniform(-0.01, 0.01, len(study)) if "jaccard" in y else 0
        ax.scatter(study[x], study[y] + jitter, s=22, color=P["teal"], alpha=0.75,
                   edgecolor="white", linewidth=0.6)
        r = rel[(rel.x == x) & (rel.y == y)]
        if len(r):
            r = r.iloc[0]
            ax.set_title(f"ρ = {r.spearman_rho:+.2f}, p = {r.p_value:.3g} ({r.strength})", fontsize=10.5)
        ax.set_xlabel(METRIC_LABELS[x])
        ax.set_ylabel(METRIC_LABELS[y])
    fig.suptitle(f"Explanation quality vs prediction confidence (n = {len(study)} held-out patients)",
                 fontsize=13, color=INK)
    fig.tight_layout()
    return fig


def plot_band_summary(bands: pd.DataFrame) -> plt.Figure:
    metrics = [("lime_r2", "LIME surrogate R²"), ("rank_spearman", "LIME-SHAP rank ρ"),
               ("jaccard_top3", "LIME-SHAP top-3 Jaccard"),
               ("stability_jaccard_top3", "LIME seed stability")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    labels = [f"{b}\n(n={int(bands.loc[b, 'n_patients'])})" for b in bands.index]
    for ax, (m, lab) in zip(axes, metrics):
        ax.barh(labels, bands[m], color=P["teal"], height=0.55)
        for i, v in enumerate(bands[m]):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1.18)
        ax.set_title(lab, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Mean explanation metrics by prediction-confidence band", fontsize=13, color=INK)
    fig.tight_layout()
    return fig


def plot_agreement_distribution(study: pd.DataFrame) -> plt.Figure:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6))
    ov = (study["top3_overlap"] * 3).round().astype(int).value_counts().reindex(range(4), fill_value=0)
    a1.bar([f"{k}/3" for k in ov.index], ov.values, color=P["teal"], width=0.6)
    for i, v in enumerate(ov.values):
        a1.text(i, v + 1, str(v), ha="center", fontsize=9, color=INK)
    a1.set_title("Shared features in LIME & SHAP top-3")
    a1.set_ylabel("Patients")
    a1.grid(axis="x", visible=False)

    a2.hist(study["rank_spearman"].dropna(), bins=16, range=(-1, 1), color=P["teal"],
            edgecolor="white", linewidth=1)
    a2.axvline(study["rank_spearman"].median(), color=P["purple_dark"], lw=2)
    a2.text(study["rank_spearman"].median(), a2.get_ylim()[1] * 0.95,
            f"  median {study['rank_spearman'].median():.2f}", fontsize=9, va="top", color=INK)
    a2.set_title("LIME-SHAP rank correlation (Spearman ρ)")
    a2.set_xlabel("ρ over |attributions| of 8 features")
    a2.grid(axis="x", visible=False)

    dd = study["n_direction_disagreements"].value_counts().sort_index()
    a3.bar([str(k) for k in dd.index], dd.values, color=P["teal"], width=0.6)
    for i, v in enumerate(dd.values):
        a3.text(i, v + 1, str(v), ha="center", fontsize=9, color=INK)
    a3.set_title("Features with opposite LIME/SHAP sign")
    a3.set_xlabel("Number of material sign disagreements")
    a3.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_stability_distribution(study: pd.DataFrame) -> plt.Figure:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.hist(study["stability_jaccard_top3"], bins=12, range=(0, 1), color=P["teal"],
            edgecolor="white", linewidth=1)
    a1.set_title("LIME seed stability: mean pairwise top-3 Jaccard")
    a1.set_xlabel("1.0 = identical top-3 set under every seed")
    a1.set_ylabel("Patients")
    a1.grid(axis="x", visible=False)
    a2.hist(study["lime_r2"], bins=20, range=(0, max(1.0, study["lime_r2"].max())),
            color=P["purple"], edgecolor="white", linewidth=1)
    a2.set_title("LIME surrogate fidelity (weighted R²)")
    a2.set_xlabel("R² of local surrogate on its neighbourhood")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #
def plot_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor=face, edgecolor=edge, linewidth=1.4))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", fontsize=8.4, color=MUTED,
                linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color="#7A8796", linewidth=1.3))

    T, Pp, G = (P["teal_pale"], P["teal"]), (P["purple_pale"], P["purple"]), ("#F6F8FA", "#B8C2CC")
    box(0.2, 4.1, 2.3, 1.7, "Data", "OpenML #37 ARFF\nMD5-verified\n768 x 8 + Outcome", *G)
    box(2.9, 4.1, 2.4, 1.7, "Split", "Stratified 80/20\nseed 42\n614 train / 154 test", *G)
    box(5.7, 4.1, 2.6, 1.7, "Preprocess (fit on train)", "zeros -> NaN (5 cols)\nmedian imputer\nfeature order fixed", *T)
    box(8.7, 4.1, 2.0, 1.7, "XGBoost", "5-fold CV grid\n(train only)\nbinary:logistic", *T)
    box(11.0, 4.1, 1.8, 1.7, "Baseline", "Logistic\nregression\n+ scaling", *G)
    box(5.7, 1.9, 2.6, 1.6, "LIME", "quartile perturbations\nlocal ridge surrogate\nR² + 10-seed stability", *Pp)
    box(8.7, 1.9, 2.0, 1.6, "TreeSHAP", "log-odds output\nadditivity check\nglobal + local", *Pp)
    box(11.0, 1.9, 1.8, 1.6, "Evaluation", "held-out test\nAUC, F1, CM\nROC / PR", *G)
    box(2.9, 1.9, 2.4, 1.6, "Comparison study", "top-k overlap, Jaccard\nSpearman, sign checks\n5 case studies", *Pp)
    box(0.2, 0.15, 12.6, 1.15, "Streamlit dashboard  +  outputs/ (figures, metrics, case studies)",
        "Home · Prediction · LIME · SHAP · Comparison · Experimental evaluation · "
        "Case-study explorer", "white", P["teal_dark"])

    arrow(2.5, 4.95, 2.9, 4.95)
    arrow(5.3, 4.95, 5.7, 4.95)
    arrow(8.3, 4.95, 8.7, 4.95)
    arrow(9.7, 4.1, 9.7, 3.5)
    arrow(9.0, 4.1, 7.6, 3.5)
    arrow(10.7, 4.6, 11.0, 3.4)
    arrow(11.9, 4.1, 11.9, 3.5)
    arrow(5.7, 2.7, 5.3, 2.7)
    arrow(8.7, 2.3, 5.3, 2.3)
    arrow(4.1, 1.9, 4.1, 1.3)
    arrow(11.9, 1.9, 11.9, 1.3)
    ax.text(6.5, 6.05, "ClinXAI system architecture", ha="center", fontsize=14,
            fontweight="bold", color=INK)
    return fig
