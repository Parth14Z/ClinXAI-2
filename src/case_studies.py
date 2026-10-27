"""Rule-based selection of five genuine held-out case studies.

Each case is chosen by a fixed, pre-declared rule from the per-patient study
table, and each patient can be used only once.  If no held-out patient meets
a rule, the case is reported as unavailable instead of being invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class CaseStudy:
    case_id: int
    title: str
    rule: str
    available: bool
    patient_id: int | None = None
    selection_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _pick(df: pd.DataFrame, used: set[int], sort_by: list[str], ascending: list[bool]) -> int | None:
    cand = df[~df["patient_id"].isin(used)]
    if cand.empty:
        return None
    return int(cand.sort_values(sort_by + ["patient_id"], ascending=ascending + [True])
               .iloc[0]["patient_id"])


def select_cases(study: pd.DataFrame) -> list[CaseStudy]:
    s = study.copy()
    s["dist_to_threshold"] = (s["prob_positive"] - config.DECISION_THRESHOLD).abs()
    used: set[int] = set()
    cases: list[CaseStudy] = []

    def add(case_id, title, rule, pool, sort_by, ascending, note_fn):
        pid = _pick(pool, used, sort_by, ascending)
        if pid is None:
            cases.append(CaseStudy(case_id, title, rule, False,
                                   selection_note="No held-out patient satisfies this rule."))
            return
        used.add(pid)
        cases.append(CaseStudy(case_id, title, rule, True, pid, note_fn(s.set_index("patient_id").loc[pid])))

    add(1, "High-confidence diabetes-positive prediction",
        f"Highest P(positive) among patients with P >= {config.HIGH_CONFIDENCE_POS}",
        s[s["prob_positive"] >= config.HIGH_CONFIDENCE_POS], ["prob_positive"], [False],
        lambda r: f"P(positive) = {r.prob_positive:.3f}")

    add(2, "High-confidence diabetes-negative prediction",
        f"Lowest P(positive) among patients with P <= {config.HIGH_CONFIDENCE_NEG}",
        s[s["prob_positive"] <= config.HIGH_CONFIDENCE_NEG], ["prob_positive"], [True],
        lambda r: f"P(positive) = {r.prob_positive:.3f}")

    add(3, "Borderline prediction",
        f"P(positive) closest to the {config.DECISION_THRESHOLD} threshold within "
        f"[{config.BORDERLINE_LOW}, {config.BORDERLINE_HIGH}]",
        s[s["prob_positive"].between(config.BORDERLINE_LOW, config.BORDERLINE_HIGH)],
        ["dist_to_threshold"], [True],
        lambda r: f"P(positive) = {r.prob_positive:.3f}, |P - 0.5| = {r.dist_to_threshold:.3f}")

    rank_pool = s[(s["jaccard_top3"] <= 0.5) & s["rank_spearman"].notna()]
    add(4, "LIME-SHAP feature-ranking disagreement",
        "Lowest LIME-SHAP Spearman rank correlation among patients whose top-3 sets share "
        "at most 2 features (Jaccard <= 0.5); ties broken by lower top-3 Jaccard",
        rank_pool, ["rank_spearman", "jaccard_top3"], [True, True],
        lambda r: f"Spearman rho = {r.rank_spearman:.3f}, top-3 Jaccard = {r.jaccard_top3:.2f} "
                  f"(LIME top-3: {r.lime_top3}; SHAP top-3: {r.shap_top3})")

    dir_pool = s[s["n_direction_disagreements"] > 0]
    if not dir_pool[~dir_pool["patient_id"].isin(used)].empty:
        add(5, "LIME-SHAP directional disagreement",
            "Most same-feature sign disagreements (both attributions above tolerance); ties broken "
            "by the SHAP magnitude of the disagreeing features",
            dir_pool, ["n_direction_disagreements", "direction_disagreement_shap_mass"], [False, False],
            lambda r: f"{int(r.n_direction_disagreements)} feature(s) with opposite signs: "
                      f"{r.direction_disagreement_features}")
    else:
        add(5, "LIME explanation instability",
            "No directional disagreement was found, so: lowest mean pairwise top-3 Jaccard of LIME "
            "across seeds; ties broken by lower seed rank correlation",
            s, ["stability_jaccard_top3", "stability_spearman"], [True, True],
            lambda r: f"seed top-3 Jaccard = {r.stability_jaccard_top3:.3f}")
    return cases


def describe_r2(r2: float) -> str:
    if r2 < 0.3:
        return "low - the linear surrogate captures little of the model's local behaviour"
    if r2 < 0.6:
        return "moderate"
    return "high"


def interpret_case(case: CaseStudy, row: pd.Series, lime_table: pd.DataFrame,
                   shap_table: pd.DataFrame) -> list[str]:
    """Evidence statements generated only from the measured values."""
    if not case.available:
        return [case.selection_note]
    truth = "diabetes-positive" if row.y_true == 1 else "diabetes-negative"
    pred = "diabetes-positive" if row.y_pred == 1 else "diabetes-negative"
    lines = [
        f"XGBoost score P(positive) = {row.prob_positive:.3f} (uncalibrated) -> predicted {pred}; "
        f"recorded outcome {truth} ({'correct' if row.correct else 'misclassified'}).",
    ]
    s_top = shap_table.iloc[0]
    l_top = lime_table.iloc[0]
    lines.append(
        f"Largest SHAP contribution: {s_top.feature} ({s_top.shap_logodds:+.3f} log-odds, "
        f"{s_top.direction}); largest LIME weight: {l_top.feature} ({l_top.lime_weight:+.3f}, "
        f"rule '{l_top.lime_rule}')."
    )
    lines.append(
        f"Agreement: top-3 overlap {row.top3_overlap:.2f}, top-5 overlap {row.top5_overlap:.2f}, "
        f"top-3 Jaccard {row.jaccard_top3:.2f}, Spearman rank rho "
        f"{row.rank_spearman:.2f}, sign agreement {row.sign_agreement:.2f}."
    )
    if row.n_direction_disagreements:
        lines.append(
            f"Opposite directions for {row.direction_disagreement_features}. LIME's sign reflects the "
            f"patient's quartile bin versus other bins in the perturbed neighbourhood; SHAP's sign "
            f"reflects the value versus the tree-path expectation. A sign difference shows the two "
            f"baselines differ; it does not establish which method is 'right'."
        )
    else:
        lines.append("No material directional disagreement between LIME and SHAP for this patient.")
    lines.append(f"LIME surrogate weighted R² = {row.lime_r2:.3f} ({describe_r2(row.lime_r2)}); "
                 f"surrogate-vs-model gap = {row.lime_fidelity_gap:.3f}.")
    def _names(v):
        return v if isinstance(v, str) and v else "none"
    lines.append(
        f"Across {len(config.LIME_STABILITY_SEEDS)} LIME seeds: mean top-3 Jaccard "
        f"{row.stability_jaccard_top3:.2f}, mean rank rho {row.stability_spearman:.2f}; material sign flips "
        f"(|w| >= {config.LIME_SIGN_TOL} on both sides): {_names(row.material_sign_flip_features)}; "
        f"any sign change incl. near-zero weights: {_names(row.sign_flip_features)}."
    )
    if row.n_imputed:
        lines.append(f"{int(row.n_imputed)} of this patient's values were zero placeholders replaced "
                     f"by training medians; attributions to those features describe the median, "
                     f"not a measurement.")
    return lines


def cases_frame(cases: list[CaseStudy], study: pd.DataFrame) -> pd.DataFrame:
    s = study.set_index("patient_id")
    out = []
    for c in cases:
        rec = c.to_dict()
        if c.available:
            r = s.loc[c.patient_id]
            rec.update({k: (r[k].item() if isinstance(r[k], np.generic) else r[k]) for k in (
                "patient", "y_true", "y_pred", "prob_positive", "confidence_band", "lime_r2",
                "top3_overlap", "jaccard_top3", "rank_spearman", "n_direction_disagreements",
                "direction_disagreement_features", "stability_jaccard_top3", "stability_spearman")})
        out.append(rec)
    return pd.DataFrame(out)
