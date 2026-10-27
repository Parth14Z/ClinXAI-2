"""Agreement metrics, case-study selection and relationship analysis."""
import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.case_studies import interpret_case, select_cases
from src.comparison import (compare, direction_disagreements, jaccard_topk, rank_features,
                            rank_spearman, top_k, topk_overlap)
from src.experiment import analyse_relationships, band_summary, confidence_band

F = config.FEATURES
A = dict(zip(F, [0.50, -0.40, 0.30, 0.20, -0.10, 0.05, 0.02, 0.01]))


def test_identical_vectors_agree_perfectly():
    m = compare(A, A)
    assert m["top3_overlap"] == m["top5_overlap"] == m["jaccard_top3"] == 1.0
    assert m["rank_spearman"] == pytest.approx(1.0)
    assert m["n_direction_disagreements"] == 0 and m["top1_match"]


def test_ranking_uses_absolute_values():
    assert top_k(A, 2) == ["Pregnancies", "Glucose"]
    assert rank_features({"Age": -3.0, "BMI": 1.0, "Glucose": 2.0}) == ["Age", "Glucose", "BMI"]


def test_known_overlap_and_jaccard():
    b = dict(zip(F, [0.50, 0.0, 0.0, 0.20, 0.0, 0.40, 0.0, 0.0]))  # top-3: Preg, BMI, Skin
    assert topk_overlap(A, b, 3) == pytest.approx(1 / 3)   # A top-3: Preg, Glucose, BP
    assert jaccard_topk(A, b, 3) == pytest.approx(1 / 5)   # {Preg} / {Preg,Glu,BP,BMI,Skin}


def test_reversed_ranking_gives_negative_spearman():
    rev = dict(zip(F, list(reversed([abs(v) for v in A.values()]))))
    assert rank_spearman(A, rev) == pytest.approx(-1.0)


def test_spearman_undefined_for_constant_vector():
    assert math.isnan(rank_spearman(A, dict.fromkeys(F, 1.0)))


def test_direction_disagreement_respects_tolerance():
    lime = dict(A)
    shap = {f: v for f, v in A.items()}
    shap["Pregnancies"] = -0.5          # material conflict
    lime["Age"] = 0.001                  # below LIME tolerance -> ignored
    shap["Age"] = -0.5
    assert direction_disagreements(lime, shap) == ["Pregnancies"]


def test_topk_rejects_invalid_k():
    with pytest.raises(ValueError):
        top_k(A, 0)


def _study(probs, rho=None, n_dis=None, stab=None):
    n = len(probs)
    return pd.DataFrame({
        "patient_id": np.arange(100, 100 + n), "prob_positive": probs,
        "y_true": [int(p > 0.5) for p in probs], "y_pred": [int(p >= 0.5) for p in probs],
        "correct": 1, "confidence": [max(p, 1 - p) for p in probs],
        "confidence_band": [confidence_band(p) for p in probs],
        "rank_spearman": rho if rho is not None else [0.9] * n,
        "jaccard_top3": [0.2] * n, "n_direction_disagreements": n_dis if n_dis is not None else [0] * n,
        "direction_disagreement_shap_mass": [0.0] * n, "direction_disagreement_features": [""] * n,
        "stability_jaccard_top3": stab if stab is not None else [1.0] * n, "stability_spearman": [1.0] * n,
        "lime_top3": ["a;b;c"] * n, "shap_top3": ["a;b;d"] * n,
    })


def test_case_selection_picks_distinct_rule_matching_patients():
    s = _study([0.95, 0.02, 0.51, 0.30, 0.70, 0.88], rho=[0.9, 0.9, 0.9, -0.5, 0.2, 0.9],
               n_dis=[0, 0, 0, 0, 2, 0])
    cases = select_cases(s)
    ids = [c.patient_id for c in cases]
    assert [c.case_id for c in cases] == [1, 2, 3, 4, 5]
    assert all(c.available for c in cases) and len(set(ids)) == 5
    assert ids[0] == 100 and ids[1] == 101 and ids[2] == 102 and ids[3] == 103 and ids[4] == 104


def test_unavailable_case_is_reported_not_invented():
    s = _study([0.30, 0.45, 0.70])     # nothing >= 0.85 or <= 0.15
    cases = {c.case_id: c for c in select_cases(s)}
    assert not cases[1].available and cases[1].patient_id is None
    assert not cases[2].available
    assert "No held-out patient" in cases[1].selection_note
    assert "No held-out patient" in interpret_case(cases[1], None, None, None)[0]


def test_case5_falls_back_to_instability_when_no_sign_conflict():
    s = _study([0.95, 0.02, 0.5, 0.3, 0.7, 0.6], rho=[0.9, 0.9, 0.9, -0.2, 0.9, 0.9],
               stab=[1, 1, 1, 1, 0.4, 0.9])
    c5 = select_cases(s)[4]
    assert c5.title == "LIME explanation instability" and c5.patient_id == 104


def test_confidence_bands():
    assert confidence_band(0.9) == "high-confidence positive"
    assert confidence_band(0.05) == "high-confidence negative"
    assert confidence_band(0.5) == "borderline"
    assert confidence_band(0.3) == "moderate"


def test_relationship_analysis_reports_rho_and_p():
    rng = np.random.default_rng(0)
    n = 60
    conf = rng.uniform(0.5, 1, n)
    df = pd.DataFrame({"confidence": conf, "lime_r2": conf + rng.normal(0, 0.05, n),
                       "lime_fidelity_gap": rng.uniform(0, 0.2, n), "rank_spearman": rng.uniform(-1, 1, n),
                       "jaccard_top3": rng.uniform(0, 1, n), "stability_jaccard_top3": rng.uniform(0, 1, n)})
    rel = analyse_relationships(df)
    row = rel.query("x == 'confidence' and y == 'lime_r2'").iloc[0]
    assert row.spearman_rho > 0.8 and row.significant_bonferroni
    assert len(rel) == 8 and rel["p_value"].between(0, 1).all()


def test_band_summary_orders_bands():
    s = _study([0.9, 0.05, 0.5, 0.3])
    for c in ("lime_r2", "lime_fidelity_gap", "top3_overlap", "jaccard_top5", "sign_agreement",
              "stability_spearman", "sign_consistency"):
        s[c] = 0.5
    out = band_summary(s)
    assert list(out.index) == ["high-confidence negative", "moderate", "borderline", "high-confidence positive"]
    assert out["n_patients"].sum() == 4


def test_run_study_exposes_the_exact_draws_behind_its_table(fitted, split):
    from src.experiment import run_study
    n = 3
    explanations: dict = {}
    res = run_study(fitted, split.X_train, split.X_test.iloc[:n], split.y_test.iloc[:n], split.ids_test[:n],
                    seeds=(0, 1), num_samples=300, explanations=explanations)
    assert sorted(explanations) == sorted(int(p) for p in split.ids_test[:n])
    lw = res["lime_weights"].pivot(index="patient_id", columns="feature", values="lime_weight")
    sw = res["stability_weights"]
    for pid, (lime_res, stab) in explanations.items():
        row = res["study"].set_index("patient_id").loc[pid]
        assert lime_res.neighborhood_X is not None and lime_res.r2 == row.lime_r2
        assert all(lime_res.weights[f] == lw.loc[pid, f] for f in F)
        assert np.allclose(stab.weights.to_numpy(),
                           sw[sw.patient_id == pid].pivot(index="seed", columns="feature",
                                                          values="lime_weight")[F].to_numpy())
