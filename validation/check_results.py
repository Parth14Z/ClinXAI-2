"""Independent re-verification of ClinXAI's saved results.

Usage:  python validation/check_results.py

Recomputes every reported number from the saved model, the saved split and the saved study
tables with its own numpy / scikit-learn / scipy code, and compares it with what the project
reports. Nothing is retrained or overwritten. Writes outputs/validation/results_check.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import lime_explainer as lx  # noqa: E402
from src import plotting as viz  # noqa: E402
from src import shap_explainer as sx  # noqa: E402
from src.case_studies import select_cases  # noqa: E402
from src.data import DatasetError, ensure_dataset, load_dataset, load_split_csv  # noqa: E402
from src.model import build_xgb_pipeline, load_metadata, load_pipeline, transform_features  # noqa: E402

OUT = ROOT / "outputs" / "validation"
F = config.FEATURES
results: list[dict] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> None:
    results.append({"section": section, "check": name, "passed": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} | {name}" + (f" - {detail}" if detail else ""))


def close(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0))


def load(rel):
    return json.loads((ROOT / rel).read_text("utf-8"))


# --------------------------------------------------------------------------- #
def provenance() -> None:
    s = "1 provenance"
    arff = config.RAW_ARFF_PATH
    md5 = hashlib.md5(arff.read_bytes()).hexdigest()
    prov = load("data/raw/provenance.json")
    check(s, "raw ARFF MD5 equals OpenML's published MD5", md5 == config.OPENML_MD5 == prov["md5_actual"],
          md5)
    check(s, "SHA-256 matches provenance.json",
          hashlib.sha256(arff.read_bytes()).hexdigest() == prov["sha256"])
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "corrupt.arff"
        shutil.copy(arff, bad)
        data = bytearray(bad.read_bytes())
        data[-10] = ord("9") if data[-10] != ord("9") else ord("8")
        bad.write_bytes(bytes(data))
        try:
            ensure_dataset(bad)
            rejected = False
        except DatasetError:
            rejected = True
    check(s, "a one-byte-corrupted copy is rejected by ensure_dataset", rejected)
    df = load_dataset()
    check(s, "768 rows x 8 features, 500 negative / 268 positive",
          df.shape[0] == 768 and list(df.columns[1:9]) == F and df.Outcome.value_counts().to_dict() == {0: 500, 1: 268})


def positive_class(xgb, X_test, y_test) -> None:
    s = "2 positive class"
    text = config.RAW_ARFF_PATH.read_text("utf-8")
    n_pos_raw = len(re.findall(r",tested_positive\s*$", text, flags=re.M))
    df = load_dataset()
    check(s, "ARFF 'tested_positive' rows == encoded Outcome==1 rows", n_pos_raw == int(df.Outcome.sum()) == 268,
          f"{n_pos_raw} raw positive rows")
    clf = xgb.named_steps[list(xgb.named_steps)[-1]]
    check(s, "classifier classes_ == [0, 1]", list(clf.classes_) == [0, 1], str(list(clf.classes_)))
    p1 = xgb.predict_proba(X_test)[:, 1]
    margin = sx.predict_margin(xgb, X_test)
    check(s, "predict_proba[:, 1] == sigmoid(raw margin)", close(p1, 1 / (1 + np.exp(-margin)), 1e-6))
    check(s, "column 1 ranks recorded positives higher (AUC > 0.5)", roc_auc_score(y_test, p1) > 0.5,
          f"AUC with column 1 = {roc_auc_score(y_test, p1):.4f}")
    check(s, "CLASS_NAMES[1] is the diabetes-positive class", "positive" in config.CLASS_NAMES[1].lower())


def leakage(xgb, X_train, y_train, X_test, ids_train, ids_test) -> None:
    s = "3 leakage"
    check(s, "train/test patient ids disjoint and cover all 768 rows",
          not set(ids_train) & set(ids_test) and len(set(ids_train) | set(ids_test)) == 768,
          f"train {len(ids_train)}, test {len(ids_test)}")
    tr = X_train.copy()
    tr[config.ZERO_AS_MISSING] = tr[config.ZERO_AS_MISSING].replace(0, np.nan)
    te = X_test.copy()
    te[config.ZERO_AS_MISSING] = te[config.ZERO_AS_MISSING].replace(0, np.nan)
    imputer = xgb.named_steps["preprocess"].named_steps["impute"]
    stats = dict(zip(F, imputer.statistics_))
    train_med = tr.median()
    check(s, "imputer medians == training-split medians",
          all(np.isclose(stats[f], train_med[f]) for f in F), str({f: stats[f] for f in config.ZERO_AS_MISSING}))
    allmed = pd.concat([tr, te]).median()
    differs = [f for f in config.ZERO_AS_MISSING if not np.isclose(train_med[f], allmed[f])]
    check(s, "imputer medians are not full-data medians (at least one differs)", bool(differs),
          f"differs for {differs}")
    meta = load_metadata()
    best = meta["best_params"]
    refit = build_xgb_pipeline(**{k.split("__")[-1]: v for k, v in best.items()}).fit(X_train, y_train)
    same = close(refit.predict_proba(X_test)[:, 1], xgb.predict_proba(X_test)[:, 1], 1e-6)
    check(s, "refitting best params on the 614 training rows only reproduces the saved model (in memory)",
          same, f"best params {best}")
    src = (ROOT / "train_model.py").read_text("utf-8")
    fit_lines = [ln for ln in src.splitlines() if re.search(r"\.fit\(|tune_xgb\(|cross_validate_pipeline\(", ln)]
    check(s, "no fit/tune/CV call in train_model.py receives test data",
          all("test" not in ln.lower() for ln in fit_lines), f"{len(fit_lines)} fit-type calls inspected")
    check(s, "decision threshold fixed at 0.5", config.DECISION_THRESHOLD == 0.5)


def shap_checks(xgb, tree, X_train, X_test) -> None:
    s = "4 SHAP additivity"
    saved = load("outputs/metrics/shap_additivity.json")
    for name, X in (("test", X_test), ("train", X_train)):
        e = sx.explain(tree, xgb, X)
        margin = sx.predict_margin(xgb, X)
        err = np.abs(e.base_values + e.values.sum(1) - margin).max()
        perr = np.abs(1 / (1 + np.exp(-(e.base_values + e.values.sum(1)))) - xgb.predict_proba(X)[:, 1]).max()
        naive = np.abs(1 / (1 + np.exp(-e.base_values)) + e.values.sum(1) - xgb.predict_proba(X)[:, 1]).max()
        check(s, f"{name}: base + sum(SHAP) == log-odds margin", err < 1e-4, f"max error {err:.2e}")
        check(s, f"{name}: sigmoid(base + sum(SHAP)) == P(positive)", perr < 1e-5, f"max error {perr:.2e}")
        check(s, f"{name}: errors match shap_additivity.json",
              close([err, naive], [saved[name]["max_abs_error_logodds"],
                                   saved[name]["max_abs_error_naive_probability_sum"]], 1e-6),
              f"naive prob + SHAP error {naive:.3f}")
    e = sx.explain(tree, xgb, X_test)
    sv = pd.read_csv(ROOT / "outputs/experiments/shap_values.csv").pivot(index="patient_id", columns="feature",
                                                                           values="shap_logodds")
    _, _, ids = load_split_csv(config.TEST_CSV)
    check(s, "shap_values.csv equals freshly computed SHAP values",
          close(sv.loc[ids, F].to_numpy(), e.values, 1e-6))


# --------------------------------------------------------------------------- #
def metrics_checks(xgb, base, X_test, y_test) -> None:
    s = "5 model metrics"
    saved = load("outputs/metrics/model_metrics.json")
    y = y_test.to_numpy().astype(int)
    probs = {"xgboost": xgb.predict_proba(X_test)[:, 1], "logistic_regression": base.predict_proba(X_test)[:, 1]}
    for name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        mine = {
            "accuracy": (tp + tn) / len(y), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "specificity": tn / (tn + fp), "f1": 2 * tp / (2 * tp + fp + fn),
            "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p),
        }
        bad = {k: (v, saved[name][k]) for k, v in mine.items() if not np.isclose(v, saved[name][k], atol=1e-12)}
        check(s, f"{name}: 8 metrics recomputed == model_metrics.json", not bad, str(bad) if bad else
              ", ".join(f"{k} {v:.3f}" for k, v in mine.items()))
        check(s, f"{name}: confusion matrix", [[tn, fp], [fn, tp]] == saved[name]["confusion_matrix"],
              f"TN {tn} FP {fp} FN {fn} TP {tp}")
        # Independent percentile bootstrap with the documented settings (2000 resamples, seed 42).
        rng = np.random.default_rng(config.RANDOM_SEED)
        aucs = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() != y[idx].max():
                aucs.append(roc_auc_score(y[idx], p[idx]))
        ci = [np.quantile(aucs, 0.025), np.quantile(aucs, 0.975)]
        check(s, f"{name}: ROC-AUC 95% CI reproduced from the test predictions", close(ci, saved[name]["roc_auc_ci95"]),
              f"{ci[0]:.4f}-{ci[1]:.4f}")
    rng = np.random.default_rng(config.RANDOM_SEED)
    diffs = []
    pa, pb = probs["xgboost"], probs["logistic_regression"]
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = saved["auc_difference_xgb_minus_lr"]
    ci = [np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)]
    check(s, "paired AUC difference and CI reproduced",
          close([roc_auc_score(y, pa) - roc_auc_score(y, pb), *ci], [d["observed_diff"], *d["ci95"]]),
          f"{d['observed_diff']:+.4f} ({ci[0]:+.4f} to {ci[1]:+.4f})")
    meta = load_metadata()
    check(s, "CV AUC in metadata (XGB 0.841±0.020, LR 0.843±0.019)",
          round(meta["cv_xgb_train_split"]["roc_auc"]["mean"], 3) == 0.841
          and round(meta["cv_baseline_train_split"]["roc_auc"]["mean"], 3) == 0.843)
    reported = {"accuracy": 0.766, "precision": 0.688, "recall": 0.611, "specificity": 0.850, "f1": 0.647,
                "roc_auc": 0.822, "average_precision": 0.681, "brier_score": 0.163}
    check(s, "README/report XGBoost figures match saved metrics to 3 d.p.",
          all(round(saved["xgboost"][k], 3) == v for k, v in reported.items()))


# --------------------------------------------------------------------------- #
def _rank(w: dict) -> list[str]:
    return sorted(w, key=lambda f: (-abs(w[f]), F.index(f)))


def _jac(a, b, k):
    sa, sb = set(_rank(a)[:k]), set(_rank(b)[:k])
    return len(sa & sb) / len(sa | sb)


def _rho(a, b):
    x, y = np.abs([a[f] for f in F]), np.abs([b[f] for f in F])
    return np.nan if np.ptp(x) == 0 or np.ptp(y) == 0 else spearmanr(x, y).statistic


def study_checks() -> pd.DataFrame:
    s = "6 explanation study"
    study = pd.read_csv(ROOT / "outputs/experiments/study.csv")
    lw = pd.read_csv(ROOT / "outputs/experiments/lime_weights.csv").pivot(index="patient_id", columns="feature",
                                                                          values="lime_weight")
    sv = pd.read_csv(ROOT / "outputs/experiments/shap_values.csv").pivot(index="patient_id", columns="feature",
                                                                         values="shap_logodds")
    stw = pd.read_csv(ROOT / "outputs/experiments/stability_weights.csv")
    mism = {k: 0 for k in ("top3_overlap", "jaccard_top3", "rank_spearman", "n_direction_disagreements",
                           "stability_jaccard_top3", "stability_spearman", "n_material_sign_flips",
                           "sign_consistency")}
    for _, r in study.iterrows():
        a, b = lw.loc[r.patient_id, F].to_dict(), sv.loc[r.patient_id, F].to_dict()
        mine = {
            "top3_overlap": len(set(_rank(a)[:3]) & set(_rank(b)[:3])) / 3,
            "jaccard_top3": _jac(a, b, 3),
            "rank_spearman": _rho(a, b),
            "n_direction_disagreements": sum(abs(a[f]) >= 0.01 and abs(b[f]) >= 0.05 and np.sign(a[f]) != np.sign(b[f])
                                             for f in F),
        }
        W = stw[stw.patient_id == r.patient_id].pivot(index="seed", columns="feature", values="lime_weight")[F]
        rows = [W.loc[i].to_dict() for i in W.index]
        pairs = list(combinations(range(len(rows)), 2))
        mine["stability_jaccard_top3"] = np.mean([_jac(rows[i], rows[j], 3) for i, j in pairs])
        mine["stability_spearman"] = np.nanmean([_rho(rows[i], rows[j]) for i, j in pairs])
        mine["n_material_sign_flips"] = sum(((W[f] >= 0.01).any() and (W[f] <= -0.01).any()) for f in F)
        sg = np.sign(W.to_numpy())
        mine["sign_consistency"] = (sg == sg[0]).all(0).mean()
        for k, v in mine.items():
            if not (np.isclose(v, r[k], atol=1e-9) or (np.isnan(v) and np.isnan(r[k]))):
                mism[k] += 1
    for k, n in mism.items():
        check(s, f"per-patient {k} recomputed from raw attribution tables", n == 0,
              f"{len(study)} patients, {n} mismatches")
    summ = load("outputs/metrics/explanation_study_summary.json")
    mine = {
        "top1_match_rate": study.top1_match.mean(), "mean_top3_overlap": study.top3_overlap.mean(),
        "mean_jaccard_top3": study.jaccard_top3.mean(), "median_rank_spearman": study.rank_spearman.median(),
        "share_with_direction_disagreement": (study.n_direction_disagreements > 0).mean(),
        "median_lime_r2": study.lime_r2.median(), "mean_stability_jaccard_top3": study.stability_jaccard_top3.mean(),
        "share_material_sign_flip_across_seeds": (study.n_material_sign_flips > 0).mean(),
    }
    check(s, "headline summary recomputed == explanation_study_summary.json",
          all(np.isclose(v, summ[k]) for k, v in mine.items()),
          ", ".join(f"{k} {v:.3f}" for k, v in mine.items()))
    check(s, "LIME stability uses 10 seeds x 5,000 samples per patient",
          summ["lime_seeds"] == list(range(10)) and summ["lime_num_samples"] == 5000
          and stw.groupby("patient_id").seed.nunique().eq(10).all())
    rel = pd.read_csv(ROOT / "outputs/experiments/relationships.csv")
    pairs = [("confidence", "lime_r2"), ("confidence", "lime_fidelity_gap"), ("confidence", "rank_spearman"),
             ("confidence", "jaccard_top3"), ("confidence", "stability_jaccard_top3"), ("lime_r2", "rank_spearman"),
             ("lime_r2", "stability_jaccard_top3"), ("stability_jaccard_top3", "rank_spearman")]
    ok = True
    for (x, yv), (_, row) in zip(pairs, rel.iterrows()):
        d = study[[x, yv]].dropna()
        res = spearmanr(d[x], d[yv])
        ok &= np.isclose(res.statistic, row.spearman_rho) and np.isclose(res.pvalue, row.p_value)
        ok &= bool(row.significant_bonferroni) == bool(res.pvalue < 0.05 / 8)
    check(s, "8 Spearman relationships and Bonferroni flags recomputed", bool(ok))
    bands = study.prob_positive.apply(lambda p: "high-confidence positive" if p >= 0.85 else
                                      "high-confidence negative" if p <= 0.15 else
                                      "borderline" if 0.40 <= p <= 0.60 else "moderate")
    check(s, "confidence bands recomputed from probabilities", (bands == study.confidence_band).all(),
          bands.value_counts().to_dict().__str__())
    return study


# --------------------------------------------------------------------------- #
def case_checks(xgb, tree, X_train, X_test, y_test, ids_test, study) -> None:
    s = "7 case studies"
    cases = load("outputs/case_studies/case_studies.json")
    expected = {1: 245, 2: 32, 3: 78, 4: 351, 5: 486}
    check(s, "selected patients are 245, 32, 78, 351, 486",
          {c["case_id"]: c["patient_id"] for c in cases} == expected)
    check(s, "select_cases on study.csv reproduces the same selection",
          {c.case_id: c.patient_id for c in select_cases(study)} == expected)
    st_ = study.set_index("patient_id")
    # Each rule checked directly with pandas.
    hp = study[study.prob_positive >= 0.85]
    hn = study[study.prob_positive <= 0.15]
    bl = study[study.prob_positive.between(0.4, 0.6)]
    check(s, "case 1 = highest P among P >= 0.85", hp.loc[hp.prob_positive.idxmax(), "patient_id"] == 245,
          f"P = {st_.loc[245].prob_positive:.3f}, {len(hp)} candidates")
    check(s, "case 2 = lowest P among P <= 0.15", hn.loc[hn.prob_positive.idxmin(), "patient_id"] == 32,
          f"P = {st_.loc[32].prob_positive:.3f}, {len(hn)} candidates")
    check(s, "case 3 = P closest to 0.5 within [0.4, 0.6]",
          bl.loc[(bl.prob_positive - 0.5).abs().idxmin(), "patient_id"] == 78,
          f"P = {st_.loc[78].prob_positive:.3f}, {len(bl)} candidates")
    rp = study[(study.jaccard_top3 <= 0.5) & study.rank_spearman.notna() & ~study.patient_id.isin([245, 32, 78])]
    check(s, "case 4 = lowest rank rho among top-3 Jaccard <= 0.5",
          rp.sort_values(["rank_spearman", "jaccard_top3"]).iloc[0].patient_id == 351,
          f"rho = {st_.loc[351].rank_spearman:.3f}, Jaccard = {st_.loc[351].jaccard_top3:.2f}")
    dp = study[(study.n_direction_disagreements > 0) & ~study.patient_id.isin([245, 32, 78, 351])]
    top = dp.sort_values(["n_direction_disagreements", "direction_disagreement_shap_mass"], ascending=False).iloc[0]
    check(s, "case 5 = most direction disagreements (tie-break SHAP mass)", top.patient_id == 486,
          f"{int(top.n_direction_disagreements)} features: {top.direction_disagreement_features}; "
          f"max in pool = {int(dp.n_direction_disagreements.max())}")

    lw = pd.read_csv(ROOT / "outputs/experiments/lime_weights.csv")
    stw = pd.read_csv(ROOT / "outputs/experiments/stability_weights.csv")
    pos = {int(p): i for i, p in enumerate(ids_test)}
    X_train_t = transform_features(xgb, X_train)
    for c in cases:
        pid = c["patient_id"]
        x_raw = X_test.iloc[[pos[pid]]]
        x_t = transform_features(xgb, x_raw)
        row = st_.loc[pid]
        case_lime = pd.DataFrame(c["lime_table"]).set_index("feature").lime_weight
        study_lime = lw[lw.patient_id == pid].set_index("feature").lime_weight
        check(s, f"case {c['case_id']}: LIME table == the study's LIME draw",
              close(case_lime.reindex(F), study_lime.reindex(F), 1e-12))
        cs = pd.DataFrame(c["stability_weights"]).set_index("seed")[F]
        ss = stw[stw.patient_id == pid].pivot(index="seed", columns="feature", values="lime_weight")[F]
        check(s, f"case {c['case_id']}: stability weights == the study's 10-seed draws", close(cs, ss, 1e-12))
        check(s, f"case {c['case_id']}: metrics block == study row",
              all((c["metrics"][k] == row[k]) or (isinstance(row[k], float) and np.isclose(c["metrics"][k], row[k]))
                  for k in ("lime_r2", "top3_overlap", "jaccard_top3", "rank_spearman", "n_direction_disagreements",
                            "stability_jaccard_top3")))
        sp = sx.explain_patient(tree, xgb, x_raw)
        check(s, f"case {c['case_id']}: SHAP margin/probability reproduce",
              np.isclose(sp.margin, c["shap_margin_logodds"], atol=1e-6)
              and np.isclose(sp.probability, c["prob_positive"], atol=1e-6)
              and sp.additivity_error < 1e-4, f"P = {sp.probability:.3f}, margin {sp.margin:+.3f}")
        # Same patient, same input, same class for both explainers.
        res = lx.explain_patient(xgb, lx.build_lime_explainer(X_train_t, 42), x_t, 2000, 42, keep_neighborhood=True)
        same_input = close(res.neighborhood_X[0], x_t.to_numpy()[0], 1e-9) and \
            close(list(c["model_input_values"].values()), x_t.to_numpy()[0], 1e-9) and \
            close(sx.explain(tree, xgb, x_raw).data[0], x_t.to_numpy()[0], 1e-9)
        check(s, f"case {c['case_id']}: LIME instance == SHAP instance == preprocessed patient", same_input)
        check(s, f"case {c['case_id']}: LIME explains P(class 1) == SHAP's sigmoid(margin)",
              np.isclose(res.model_prob, sp.probability, atol=1e-6) and np.isclose(res.neighborhood_probs[0], sp.probability, atol=1e-6))

    # Specific claims made in README / docs/experiment_report.md
    c = {x["case_id"]: x for x in cases}
    r = lambda pid: st_.loc[pid]  # noqa: E731
    sv = lambda case, f: next(t["shap_logodds"] for t in c[case]["shap_table"] if t["feature"] == f)  # noqa: E731
    lwt = lambda case, f: next(t["lime_weight"] for t in c[case]["lime_table"] if t["feature"] == f)  # noqa: E731
    lrule = lambda case, f: next(t["lime_rule"] for t in c[case]["lime_table"] if t["feature"] == f)  # noqa: E731
    claims = [
        ("case 1: P 0.934, truth positive, Jaccard 1.0, rho 0.93, R2 0.59, identical top-3 over 10 seeds",
         round(r(245).prob_positive, 3) == 0.934 and r(245).y_true == 1 and r(245).jaccard_top3 == 1.0
         and round(r(245).rank_spearman, 2) == 0.93 and round(r(245).lime_r2, 2) == 0.59
         and r(245).stability_jaccard_top3 == 1.0 and _rank(sx.explain_patient(tree, xgb, X_test.iloc[[pos[245]]]).values)[0] == "Glucose"),
        ("case 2: P 0.011, truth negative, no sign conflicts, surrogate predicts -0.20 (gap 0.21), Glucose <= 99 top in both",
         round(r(32).prob_positive, 3) == 0.011 and r(32).y_true == 0 and r(32).n_direction_disagreements == 0
         and round(r(32).lime_local_pred, 2) == -0.20 and round(r(32).lime_fidelity_gap, 2) == 0.21
         and c[2]["lime_table"][0]["feature"] == "Glucose" and "<= 99" in lrule(2, "Glucose")
         and c[2]["shap_table"][0]["feature"] == "Glucose"),
        ("case 3: P 0.508, truth positive, BMI > 36.5 leads both, Glucose sign conflict, R2 0.07, 3 imputed",
         round(r(78).prob_positive, 3) == 0.508 and r(78).y_true == 1 and c[3]["lime_table"][0]["feature"] == "BMI"
         and "36.50" in lrule(3, "BMI") and c[3]["shap_table"][0]["feature"] == "BMI"
         and "Glucose" in r(78).direction_disagreement_features and round(r(78).lime_r2, 2) == 0.07 and r(78).n_imputed == 3),
        ("case 4: P 0.455, truth negative, Jaccard 0, rho -0.24, LIME top = Insulin, SHAP top = Glucose +0.42 pushed negative by LIME, R2 0.03",
         round(r(351).prob_positive, 3) == 0.455 and r(351).y_true == 0 and r(351).jaccard_top3 == 0
         and round(r(351).rank_spearman, 2) == -0.24 and c[4]["lime_table"][0]["feature"] == "Insulin"
         and c[4]["shap_table"][0]["feature"] == "Glucose" and round(sv(4, "Glucose"), 2) == 0.42
         and lwt(4, "Glucose") < 0 and round(r(351).lime_r2, 2) == 0.03),
        ("case 5: P 0.459, truth negative, Glucose & Insulin opposite signs, LIME Glucose negative in all 10 seeds, SHAP Glucose +0.48, R2 0.09",
         round(r(486).prob_positive, 3) == 0.459 and r(486).y_true == 0
         and set(r(486).direction_disagreement_features.split(";")) == {"Glucose", "Insulin"}
         and (stw[(stw.patient_id == 486) & (stw.feature == "Glucose")].lime_weight < 0).all()
         and round(sv(5, "Glucose"), 2) == 0.48 and round(r(486).lime_r2, 2) == 0.09),
    ]
    for name, ok in claims:
        check(s, f"documented claim - {name}", bool(ok))
    g = {k: lwt(k, "Glucose") for k in (3, 4, 5)}
    raw_g = {k: c[k]["raw_values"]["Glucose"] for k in (3, 4, 5)}
    check(s, "cases 3-5: Glucose in (117, 140] bin with LIME weight between -0.011 and -0.022",
          all(117 < raw_g[k] <= 140 and "117.00 < Glucose <= 140.00" in lrule(k, "Glucose")
              and -0.0225 <= g[k] <= -0.0105 for k in g),
          ", ".join(f"case {k}: glucose {raw_g[k]:.0f}, w {g[k]:+.4f}" for k in g))
    q = np.percentile(X_train_t["Insulin"], [25, 50, 75])
    check(s, "Insulin quartile edges collapse to 120 / 125 / 130", close(q, [120, 125, 130], 1e-9), str(q))


def reproducibility() -> None:
    s = "8 reproducibility"
    e = sx.explain(sx.build_tree_explainer(load_pipeline(config.XGB_MODEL_PATH)),
                   load_pipeline(config.XGB_MODEL_PATH), load_split_csv(config.TEST_CSV)[0])
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
        viz.save(viz.plot_shap_beeswarm(e), a)
        viz.save(viz.plot_shap_beeswarm(e), b)
        same = a.read_bytes() == b.read_bytes()
    check(s, "SHAP beeswarm renders byte-identically twice (seeded jitter)", same)
    pipe = load_pipeline(config.XGB_MODEL_PATH)
    X_train = load_split_csv(config.TRAIN_CSV)[0]
    X_test = load_split_csv(config.TEST_CSV)[0]
    Xt = transform_features(pipe, X_train)
    x = transform_features(pipe, X_test.iloc[[0]])
    w1 = lx.explain_patient(pipe, lx.build_lime_explainer(Xt, 42), x, 2000, 42).weights
    w2 = lx.explain_patient(pipe, lx.build_lime_explainer(Xt, 42), x, 2000, 42).weights
    check(s, "fresh LIME explainer with the same seed gives identical weights (dashboard behaviour)", w1 == w2)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    xgb = load_pipeline(config.XGB_MODEL_PATH)
    base = load_pipeline(config.BASELINE_MODEL_PATH)
    X_train, y_train, ids_train = load_split_csv(config.TRAIN_CSV)
    X_test, y_test, ids_test = load_split_csv(config.TEST_CSV)
    tree = sx.build_tree_explainer(xgb)
    provenance()
    positive_class(xgb, X_test, y_test)
    leakage(xgb, X_train, y_train, X_test, ids_train, ids_test)
    shap_checks(xgb, tree, X_train, X_test)
    metrics_checks(xgb, base, X_test, y_test)
    study = study_checks()
    case_checks(xgb, tree, X_train, X_test, y_test, ids_test, study)
    reproducibility()
    n_fail = sum(not r["passed"] for r in results)
    (OUT / "results_check.json").write_text(json.dumps(
        {"n_checks": len(results), "n_passed": len(results) - n_fail, "n_failed": n_fail, "checks": results},
        indent=2, default=str), "utf-8")
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
