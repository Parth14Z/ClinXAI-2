"""Evaluate the trained models and run the LIME-vs-SHAP consistency study.

Produces every report artefact under outputs/:
  metrics/      model metrics, SHAP additivity, study summaries
  figures/      evaluation, SHAP, LIME, comparison, study and architecture figures
  experiments/  per-patient study tables (agreement, stability, raw attributions)
  case_studies/ five rule-selected held-out cases with figures and interpretation

Usage:  python evaluate.py [--num-samples 5000] [--n-seeds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src import config
from src import plotting as viz
from src import shap_explainer as sx
from src.case_studies import cases_frame, interpret_case, select_cases
from src.data import format_patient_id, load_split_csv
from src.experiment import analyse_relationships, band_summary, overall_summary, run_study
from src.metrics import (bootstrap_auc_ci, classification_metrics, paired_bootstrap_auc_diff,
                         pr_points, roc_points)
from src.model import load_metadata, load_pipeline


def jdump(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def evaluate_models(xgb, base, X_test, y_test) -> dict:
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    p_lr = base.predict_proba(X_test)[:, 1]
    m_xgb = classification_metrics(y_test, p_xgb)
    m_lr = classification_metrics(y_test, p_lr)
    m_xgb["roc_auc_ci95"] = bootstrap_auc_ci(y_test, p_xgb)
    m_lr["roc_auc_ci95"] = bootstrap_auc_ci(y_test, p_lr)
    result = {"xgboost": m_xgb, "logistic_regression": m_lr,
              "auc_difference_xgb_minus_lr": paired_bootstrap_auc_diff(y_test.to_numpy(), p_xgb, p_lr)}
    jdump(result, config.METRICS_DIR / "model_metrics.json")

    keys = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision"]
    table = pd.DataFrame({"XGBoost": [m_xgb[k] for k in keys],
                          "Logistic regression": [m_lr[k] for k in keys]}, index=keys)
    table.round(4).to_csv(config.METRICS_DIR / "model_comparison.csv")

    viz.save(viz.plot_confusion_matrix(m_xgb["confusion_matrix"], "XGBoost confusion matrix (test set)"),
             config.FIGURES_DIR / "confusion_matrix.png")
    viz.save(viz.plot_confusion_matrix(m_lr["confusion_matrix"], "Logistic regression confusion matrix"),
             config.FIGURES_DIR / "confusion_matrix_baseline.png")
    viz.save(viz.plot_roc_curves({"XGBoost": roc_points(y_test, p_xgb),
                                  "Logistic regression": roc_points(y_test, p_lr)}),
             config.FIGURES_DIR / "roc_curve.png")
    viz.save(viz.plot_pr_curves({"XGBoost": pr_points(y_test, p_xgb),
                                 "Logistic regression": pr_points(y_test, p_lr)}, float(y_test.mean())),
             config.FIGURES_DIR / "pr_curve.png")
    viz.save(viz.plot_model_comparison(table[table.index != "average_precision"]),
             config.FIGURES_DIR / "model_comparison.png")
    for name, m in (("XGBoost", m_xgb), ("LogReg", m_lr)):
        print(f"[test:{name}] acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
              f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f} (95% CI {m['roc_auc_ci95'][0]:.3f}-"
              f"{m['roc_auc_ci95'][1]:.3f})")
    return result


def shap_global(xgb, X_train, X_test) -> None:
    tree = sx.build_tree_explainer(xgb)
    exp_test = sx.explain(tree, xgb, X_test)
    checks = {"test": sx.verify_additivity(exp_test, xgb, X_test),
              "train": sx.verify_additivity(sx.explain(tree, xgb, X_train), xgb, X_train),
              "base_value_logodds": sx.base_value(tree),
              "base_value_as_probability": float(sx.sigmoid(sx.base_value(tree))),
              "units": sx.OUTPUT_UNITS}
    jdump(checks, config.METRICS_DIR / "shap_additivity.json")
    print(f"[shap] additivity max |error| test={checks['test']['max_abs_error_logodds']:.2e} log-odds; "
          f"naive prob-sum error={checks['test']['max_abs_error_naive_probability_sum']:.3f}")
    imp = sx.global_importance(exp_test)
    imp.rename("mean_abs_shap_logodds").to_csv(config.METRICS_DIR / "shap_global_importance.csv")
    viz.save(viz.plot_shap_importance(imp), config.FIGURES_DIR / "shap_global_importance.png")
    viz.save(viz.plot_shap_beeswarm(exp_test), config.FIGURES_DIR / "shap_beeswarm.png")


def build_case_studies(xgb, X_test, ids_test, study, explanations) -> list[dict]:
    """Case artefacts reuse the exact LIME and stability draws recorded by run_study."""
    cases = select_cases(study)
    cases_frame(cases, study).to_csv(config.CASE_STUDIES_DIR / "case_studies_summary.csv", index=False)
    tree = sx.build_tree_explainer(xgb)
    pos = {int(p): i for i, p in enumerate(ids_test)}
    srow = study.set_index("patient_id")
    records = []
    for c in cases:
        rec = c.to_dict()
        if not c.available:
            rec["interpretation"] = [c.selection_note]
            records.append(rec)
            print(f"[case {c.case_id}] unavailable: {c.title}")
            continue
        i = pos[c.patient_id]
        x_raw = X_test.iloc[[i]]
        lime_res, stab = explanations[c.patient_id]
        shap_p = sx.explain_patient(tree, xgb, x_raw)
        row = srow.loc[c.patient_id]
        lt, st = lime_res.table(), shap_p.table()
        label = f"Case {c.case_id} · {format_patient_id(c.patient_id)}"
        prefix = config.CASE_STUDIES_DIR / f"case{c.case_id}"
        exp_row = sx.explain(tree, xgb, x_raw)[0]
        dis = row.direction_disagreement_features.split(";") if row.direction_disagreement_features else []
        figs = {
            "lime": viz.save(viz.plot_lime_explanation(lime_res.weights, lime_res.rules,
                                                       f"{label}: LIME explanation"), f"{prefix}_lime.png"),
            "shap_waterfall": viz.save(viz.plot_shap_waterfall(exp_row, f"{label}: SHAP waterfall"),
                                       f"{prefix}_shap_waterfall.png"),
            "shap_force": viz.save(viz.plot_shap_force(shap_p.base_value, exp_row.values,
                                                       exp_row.display_data), f"{prefix}_shap_force.png"),
            "side_by_side": viz.save(viz.plot_side_by_side(lime_res.weights, shap_p.values,
                                                           f"{label}: LIME vs SHAP", dis),
                                     f"{prefix}_side_by_side.png"),
            "rank_slope": viz.save(viz.plot_rank_slope(lime_res.weights, shap_p.values,
                                                       f"{label}: rank shift"), f"{prefix}_rank_slope.png"),
            "stability": viz.save(viz.plot_stability(stab.weights, f"{label}: LIME weights"),
                                  f"{prefix}_stability.png"),
            "neighborhood": viz.save(viz.plot_neighborhood(lime_res.neighborhood_probs, lime_res.model_prob),
                                     f"{prefix}_neighborhood.png"),
        }
        rec.update({
            "patient": format_patient_id(c.patient_id),
            "raw_values": {f: float(x_raw.iloc[0][f]) for f in config.FEATURES},
            "model_input_values": shap_p.model_input,
            "imputed": shap_p.imputed,
            "y_true": int(row.y_true), "y_pred": int(row.y_pred),
            "prob_positive": float(row.prob_positive),
            "shap_base_value_logodds": shap_p.base_value, "shap_margin_logodds": shap_p.margin,
            "shap_additivity_error": shap_p.additivity_error,
            "metrics": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in (
                "confidence", "confidence_band", "lime_r2", "lime_local_pred", "lime_fidelity_gap",
                "top1_match", "top3_overlap", "top5_overlap", "jaccard_top3", "jaccard_top5",
                "rank_spearman", "sign_agreement", "n_direction_disagreements",
                "direction_disagreement_features", "stability_jaccard_top3", "stability_spearman",
                "sign_consistency", "sign_flip_features", "material_sign_flip_features",
                "r2_mean", "r2_std")},
            "lime_table": lt.to_dict(orient="records"),
            "shap_table": st.to_dict(orient="records"),
            "stability_weights": stab.weights.reset_index().to_dict(orient="records"),
            "neighborhood": lime_res.neighborhood_summary(),
            "interpretation": interpret_case(c, row, lt, st),
            "figures": {k: str(v.relative_to(config.PROJECT_ROOT)).replace("\\", "/") for k, v in figs.items()},
        })
        records.append(rec)
        print(f"[case {c.case_id}] {c.title}: {format_patient_id(c.patient_id)} - {c.selection_note}")

    jdump(records, config.CASE_STUDIES_DIR / "case_studies.json")
    write_case_markdown(records)
    # Representative evidence figures (case 1) under figures/ for the report
    first = next((r for r in records if r.get("available")), None)
    if first:
        import shutil
        for key, name in (("lime", "lime_patient_explanation.png"), ("shap_waterfall", "shap_waterfall.png"),
                          ("shap_force", "shap_force.png"), ("side_by_side", "lime_shap_side_by_side.png"),
                          ("stability", "lime_stability_example.png"),
                          ("neighborhood", "lime_neighborhood_example.png")):
            shutil.copy(config.PROJECT_ROOT / first["figures"][key], config.FIGURES_DIR / name)
    return records


def write_case_markdown(records: list[dict]) -> None:
    lines = ["# ClinXAI case studies (held-out patients)", "",
             "Selected automatically by the pre-declared rules in `src/case_studies.py`. "
             "Probabilities are uncalibrated XGBoost scores; SHAP values are log-odds; "
             "LIME weights are local-surrogate coefficients.", ""]
    for r in records:
        lines += [f"## Case {r['case_id']}: {r['title']}", "", f"*Rule:* {r['rule']}", ""]
        if not r["available"]:
            lines += [f"**Unavailable.** {r['selection_note']}", ""]
            continue
        lines += [f"*Patient:* {r['patient']} · *Selection:* {r['selection_note']}", ""]
        lines += [f"- {s}" for s in r["interpretation"]] + [""]
        shap_rows = {t["feature"]: (i + 1, t["shap_logodds"]) for i, t in enumerate(r["shap_table"])}
        lime_rows = {t["feature"]: (i + 1, t["lime_weight"], t["lime_rule"]) for i, t in enumerate(r["lime_table"])}
        conflicts = set(filter(None, r["metrics"]["direction_disagreement_features"].split(";")))
        lines += ["| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight "
                  "| LIME rank | Material sign conflict |", "|---|---|---|---|---|---|---|---|---|"]
        for f in config.FEATURES:
            s_rank, s_val = shap_rows[f]
            l_rank, l_val, rule = lime_rows[f]
            imputed = " (imputed)" if r["imputed"][f] else ""
            lines.append(f"| {f} | {r['raw_values'][f]:g} | {r['model_input_values'][f]:g}{imputed} | {s_val:+.3f} "
                         f"| {s_rank} | `{rule}` | {l_val:+.4f} | {l_rank} | {'yes' if f in conflicts else ''} |")
        lines += ["", f"SHAP: base {r['shap_base_value_logodds']:+.3f} + sum = margin "
                      f"{r['shap_margin_logodds']:+.3f} log-odds -> P = {r['prob_positive']:.3f}. "
                      f"LIME: R² {r['metrics']['lime_r2']:.3f}, surrogate prediction "
                      f"{r['metrics']['lime_local_pred']:.3f}.", ""]
    (config.CASE_STUDIES_DIR / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def write_findings(model_metrics, overall, rel, bands, records, meta) -> None:
    x, b = model_metrics["xgboost"], model_metrics["logistic_regression"]
    d = model_metrics["auc_difference_xgb_minus_lr"]
    L = ["# ClinXAI - generated findings", "",
         "All numbers below are computed by `evaluate.py` from the saved model and the untouched "
         "154-patient held-out split.", "", "## Model performance (threshold 0.5, uncalibrated)", "",
         "| Metric | XGBoost | Logistic regression |", "|---|---|---|"]
    for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "brier_score"):
        L.append(f"| {k} | {x[k]:.3f} | {b[k]:.3f} |")
    L += ["", f"XGBoost test ROC-AUC 95% bootstrap CI: {x['roc_auc_ci95'][0]:.3f}-{x['roc_auc_ci95'][1]:.3f}. "
          f"Paired bootstrap AUC difference (XGB - LR): {d['observed_diff']:+.3f} "
          f"(95% CI {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}).",
          f"Training-split 5-fold CV ROC-AUC: XGBoost {meta['cv_xgb_train_split']['roc_auc']['mean']:.3f} "
          f"± {meta['cv_xgb_train_split']['roc_auc']['std']:.3f}; logistic regression "
          f"{meta['cv_baseline_train_split']['roc_auc']['mean']:.3f} ± "
          f"{meta['cv_baseline_train_split']['roc_auc']['std']:.3f}.", "",
          "## LIME vs SHAP agreement across all held-out patients", ""]
    for k, v in overall.items():
        L.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")
    L += ["", "## Relationships (Spearman; Bonferroni threshold "
          f"{rel['bonferroni_threshold'].iloc[0]:.4f})", "",
          "| Relationship | rho | p | strength | sig. 0.05 | sig. Bonferroni |", "|---|---|---|---|---|---|"]
    for _, r in rel.iterrows():
        L.append(f"| {r.relationship} | {r.spearman_rho:+.3f} | {r.p_value:.3g} | {r.strength} | "
                 f"{'yes' if r['significant_alpha_0.05'] else 'no'} | "
                 f"{'yes' if r.significant_bonferroni else 'no'} |")
    cols = ["n_patients", "prob_positive", "lime_r2", "jaccard_top3", "rank_spearman",
            "n_direction_disagreements", "stability_jaccard_top3"]
    L += ["", "## Mean metrics by confidence band", "", "| band | " + " | ".join(cols) + " |",
          "|---" * (len(cols) + 1) + "|"]
    L += [f"| {band} | " + " | ".join(f"{bands.loc[band, c]:g}" for c in cols) + " |" for band in bands.index]
    L += ["", "## Case studies", ""]
    for r in records:
        status = r.get("patient", "unavailable")
        L.append(f"- Case {r['case_id']} ({r['title']}): {status} - {r['selection_note']}")
    (config.OUTPUTS_DIR / "findings.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=config.LIME_NUM_SAMPLES)
    ap.add_argument("--n-seeds", type=int, default=len(config.LIME_STABILITY_SEEDS))
    args = ap.parse_args()
    seeds = tuple(config.LIME_STABILITY_SEEDS[: args.n_seeds])
    config.ensure_directories()

    try:
        xgb = load_pipeline(config.XGB_MODEL_PATH)
        base = load_pipeline(config.BASELINE_MODEL_PATH)
        meta = load_metadata()
        X_train, _, _ = load_split_csv(config.TRAIN_CSV)
        X_test, y_test, ids_test = load_split_csv(config.TEST_CSV)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 1

    summary = json.loads((config.METRICS_DIR / "dataset_summary.json").read_text("utf-8"))
    viz.save(viz.plot_dataset_overview(summary), config.FIGURES_DIR / "dataset_overview.png")
    viz.save(viz.plot_architecture(), config.FIGURES_DIR / "architecture.png")

    model_metrics = evaluate_models(xgb, base, X_test, y_test)
    shap_global(xgb, X_train, X_test)

    t0 = time.perf_counter()

    def progress(i, n):
        if i % 20 == 0 or i == n:
            print(f"[study] {i}/{n} patients ({time.perf_counter() - t0:.0f}s)", flush=True)

    explanations: dict = {}
    res = run_study(xgb, X_train, X_test, y_test, ids_test, seeds, args.num_samples, progress, explanations)
    study = res["study"]
    for name, df in res.items():
        df.to_csv(config.EXPERIMENTS_DIR / f"{name}.csv", index=False)
    rel = analyse_relationships(study)
    rel.to_csv(config.EXPERIMENTS_DIR / "relationships.csv", index=False)
    bands = band_summary(study)
    bands.to_csv(config.EXPERIMENTS_DIR / "band_summary.csv")
    overall = overall_summary(study)
    overall["lime_num_samples"] = args.num_samples
    overall["lime_seeds"] = list(seeds)
    jdump(overall, config.METRICS_DIR / "explanation_study_summary.json")
    viz.save(viz.plot_relationships(study, rel), config.FIGURES_DIR / "confidence_relationships.png")
    viz.save(viz.plot_band_summary(bands), config.FIGURES_DIR / "band_summary.png")
    viz.save(viz.plot_agreement_distribution(study), config.FIGURES_DIR / "agreement_distribution.png")
    viz.save(viz.plot_stability_distribution(study), config.FIGURES_DIR / "stability_distribution.png")
    print(f"[study] top-3 overlap {overall['mean_top3_overlap']:.3f}, median rank rho "
          f"{overall['median_rank_spearman']:.3f}, direction disagreement in "
          f"{100 * overall['share_with_direction_disagreement']:.1f}% of patients, "
          f"seed stability {overall['mean_stability_jaccard_top3']:.3f}")

    records = build_case_studies(xgb, X_test, ids_test, study, explanations)
    write_findings(model_metrics, overall, rel, bands, records, meta)
    print("[done] outputs written to", config.OUTPUTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
