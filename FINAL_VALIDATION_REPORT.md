# ClinXAI: final technical validation report

**Date:** 27 October 2026 · **Scope:** the diabetes-focused ClinXAI project in this repository · **Status:** all checks below were executed. Nothing is reported as passed unless it actually ran.

| Validation layer | Tool | Result |
|---|---|---|
| Automated test suite | `pytest` 9.1.1, Python 3.12.10 | **92 passed, 0 failed, 0 skipped** (23.5 s) |
| Independent re-verification of every reported number | `validation/check_results.py` | **89 / 89 checks passed** |
| Real-browser click-through of the dashboard | Playwright 1.64.1 driving Microsoft Edge 156.0.4314.42 (headless) | **29 / 29 checks passed**, 0 console errors, 0 page errors |
| Concurrent visitors (public-link readiness, section 11) | Playwright, 6 simultaneous Edge sessions × 4 rounds | **24 / 24 sessions passed**, 0 app exceptions in the server log |

The model was **not retrained**. `evaluate.py` was rerun to apply the fixes in section 2, and every study output stayed byte-identical (section 2.1).

---

## 1. Automated tests

Command: `python -m pytest`. The full output is in [outputs/test_results.txt](outputs/test_results.txt).

| Test file | Tests | Result |
|---|---|---|
| `tests/test_data.py` | 14 | passed |
| `tests/test_preprocessing.py` | 15 | passed |
| `tests/test_model.py` | 12 | passed |
| `tests/test_explainers.py` | 12 | passed |
| `tests/test_comparison_and_cases.py` | 14 | passed |
| `tests/test_plotting_and_app.py` | 25 | passed |
| **Total** | **92** | **92 passed** |

At the start of this validation the suite was **90 passed** (12.3 s). Two regression tests were added: `test_run_study_exposes_the_exact_draws_behind_its_table` (reproducibility fix, section 2) and `test_concurrent_sessions_get_their_own_shap_figures` (concurrency fix, section 11).

---

## 2. Issues found during validation, and how they were resolved

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | **Case-study LIME artefacts were drawn separately from the study.** `LimeTabularExplainer` keeps one random generator that advances on every call. The study explained all 154 patients with one explainer. The case-study step then re-explained the 5 case patients with a fresh explainer, so the case tables and figures came from a different random draw than the study metrics quoted beside them (max weight difference 0.032). | Medium: internal inconsistency in report evidence | `run_study` now hands over its exact per-patient LIME and stability results ([src/experiment.py](src/experiment.py)), and [evaluate.py](evaluate.py) builds the case artefacts from them. Verified: case tables equal the study tables exactly (section 5). |
| 2 | **The dashboard's LIME depended on browsing history.** One cached explainer per seed was shared, so a patient's explanation depended on which patients had been viewed before. | Medium: not reproducible across sessions | [app.py](app.py) now builds a fresh explainer per explanation, so the same patient, seed and sample count always gives the same result. Verified by check 8 and in the browser. |
| 3 | **The SHAP beeswarm was not reproducible.** `shap.plots.beeswarm` jitters its dots with NumPy's unseeded global generator. | Low: cosmetic, same data | Seeded before plotting ([src/plotting.py](src/plotting.py)). Verified: two renders are byte-identical. |
| 4 | **One documented number came from the superseded draw.** The report gave LIME's Glucose weight for cases 3–5 as "−0.010 to −0.016"; the study's draw gives −0.022, −0.013 and −0.011. | Low | Corrected to "−0.011 to −0.022" in [docs/experiment_report.md](docs/experiment_report.md). |
| 5 | **Direction conflicts sit close to the materiality tolerance** (details in section 6.3). Of the 44 material LIME–SHAP sign conflicts, 31 have \|LIME weight\| < 0.02, against a 0.01 tolerance. | Interpretation caveat | A robustness caveat was added to [README.md](README.md) and [docs/experiment_report.md](docs/experiment_report.md). No numbers were changed. |
| 6 | The LIME seed-stability chart stretched across the full page width at an oversized scale. | Cosmetic | Placed in a column with a reading note (layout only). |
| 7 | No per-case feature tables existed in Markdown, which the report evidence requires. | Evidence gap | [outputs/case_studies/case_studies.md](outputs/case_studies/case_studies.md) now has one 8-feature table per case: values, imputation flags, SHAP and LIME values and ranks, and conflicts. |

### 2.1 Proof that the experimental findings were preserved

Before the fixes, all 76 output files were backed up and then diffed against the regenerated outputs.
- **Byte-identical:** all metrics JSON/CSV, all `outputs/experiments/*.csv` (study, LIME weights, SHAP values, stability weights, relationships, bands), `findings.md`, `case_studies_summary.csv` and every study and evaluation figure.
- **Changed, as intended:** only the LIME-derived case-study figures and tables (now the study's own draw), the copies of case 1 figures in `figures/`, and the beeswarm (seeded jitter).

The five selected patients and every headline statistic are unchanged. In the case narratives only the third decimal of the "largest LIME weight" changed, with the same feature and rule in every case (for example, case 1 Glucose +0.446 → +0.440).

A second full run of `evaluate.py` reproduced every output byte-for-byte, apart from the log's timings.

---

## 3. Browser and UI verification

- **Command:** `python validation/browser_check.py` against `streamlit run app.py`. Results are in [outputs/validation/browser_check.json](outputs/validation/browser_check.json).
- **Browser:** Microsoft Edge 156.0.4314.42 (Chromium engine), headless, 1440 × 900 viewport, freshly restarted server.
- **Playwright:** installed in a separate scratch environment. It is not added to `requirements.txt`.

| Requirement | Check performed in the browser | Result |
|---|---|---|
| All seven pages load | Each page visited via the sidebar; no `stException` element on any page | **Pass** (7/7) |
| Patient selection updates probabilities | Selector: PIMA-0245 (P 0.934) → PIMA-0032 (P 0.011) → PIMA-0078 (P 0.508) | **Pass** |
| What-if editor works | PIMA-0245, Glucose 184 → 120, *Apply*: P 0.934 → **0.735**. The edit carries over to the comparison page. *Reset* restores P 0.934 | **Pass** |
| LIME visualisations render | LIME bar chart and neighbourhood histogram load (2 images, 0 broken); R² 0.594 shown | **Pass** |
| LIME stability works | Toggle re-runs 5 seeds; KPIs shown (top-3 Jaccard 1.00, ρ 0.96, 88% stable signs, R² 0.58 ± 0.02); stability chart loads | **Pass** |
| SHAP beeswarm, waterfall and force plots render | Global tab: importance + beeswarm + additivity text. Patient tab: waterfall + force plot, base −0.622, margin +2.645 | **Pass** |
| Comparison page | Overlap/Jaccard/ρ/conflict KPIs, side-by-side chart, rank-slope chart, "why they differ" text | **Pass** |
| Evaluation charts and tables | Tab 1: metrics table, CI text (0.746–0.883, ΔAUC +0.009), confusion/ROC/PR/comparison charts. Tab 2: KPIs 73%/0.77/0.82/26%, 3 figures, relationship and band tables. Tab 3: 0.88 / 11% KPIs, distribution figure, per-patient table | **Pass** |
| All five case studies load | Cases 1–5 each show the correct patient and P, and 6 figures with 0 broken images | **Pass** (5/5) |
| Case → live pages | "Open PIMA-0486 …" sets the sidebar patient to PIMA-0486 and the comparison page follows | **Pass** |
| Browser console | JavaScript page errors: **0**. Console errors: **0** | **Pass** |

**Screenshots** (genuine, captured by the run above) are in [outputs/screenshots/](outputs/screenshots/):

| File | Content |
|---|---|
| `01_home.png`, `01_home_viewport.png` | Home: hero, disclaimer, KPIs, provenance, zero-placeholder chart, feature table, architecture |
| `02_prediction_PIMA-0245.png` | Prediction, probability meter, uncalibrated note, recorded vs model values |
| `03_prediction_whatif_glucose120.png` | What-if profile, P = 0.735, imputed Insulin flagged |
| `04_lime_PIMA-0245.png`, `05_lime_stability_PIMA-0245.png` | LIME explanation, neighbourhood, 5-seed stability |
| `06_shap_global_beeswarm.png`, `07_shap_waterfall_force_PIMA-0245.png` | SHAP global and patient views |
| `08_comparison_PIMA-0245.png` | LIME vs SHAP comparison |
| `09_…`, `10_…`, `11_evaluation_*.png` | The three evaluation tabs |
| `12_case1_…` to `16_case5_….png` | The five case studies |
| `17_case5_live_comparison_PIMA-0486.png` | Case 5 patient opened in the live comparison page |

**What the browser run does not cover**
- Six of the 18 screenshots were inspected by eye (home, what-if, LIME stability, SHAP patient, study tab, case 4); the rest were checked programmatically for loaded, non-broken images and expected text.
- Only Edge/Chromium at desktop width was tested. Firefox, Safari and narrow (mobile) layouts were not.
- Out-of-range what-if input is blocked by the number inputs' own bounds. The `validate_patient_input` rejection paths are covered by 10 unit tests, not by the browser run.

---

## 4. Model and data verification (`validation/check_results.py`)

These checks recompute each result with their own NumPy, scikit-learn and SciPy code; the project is used only to load the saved model and data. Full output: [outputs/validation/results_check.json](outputs/validation/results_check.json).

| Area | What was checked | Result |
|---|---|---|
| Provenance and checksum | ARFF MD5 `3cbaa3e54586aa88cf6aacb4033e4470` = OpenML's published value = `provenance.json`; SHA-256 matches; **a copy with one corrupted byte is rejected** by `ensure_dataset`; 768 × 8, 500/268 | Pass |
| Positive class | 268 raw `tested_positive` rows = 268 encoded `Outcome = 1`; `classes_ = [0, 1]`; `predict_proba[:, 1]` = sigmoid(raw margin) to 1e-6; column 1 gives AUC 0.822 (> 0.5, so not inverted) | Pass |
| No data leakage | Train/test ids disjoint and cover all 768 rows; imputer medians equal the **training-split** medians (Glucose 117, BP 72, Skin 29, Insulin 125, BMI 32.4); refitting the chosen settings on the 614 training rows alone reproduces the saved model's test predictions (in memory, nothing saved); no fit/tune/CV call in `train_model.py` receives test data; threshold fixed at 0.5 | Pass |
| SHAP additivity (log-odds) | base + ΣSHAP = margin: max error 1.73e-6 (test), 2.58e-6 (train); sigmoid(base + ΣSHAP) = P: 1.6e-7; the naive "probability + SHAP" sum is off by up to 3.56 (test) / 3.73 (train), demonstrating the unit trap; saved `shap_values.csv` equals fresh values | Pass |
| Same patient, same input, same class | For all 5 cases: LIME's explained instance (row 0 of its neighbourhood) = SHAP's explained row = the preprocessed patient; LIME's target P(class 1) = SHAP's sigmoid(margin) | Pass |
| Metrics match saved outputs | All 8 metrics and the confusion matrix for both models recomputed from the test predictions with plain formulas and scikit-learn match `model_metrics.json` to 1e-12; the README/report values match to 3 d.p. | Pass |
| Confidence intervals | Percentile bootstrap re-implemented (2,000 resamples, seed 42, single-class resamples skipped) reproduces the XGBoost CI 0.7457–0.8831, the LR CI 0.7396–0.8729 and the paired ΔAUC +0.0087 (−0.0177 to +0.0370) exactly from the genuine test predictions | Pass |
| LIME–SHAP comparison | Top-3 overlap, Jaccard, Spearman ρ and direction conflicts recomputed for all 154 patients from the raw attribution tables: 0 mismatches | Pass |
| LIME stability statistics | Pairwise top-3 Jaccard, pairwise ρ, sign consistency and material sign flips recomputed from `stability_weights.csv` (154 patients × 10 seeds × 8 features) by independent code: 0 mismatches; 10 seeds × 5,000 samples confirmed | Pass |
| Study summaries | Headline summary, all 8 Spearman relationships (ρ, p, Bonferroni flags) and the confidence bands recomputed | Pass |
| Reproducibility | Two full runs of `evaluate.py` are byte-identical; the beeswarm renders identically twice; the dashboard's LIME gives identical weights for the same patient and seed | Pass |

**Note on leakage:** `train_model.py` reads the test split only to log its size and positive rate, and to report its zero counts in `preprocessing_summary.json`. None of these values feeds into fitting, tuning or the threshold. Training-split medians differ from full-data medians only for BMI (32.4 vs 32.3); for the other four features they happen to coincide.

---

## 5. Case-study verification

Each selection rule was re-applied directly with pandas, and `select_cases` was re-run. Both give the same five patients. In every case, the LIME table, the 10-seed stability weights and the metrics block equal the study's own rows exactly, and SHAP's margin and probability reproduce.

| Case | Patient | Rule re-checked | Verified evidence |
|---|---|---|---|
| 1 High-confidence positive | **PIMA-0245** | highest P among the 12 patients with P ≥ 0.85 | P 0.934, truth positive, margin +2.645; Glucose leads in both methods; top-3 Jaccard 1.0, ρ 0.93, LIME R² 0.59; identical top-3 under all 10 seeds |
| 2 High-confidence negative | **PIMA-0032** | lowest P among the 61 patients with P ≤ 0.15 | P 0.011, truth negative; `Glucose <= 99` leads in both methods; no sign conflicts; the LIME surrogate predicts −0.20 (an impossible probability), gap 0.21 |
| 3 Borderline | **PIMA-0078** | P closest to 0.5 among the 22 patients in [0.40, 0.60] | P 0.508, truth positive; `BMI > 36.50` leads in both methods; Glucose sign conflict (SHAP +0.250, LIME −0.022); R² 0.07; 3 imputed values |
| 4 Ranking disagreement | **PIMA-0351** | lowest ρ among patients with top-3 Jaccard ≤ 0.5 | P 0.455, truth negative; top-3 Jaccard **0**, ρ **−0.238**; LIME ranks the imputed Insulin bin (`120 < Insulin <= 125`, value imputed to 125) first; SHAP ranks Glucose first (+0.421), which LIME pushes slightly negative (−0.013); R² 0.03 |
| 5 Directional disagreement | **PIMA-0486** | most material conflicts (2, the maximum in the pool; tie-break by SHAP mass) | P 0.459, truth negative; Glucose (SHAP +0.483, LIME −0.011) and Insulin (SHAP −0.184, LIME +0.033) conflict; LIME's Glucose weight is negative under all 10 seeds; R² 0.09 |

All the specific claims in the README and the experiment report about these five patients were checked programmatically and hold.

---

## 6. Experimental findings (verified values)

### 6.1 Model performance (held-out n = 154, threshold 0.5, uncalibrated)

| Metric | XGBoost | Logistic regression |
|---|---|---|
| Accuracy | 0.766 | 0.708 |
| Precision | 0.688 | 0.600 |
| Recall | 0.611 | 0.500 |
| Specificity | 0.850 | 0.820 |
| F1 | 0.647 | 0.545 |
| ROC-AUC (95% CI) | 0.822 (0.746–0.883) | 0.813 (0.740–0.873) |
| Average precision | 0.681 | 0.673 |
| Brier score | 0.163 | 0.175 |
| Confusion matrix (TN FP / FN TP) | 85 15 / 21 33 | 82 18 / 27 27 |

- **Paired ΔAUC (XGB − LR):** +0.009 (95% CI −0.018 to +0.037), so the two models are not distinguishable on ranking.
- **Training-split CV AUC:** 0.841 ± 0.020 vs 0.843 ± 0.019.

### 6.2 LIME vs SHAP (154 patients, LIME 5,000 samples, stability over 10 seeds)
- **Agreement:** same top feature for 73.4% of patients; mean top-3 overlap 0.768 (Jaccard 0.666); median rank ρ 0.821.
- **Conflicts:** 26.0% of patients have at least one material direction conflict, involving Glucose in 16 patients and Insulin in 14.
- **LIME's fit:** median LIME R² 0.232.
- **Seed stability:** top-3 Jaccard 0.879; material sign flips across seeds in 11.0% of patients.
- **Bonferroni-significant relationships:**
  - confidence vs R²: ρ = +0.327
  - confidence vs rank ρ: +0.255
  - confidence vs top-3 Jaccard: +0.279
  - R² vs rank ρ: +0.365
- **Not significant:** confidence vs seed stability, ρ = +0.002.
- **By band:** borderline predictions agree least (top-3 Jaccard 0.564, 0.59 conflicts per patient).

### 6.3 Robustness of the direction conflicts (new in this validation)

The **signs** of the conflicts are robust. Their **materiality** is borderline. The conflicting LIME weights are small: the median |w| is 0.017, and 31 of the 44 conflicts fall below 0.02, against the 0.01 tolerance.

| Patient | Feature | SHAP | LIME opposes SHAP (10 seeds) | Material (\|w\| ≥ 0.01) |
|---|---|---|---|---|
| PIMA-0078 | Glucose | +0.250 | 10 / 10 | 7 / 10 |
| PIMA-0351 | Glucose | +0.421 | 9 / 10 | 8 / 10 |
| PIMA-0486 | Glucose | +0.483 | 10 / 10 | 8 / 10 |
| PIMA-0486 | Insulin | −0.184 | 9 / 10 | 5 / 10 |

**Consequence for the live demo.** The dashboard computes a fresh LIME draw, which is reproducible but not the study's draw. For PIMA-0078, PIMA-0351 and PIMA-0486, that fresh draw puts the Glucose weight just under the tolerance, so the live comparison page shows **0** material conflicts, while the study and the case explorer show 1–2. Top-3 overlap, Jaccard and R² stay close to the study's values; for example, case 4 has Jaccard 0 in both draws, with R² 0.030 live vs 0.029 in the study. Full output: [outputs/validation/live_vs_study_lime.txt](outputs/validation/live_vs_study_lime.txt).

The 26% headline should therefore be presented as **tolerance-dependent**, and Case 5 should be demonstrated from the case explorer. The README's demo plan has been updated accordingly.

---

## 7. Report evidence inventory

| Evidence | Location |
|---|---|
| Architecture diagram | `outputs/figures/architecture.png` (+ Mermaid version in `docs/technical_documentation.md`) |
| Dataset overview and missing-value chart | `outputs/figures/dataset_overview.png`, `data/README.md`, `outputs/metrics/dataset_summary.json` |
| Model metrics and confusion matrix | `outputs/metrics/model_metrics.json`, `outputs/metrics/model_comparison.csv`, `outputs/figures/confusion_matrix.png`, `confusion_matrix_baseline.png`, `model_comparison.png` |
| ROC and PR curves | `outputs/figures/roc_curve.png`, `outputs/figures/pr_curve.png` |
| LIME local explanations | `outputs/figures/lime_patient_explanation.png`, `lime_neighborhood_example.png`, `outputs/case_studies/case*_lime.png` |
| SHAP global summary | `outputs/figures/shap_global_importance.png`, `shap_beeswarm.png`, `outputs/metrics/shap_global_importance.csv` |
| SHAP waterfall and force plots | `outputs/figures/shap_waterfall.png`, `shap_force.png`, `outputs/case_studies/case*_shap_{waterfall,force}.png` |
| SHAP additivity proof | `outputs/metrics/shap_additivity.json` |
| LIME-vs-SHAP comparison | `outputs/figures/lime_shap_side_by_side.png`, `agreement_distribution.png`, `confidence_relationships.png`, `band_summary.png`, `outputs/experiments/*.csv` |
| Explanation stability analysis | `outputs/figures/lime_stability_example.png`, `stability_distribution.png`, `outputs/case_studies/case*_stability.png`, `outputs/experiments/stability_weights.csv` |
| Five case-study tables | `outputs/case_studies/case_studies.md` (one feature table per case), `case_studies_summary.csv`, `case_studies.json` |
| Real dashboard screenshots | `outputs/screenshots/` (18 PNGs) |
| Validation records | `outputs/test_results.txt`, `outputs/validation/results_check.json`, `outputs/validation/browser_check.json`, `outputs/validation/live_vs_study_lime.txt` |
| Generated findings | `outputs/findings.md`, `docs/experiment_report.md` |

---

## 8. Remaining limitations

- **Data:** one historical cohort (Pima women aged 21+, one community). The label is partly defined by glucose. 48.7% of Insulin and 29.6% of SkinThickness values are imputed, so attributions to those features often describe a training median.
- **Statistics:** a single 154-patient test split, so the confidence intervals are wide. The high-confidence-positive band has only 12 patients. XGBoost is not shown to be better than logistic regression.
- **Calibration:** probabilities are uncalibrated. Recall of 0.61 means many positive patients are missed.
- **Explanations:**
  - LIME results depend on its settings (quartile discretiser, kernel width, 5,000 samples) and on the random draw.
  - Direction-conflict counts are sensitive to the 0.01 tolerance (section 6.3).
  - The dashboard's live LIME is a different draw from the study's.
- **Browser coverage:** Edge/Chromium at desktop width only (section 3).
- **Modality:** tabular data only. The "Multi-Modal" course title is explicitly disclaimed in the app and the docs.

---

## 9. Files changed in this validation pass

- **Code:** [src/experiment.py](src/experiment.py), [evaluate.py](evaluate.py), [app.py](app.py), [src/plotting.py](src/plotting.py).
- **Tests:** [tests/test_comparison_and_cases.py](tests/test_comparison_and_cases.py), with one new test.
- **Docs:** [README.md](README.md), [docs/experiment_report.md](docs/experiment_report.md), [docs/technical_documentation.md](docs/technical_documentation.md).
- **New:** [validation/check_results.py](validation/check_results.py), [validation/browser_check.py](validation/browser_check.py), [validation/concurrency_check.py](validation/concurrency_check.py), `outputs/screenshots/`, `outputs/validation/` and this report.
- **Deployment config:** [.streamlit/config.toml](.streamlit/config.toml) (viewer toolbar, no tracebacks), [.gitignore](.gitignore) (excludes `secrets.toml`).

The model files in `models/` and the data split were not modified.

## 10. Readiness verdict

**Ready for the academic report.**

Every check that was run passed: 92/92 tests, 89/89 independent result checks, 29/29 browser checks and 24/24 concurrent sessions. The issues found were fixed without changing any study outcome. The report should carry two caveats:
1. Describe the 26% direction-conflict rate as sensitive to the tolerance.
2. Quote case-study LIME values from the case-study files (the study's draw), not from the live dashboard.

---

## 11. Public-link readiness (deployment hardening)

These checks were added after the question of sharing a public link. Until then the dashboard had only been tested with one visitor at a time.

| Risk for a public link | Finding | Fix | Verification |
|---|---|---|---|
| Memory growth | The LIME caches had no size limit. Every new patient, seed, sample-size or what-if combination added an entry (about 0.7 MB at 10,000 samples) and none were evicted. Free hosting tiers have roughly 1 GB of memory. | `max_entries=48` on both LIME caches ([app.py](app.py)) | Tests and browser checks pass |
| Charts mixing between visitors | Streamlit runs each visitor in its own thread, and SHAP's plots draw on pyplot's process-global "current figure". A first stress test (4 simultaneous sessions × 3 rounds) produced 2 `IndexError` crashes in the beeswarm. Instrumenting showed that even with a shared lock, a figure left open by an interrupted rerun could become "current" and break the next session's plot. | `plotting.exclusive_figure`: one process-wide lock around building and rendering each figure; it closes leftover figures and always closes its own. The beeswarm now gets an explicit axes. | New pytest renders SHAP waterfalls for 4 patients from 4 threads × 4 rounds and requires byte-identical images. **Without the lock** the same harness produced wrong images (3 of 15 in one trial) and exceptions; **with the lock**, 0 of 16 wrong in every trial. Browser: 24/24 concurrent sessions each showed their own patient with charts loaded, and the server logged 0 app exceptions. The saved beeswarm figure is byte-identical after the change. |
| Developer UI exposed | Visitors would see the "Deploy" button, the developer menu and full Python tracebacks | `client.toolbarMode = "viewer"`, `client.showErrorDetails = "none"` (validated with `streamlit config show`) | Config validated |
| Linux hosting | The project was built on Windows | None needed | All 10 pinned runtime packages resolve for Linux, Python 3.12: 9 as binary wheels, plus LIME's pure-Python source package |
| Secrets and personal data | None found (no keys, passwords, emails or local paths in code or docs); `.streamlit/secrets.toml` is git-ignored | None needed | Searched across the repository |

**Test-harness notes (no app defect found).**
- The first browser concurrency script compared chart bytes downloaded from Streamlit's media URLs. Those URLs expire when a session reruns, so the comparison was unreliable and was replaced by the deterministic thread test above.
- Under load, a rerun that finishes after a tab click re-mounts the tabs on the first one; the script now retries the click. This is standard Streamlit behaviour. A visitor would see the first tab and could click again.
- One single-user browser run showed one broken image on the Comparison page (1 of 3 runs after this change). It did not recur in two further full runs, which is consistent with a media request racing a rerun.

**Still out of scope:** load beyond 6 simultaneous sessions, and the hosting platform's own memory and CPU limits, which can only be measured after deployment.
