# ClinXAI: Transparent Multi-Modal Clinical Diagnostics Using LIME and SHAP

**Course:** Trustworthy and Responsible AI (ADCOR1PC303) · **Domain:** Healthcare systems / societal impact
**Syllabus mapping:** Module 1: interpretability trade-offs, LIME perturbations, SHAP game theory

ClinXAI is an educational prototype. It trains an **XGBoost** classifier on the **Pima Indians Diabetes Database** and explains each prediction with **LIME** and **TreeSHAP**. It then measures how consistently the two explanation methods agree across all 154 held-out patients, at different levels of prediction confidence. A Streamlit dashboard presents the results.

> **Educational use only.** This is not a medical device. It has not been clinically validated, it does not diagnose diabetes, and it does not predict future disease onset. Its probabilities are uncalibrated model scores.

> **About "Multi-Modal".** This is the assigned course title. The project uses **one data modality: tabular clinical measurements**. It does not process images, signals or text. What it combines is two explanation paradigms (perturbation-based and game-theoretic) and several presentation modes.

## Quick start

```powershell
# Windows PowerShell, Python 3.12
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train_model.py      # downloads + verifies the data, tunes and trains the models  (~10 s)
python evaluate.py         # held-out metrics, SHAP/LIME study, case studies, figures  (~1.5 min)
python -m pytest           # 92 automated tests
streamlit run app.py       # dashboard at http://localhost:8501
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`. For a quicker trial of the study, run `python evaluate.py --num-samples 1000 --n-seeds 3`.

## Public deployment (Streamlit Community Cloud)

The dashboard runs from the committed artefacts in `models/` and `outputs/`; no training happens on the server.

1. Push this repository to GitHub (it may stay private; the app link can still be public).
2. At <https://share.streamlit.io>, choose **Create app**, then select the repository, branch `main` and main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**. `requirements.txt` is installed automatically.

For public use, `.streamlit/config.toml` hides the developer toolbar and error tracebacks. The app caps its LIME caches at 48 entries each, and it builds matplotlib figures one at a time across visitors, because SHAP draws on pyplot's global figure.

## Project structure

```
app.py                  Streamlit dashboard (7 pages)
train_model.py          data -> split -> CV-tuned XGBoost + logistic baseline -> models/
evaluate.py             test metrics, TreeSHAP, LIME study, case studies -> outputs/
requirements.txt        pinned direct dependencies (requirements-lock.txt = full environment)
src/
  config.py             paths, seeds, feature order, thresholds, palette
  data.py               OpenML download + MD5 check, ARFF parsing, validation, split
  preprocessing.py      zero->NaN, train-only median imputation, input validation
  model.py              pipelines, grid search, persistence, metadata
  metrics.py            held-out metrics, bootstrap CIs
  lime_explainer.py     LIME explanations, neighbourhood capture, seed stability
  shap_explainer.py     TreeSHAP in log-odds, additivity verification
  comparison.py         overlap, Jaccard, Spearman, direction disagreement
  experiment.py         per-patient study, relationship and band analysis
  case_studies.py       rule-based selection of 5 genuine held-out cases
  plotting.py           all figures
  ui.py                 dashboard styling
data/raw/               authentic ARFF + provenance.json      (see data/README.md)
data/processed/         train.csv / test.csv with patient ids
models/                 xgb_pipeline.joblib, logreg_baseline.joblib, model_metadata.json
outputs/figures/        evaluation, SHAP, LIME, comparison, study, architecture figures
outputs/metrics/        dataset/preprocessing summaries, model metrics, SHAP additivity
outputs/experiments/    per-patient study tables, relationships, band summary
outputs/case_studies/   five cases: figures, JSON, Markdown
outputs/findings.md     auto-generated numeric findings
tests/                  pytest suite (data -> dashboard rendering)
validation/             independent result checks and Playwright browser/concurrency checks
docs/                   technical documentation and experiment report
```

## Method in brief

| Step | Choice |
|---|---|
| Data | OpenML #37 (the original Pima file: NIDDK owner, donated to UCI), MD5-verified; 768 × 8; 500 negative and 268 positive |
| Missing values | Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI become NaN, then median imputation **fitted on training data only** |
| Split | Stratified 80/20, seed 42 (614 train / 154 test) |
| Model | `XGBClassifier`, grid searched with 5-fold CV AUC **on the training split only**; fixed threshold 0.5 |
| Baseline | Logistic regression (same imputation + standard scaling) |
| SHAP | `shap.TreeExplainer` explaining the raw margin (**log-odds**); base + ΣSHAP = margin verified to < 2e-6; P = sigmoid(margin) |
| LIME | `LimeTabularExplainer`, quartile discretisation, 5,000 samples, explaining `predict_proba[:, 1]` through the same pipeline at the same preprocessed point |
| Study | All 154 test patients: top-3/5 overlap, Jaccard, Spearman, sign conflicts, LIME R², stability over 10 seeds; Spearman relationships with Bonferroni correction |

Why XGBoost? Boosted trees capture non-linear thresholds and interactions in tabular data without scaling or feature engineering, and they support exact, fast TreeSHAP. On this dataset XGBoost is **not** reliably better than logistic regression (see below). The interpretability trade-off is part of what the project demonstrates.

## Results (held-out test set, n = 154)

| Metric | XGBoost | Logistic regression |
|---|---|---|
| Accuracy | 0.766 | 0.708 |
| Precision | 0.688 | 0.600 |
| Recall | 0.611 | 0.500 |
| Specificity | 0.850 | 0.820 |
| F1 | 0.647 | 0.545 |
| ROC-AUC | 0.822 (95% CI 0.746–0.883) | 0.813 |

The paired bootstrap AUC difference is +0.009 (95% CI −0.018 to +0.037), so the two models have statistically indistinguishable ranking ability. Training-split CV AUC: XGBoost 0.841 ± 0.020, logistic regression 0.843 ± 0.019.

**LIME vs SHAP consistency (154 patients):**
- The two methods name the same top feature for 73% of patients. Mean top-3 overlap is 0.77 and median rank ρ is 0.82.
- 26% of patients have at least one feature pushed in opposite directions. Glucose and Insulin cause most of these conflicts. The conflicting LIME weights are usually small (31 of 44 below 0.02, near the 0.01 tolerance), so this share depends on the tolerance and on the LIME draw. The conflicts' signs are stable across seeds.
- Agreement increases with prediction confidence (ρ = +0.26 to +0.28, Bonferroni-significant) and is weakest for borderline predictions (top-3 Jaccard 0.56).
- LIME's surrogate fidelity is low (median R² 0.23). Lower R² goes with lower agreement with SHAP (ρ = +0.37).
- LIME is fairly stable across seeds at 5,000 samples (top-3 Jaccard 0.88; material sign flips in 11% of patients). Stability is unrelated to confidence or to agreement, so most disagreement is systematic rather than random.
- Mechanisms: LIME's quartile discretisation (a 117–140 glucose bin weighted negatively while SHAP is positive), and median imputation collapsing the Insulin bins to 120/125/130.

Full analysis: [docs/experiment_report.md](docs/experiment_report.md). Five case studies: [outputs/case_studies/case_studies.md](outputs/case_studies/case_studies.md).

## Limitations

- **Population:** women aged 21+ of Pima heritage, from one community, collected decades ago. The results do not generalise.
- **Label:** the label is defined partly by a 2-hour glucose test, so Glucose dominating the explanations is expected, not a discovery.
- **Missing data:** 48.7% of Insulin and 29.6% of SkinThickness values are imputed. Their attributions often describe a training median, not a measurement.
- **Calibration:** probabilities are not calibrated or validated as risk estimates. Recall of 0.61 means many positive patients are missed.
- **Sample size:** one small held-out set. The confidence intervals are wide, and the high-confidence-positive band has only 12 patients.
- **Explanations:** the LIME results depend on its settings and on the random draw. The dashboard's live LIME is a fresh draw and can differ slightly from the study's draw, which the case studies use. The two methods use different baselines and units, so neither one is ground truth.
- **Modality:** tabular data only; no multi-modal fusion.

## Independence statement

This repository is an independent implementation for the assigned topic. It uses a different clinical problem (diabetes), dataset (Pima), model (XGBoost), visual design and experimental methodology (a cross-patient LIME–SHAP consistency study with rule-selected case studies) from other submissions. No other student's code, reports, figures or results were accessed or reused.

## Three-minute faculty demonstration

| Time | Page | What to show |
|---|---|---|
| 0:00–0:25 | Home | Objective, OpenML provenance with MD5, limited population, zero-placeholder chart, disclaimer, the honest note on "multi-modal" |
| 0:25–0:55 | Diabetes prediction | Select **Case 1 · PIMA-0245** (p = 0.93). Show recorded vs model-input values, with Insulin marked as imputed, and the uncalibrated-probability note. Use the what-if editor to lower Glucose from 184 to 120 and the score drops from 0.93 to 0.74 |
| 0:55–1:25 | LIME | Rule-based weights (Glucose > 140), the R² fidelity score, the 5,000-sample neighbourhood histogram, the 5-seed stability toggle |
| 1:25–1:55 | SHAP | Global beeswarm, then the waterfall and force plot. Point out the units (base −0.62 + ΣSHAP = margin 2.65 → P = 0.93) and the additivity check, versus the 3.6 error from adding log-odds to a probability |
| 1:55–2:30 | Case study explorer | Case 4 (PIMA-0351): no shared top-3 features, LIME R² 0.03. Case 5: LIME's Glucose sign opposes SHAP under all 10 seeds (systematic), but the weight is small. Show this in the case explorer: the live pages draw LIME afresh, and that draw can fall below the 0.01 materiality tolerance |
| 2:30–3:00 | Experimental evaluation | XGBoost vs logistic regression (AUC CI includes 0); agreement vs confidence scatter; conclusion: check LIME's R² and compare it with SHAP before trusting a local explanation, especially near the decision boundary |
