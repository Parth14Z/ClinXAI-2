# Experimental study: how consistent are LIME and SHAP on XGBoost diabetes predictions?

All numbers come from `python evaluate.py` on the 154-patient held-out split. The machine-generated versions are in `outputs/findings.md`, `outputs/metrics/` and `outputs/experiments/`.

## 1. Research question

> How consistently do LIME and SHAP explain XGBoost diabetes predictions across different patient profiles and prediction-confidence levels?

## 2. Protocol

1. Every test patient (n = 154) is explained by
   - **TreeSHAP**: exact Shapley values in log-odds for the positive class;
   - **LIME**: `LimeTabularExplainer` with quartile discretisation, 5,000 perturbations, seed 42, explaining P(positive) through the same fitted pipeline.
2. LIME is rerun with **10 fixed seeds (0–9)** for every patient to measure stability, giving 1,694 LIME explanations in total.
3. For each patient, the following are recorded:

| Metric | Definition |
|---|---|
| top-k overlap | \|top_k(LIME) ∩ top_k(SHAP)\| / k for k = 3 and 5, ranked by \|attribution\| |
| Jaccard (top-k) | \|∩\| / \|∪\| of the two top-k sets |
| rank Spearman ρ | Spearman correlation of \|LIME\| and \|SHAP\| over all 8 features |
| direction disagreement | a feature with opposite signs where \|LIME\| ≥ 0.01 **and** \|SHAP\| ≥ 0.05 log-odds |
| LIME R² | LIME's weighted R² of the surrogate on its own neighbourhood (`exp.score`) |
| fidelity gap | \|surrogate prediction − XGBoost P\| at the patient |
| seed stability | mean pairwise top-3 Jaccard and Spearman ρ across the 10 seeds; *material* sign flip = some seed gives ≥ +0.01 and another ≤ −0.01 for the same feature |
| confidence | max(p, 1 − p); bands: high-confidence positive p ≥ 0.85, high-confidence negative p ≤ 0.15, borderline 0.40–0.60, otherwise moderate |

4. Eight relationships are tested with Spearman correlation. Both the nominal α = 0.05 and a Bonferroni threshold (0.05 / 8 = 0.00625) are reported.
5. Five case studies are selected by fixed rules, with no patient reused (see `src/case_studies.py`).

Magnitudes are never compared across the two methods, because their units differ. Only ranks, sets and signs are compared.

## 3. Model results (held-out, threshold 0.5)

| Metric | XGBoost | Logistic regression |
|---|---|---|
| Accuracy | 0.766 | 0.708 |
| Precision | 0.688 | 0.600 |
| Recall (sensitivity) | 0.611 | 0.500 |
| Specificity | 0.850 | 0.820 |
| F1 | 0.647 | 0.545 |
| ROC-AUC | 0.822 (95% CI 0.746–0.883) | 0.813 (95% CI 0.740–0.873) |
| Average precision | 0.681 | 0.673 |
| Brier score | 0.163 | 0.175 |

- Confusion matrix (XGBoost): TN 85, FP 15, FN 21, TP 33.
- The paired bootstrap AUC difference (XGB − LR) is **+0.009 (95% CI −0.018 to +0.037)**. The ranking ability of the two models is not distinguishable on this test set.
- On training-split 5-fold CV, AUC is 0.841 ± 0.020 for XGBoost and 0.843 ± 0.019 for logistic regression.
- XGBoost's advantage in accuracy and F1 at the 0.5 threshold is real for this split, but it is small next to the sampling uncertainty with n = 154.
- Recall of 0.61 means about 4 in 10 positive patients in the test set are classified negative. This is one reason the prototype must not be used to screen anyone.
- Selected hyper-parameters (CV on training data only): max_depth 2, learning_rate 0.03, n_estimators 250, min_child_weight 1. These are shallow trees, consistent with a mostly additive signal.

## 4. Explanation-consistency results

### 4.1 Overall agreement (n = 154)

| Metric | Value |
|---|---|
| Same top-1 feature | 73.4% of patients |
| Mean top-3 overlap / Jaccard | 0.768 / 0.666 |
| Mean top-5 overlap / Jaccard | 0.827 / 0.725 |
| Rank Spearman ρ, median (mean) | 0.821 (0.745) |
| Patients with ≥ 1 material direction disagreement | 26.0% |
| Mean sign agreement over all 8 features | 0.889 |
| LIME surrogate R², median (mean) | 0.232 (0.294) |
| Mean LIME fidelity gap | 0.086 |
| LIME seed stability: top-3 Jaccard / rank ρ | 0.879 / 0.899 |
| Patients with a material LIME sign flip across seeds | 11.0% (83.1% if near-zero weights are counted) |

Global TreeSHAP importance (mean \|SHAP\|, log-odds): Glucose 0.947 > BMI 0.578 > Age 0.338 > DiabetesPedigreeFunction 0.252 > Insulin 0.178 > Pregnancies 0.155 > SkinThickness 0.059 > BloodPressure 0.046.

### 4.2 Relationships

| Relationship | ρ | p | Bonferroni-significant |
|---|---|---|---|
| Confidence vs LIME R² | +0.327 | 3.6e-05 | yes |
| Confidence vs LIME fidelity gap | −0.115 | 0.154 | no |
| Confidence vs LIME–SHAP rank ρ | +0.255 | 0.0014 | yes |
| Confidence vs LIME–SHAP top-3 Jaccard | +0.279 | 0.00045 | yes |
| Confidence vs LIME seed stability | +0.002 | 0.983 | no |
| LIME R² vs LIME–SHAP rank ρ | +0.365 | 3.2e-06 | yes |
| LIME R² vs LIME seed stability | +0.137 | 0.091 | no |
| LIME seed stability vs LIME–SHAP rank ρ | +0.073 | 0.366 | no |

### 4.3 By confidence band

| Band | n | LIME R² | top-3 Jaccard | rank ρ | mean # sign conflicts | seed stability |
|---|---|---|---|---|---|---|
| High-confidence negative | 61 | 0.275 | 0.733 | 0.793 | 0.15 | 0.880 |
| Moderate | 59 | 0.286 | 0.619 | 0.703 | 0.36 | 0.889 |
| Borderline | 22 | 0.210 | 0.564 | 0.677 | 0.59 | 0.847 |
| High-confidence positive | 12 | 0.581 | 0.750 | 0.827 | 0.08 | 0.884 |

## 5. Interpretation (limited to what the data support)

1. **Agreement is generally good but not guaranteed.** In about three of four patients both methods name the same most important feature, and the median rank correlation is high (0.82). Even so, one in four patients has at least one feature that the two methods push in opposite directions.
2. **Agreement rises with confidence, and is weakest near the decision boundary.** The correlations are weak to moderate (ρ ≈ 0.26–0.33) but survive Bonferroni correction. Borderline patients have the lowest mean Jaccard (0.56) and the most sign conflicts (0.59 per patient). The high-confidence-positive band contains only 12 patients, so its means are imprecise.
3. **LIME's own fidelity predicts agreement with SHAP.** When LIME's linear surrogate fits its neighbourhood poorly (low R²), its ranking agrees less with SHAP (ρ = +0.37). A low R² is therefore a practical warning sign that a LIME explanation may not be trustworthy. Median R² is only 0.23. A straight line over quartile indicators does not capture much of the trees' local behaviour for most patients.
4. **Disagreement is mostly systematic, not sampling noise.** With 5,000 samples LIME is fairly stable across seeds (top-3 Jaccard 0.88; material sign flips in 11% of patients). Stability is not related to confidence (ρ ≈ 0) or to LIME–SHAP agreement (ρ = +0.07, n.s.). So most LIME–SHAP differences come from how the methods are built, not from LIME's randomness. Case 5 shows this directly: LIME's negative Glucose weight keeps its sign under all 10 seeds, while SHAP's Glucose contribution is +0.48.
   *Robustness caveat.* The conflicting LIME weights are small. Of the 44 material conflicts in the study, 31 have |LIME weight| < 0.02 (median 0.017), just above the 0.01 tolerance. The **sign** of these conflicts is stable: LIME's Glucose weight opposes SHAP in 10/10, 9/10 and 10/10 seeds for cases 3, 4 and 5. Whether a conflict clears the tolerance varies between draws, though: in 7/10, 8/10 and 8/10 seeds respectively, and in only 5/10 for case 5's Insulin conflict. The 26% share of patients with a direction disagreement is therefore sensitive to the tolerance setting and to the LIME draw.
5. **Glucose and Insulin cause most sign conflicts** (in 16 and 14 patients respectively). Two mechanisms explain this:
   - *Discretisation.* LIME's Glucose quartile edges are 99, 117 and 140. Cases 3, 4 and 5 all have glucose between 131 and 139, inside the `117 < Glucose <= 140` bin. LIME gives that bin a small negative weight (−0.011 to −0.022), because it is compared with a neighbourhood that includes the high `> 140` bin. TreeSHAP compares the exact value with the model's average output and gives it a positive contribution (for example +0.42 and +0.48 log-odds in cases 4 and 5).
   - *Imputation artefact.* 48.7% of Insulin values are zero placeholders replaced by the training median (125). So much mass sits on one value that LIME's Insulin quartile edges collapse to 120, 125 and 130. LIME then treats "having the median insulin" as a distinct, informative state.
6. **Glucose dominates, as expected.** The label is defined partly by a 2-hour glucose test, so Glucose dominating SHAP importance reflects how the label was built. It is not a new clinical finding.

Correctly and incorrectly classified patients show similar agreement (mean rank ρ 0.76 vs 0.68, top-3 Jaccard 0.68 vs 0.61; 118 vs 36 patients). No significance test was run for this split, so no claim is made about it.

## 6. Case studies

| Case | Patient | P(positive) | Truth | Key evidence |
|---|---|---|---|---|
| 1 High-confidence positive | PIMA-0245 | 0.934 | positive | Full top-3 agreement (Jaccard 1.0, ρ 0.93), Glucose dominates in both methods, R² 0.59, identical top-3 under all 10 seeds |
| 2 High-confidence negative | PIMA-0032 | 0.011 | negative | Glucose ≤ 99 is the strongest factor in both methods, no sign conflicts. The fidelity gap is 0.21: the linear surrogate predicts −0.20, an impossible probability, because the model is saturated near 0 |
| 3 Borderline | PIMA-0078 | 0.508 | positive | BMI > 36.5 leads in both methods. Glucose sign conflict. LIME R² 0.07. 3 imputed values |
| 4 Ranking disagreement | PIMA-0351 | 0.455 | negative | No shared top-3 features (Jaccard 0, ρ −0.24). LIME ranks the imputed-median Insulin bin first. SHAP ranks Glucose first (+0.42), which LIME pushes slightly negative. R² 0.03 |
| 5 Directional disagreement | PIMA-0486 | 0.459 | negative | Opposite signs for Glucose and Insulin. LIME's Glucose sign is stable across 10 seeds, so the conflict is systematic, although the weight is small (−0.003 to −0.029; above the 0.01 tolerance in 8/10 seeds). The Insulin conflict is material in only 5/10 seeds. R² 0.09 |

Every case type was available, so none had to be reported as unavailable.

## 7. Threats to validity

- There is a single 154-patient test split. All correlations carry sampling uncertainty, and the high-confidence-positive band has only 12 patients.
- LIME results depend on its settings (quartile discretiser, kernel width √8 × 0.75, 5,000 samples). Other settings would change the numbers.
- The sign-tolerance thresholds (0.01 and 0.05) are judgement calls. They were fixed before the study was run.
- TreeSHAP uses tree-path-dependent expectations, not an interventional background. That is a different baseline from LIME's.
- No method is treated as ground truth. Disagreement shows that the methods answer different questions; it does not show which one is "correct".
