# ClinXAI - generated findings

All numbers below are computed by `evaluate.py` from the saved model and the untouched 154-patient held-out split.

## Model performance (threshold 0.5, uncalibrated)

| Metric | XGBoost | Logistic regression |
|---|---|---|
| accuracy | 0.766 | 0.708 |
| precision | 0.688 | 0.600 |
| recall | 0.611 | 0.500 |
| specificity | 0.850 | 0.820 |
| f1 | 0.647 | 0.545 |
| roc_auc | 0.822 | 0.813 |
| average_precision | 0.681 | 0.673 |
| brier_score | 0.163 | 0.175 |

XGBoost test ROC-AUC 95% bootstrap CI: 0.746-0.883. Paired bootstrap AUC difference (XGB - LR): +0.009 (95% CI -0.018 to +0.037).
Training-split 5-fold CV ROC-AUC: XGBoost 0.841 ± 0.020; logistic regression 0.843 ± 0.019.

## LIME vs SHAP agreement across all held-out patients

- n_patients: 154
- top1_match_rate: 0.734
- mean_top3_overlap: 0.768
- mean_top5_overlap: 0.827
- mean_jaccard_top3: 0.666
- mean_jaccard_top5: 0.725
- median_rank_spearman: 0.821
- mean_rank_spearman: 0.745
- share_with_direction_disagreement: 0.260
- mean_sign_agreement: 0.889
- mean_lime_r2: 0.294
- median_lime_r2: 0.232
- mean_lime_fidelity_gap: 0.086
- mean_stability_jaccard_top3: 0.879
- mean_stability_spearman: 0.899
- mean_sign_consistency: 0.852
- share_any_sign_flip_across_seeds: 0.831
- share_material_sign_flip_across_seeds: 0.110
- band_counts: {'high-confidence negative': 61, 'moderate': 59, 'borderline': 22, 'high-confidence positive': 12}
- lime_num_samples: 5000
- lime_seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## Relationships (Spearman; Bonferroni threshold 0.0063)

| Relationship | rho | p | strength | sig. 0.05 | sig. Bonferroni |
|---|---|---|---|---|---|
| Confidence vs LIME surrogate R² | +0.327 | 3.56e-05 | moderate | yes | yes |
| Confidence vs LIME local-prediction gap | -0.115 | 0.154 | weak | no | no |
| Confidence vs LIME-SHAP rank correlation | +0.255 | 0.00139 | weak | yes | yes |
| Confidence vs LIME-SHAP top-3 Jaccard | +0.279 | 0.000449 | weak | yes | yes |
| Confidence vs LIME seed stability (top-3 Jaccard) | +0.002 | 0.983 | negligible | no | no |
| LIME R² vs LIME-SHAP rank correlation | +0.365 | 3.2e-06 | moderate | yes | yes |
| LIME R² vs LIME seed stability | +0.137 | 0.0908 | weak | no | no |
| LIME seed stability vs LIME-SHAP rank correlation | +0.073 | 0.366 | negligible | no | no |

## Mean metrics by confidence band

| band | n_patients | prob_positive | lime_r2 | jaccard_top3 | rank_spearman | n_direction_disagreements | stability_jaccard_top3 |
|---|---|---|---|---|---|---|---|
| high-confidence negative | 61 | 0.059 | 0.275 | 0.733 | 0.793 | 0.148 | 0.88 |
| moderate | 59 | 0.479 | 0.286 | 0.619 | 0.703 | 0.356 | 0.889 |
| borderline | 22 | 0.483 | 0.21 | 0.564 | 0.677 | 0.591 | 0.847 |
| high-confidence positive | 12 | 0.901 | 0.581 | 0.75 | 0.827 | 0.083 | 0.884 |

## Case studies

- Case 1 (High-confidence diabetes-positive prediction): PIMA-0245 - P(positive) = 0.934
- Case 2 (High-confidence diabetes-negative prediction): PIMA-0032 - P(positive) = 0.011
- Case 3 (Borderline prediction): PIMA-0078 - P(positive) = 0.508, |P - 0.5| = 0.008
- Case 4 (LIME-SHAP feature-ranking disagreement): PIMA-0351 - Spearman rho = -0.238, top-3 Jaccard = 0.00 (LIME top-3: Insulin;Age;Pregnancies; SHAP top-3: Glucose;DiabetesPedigreeFunction;BMI)
- Case 5 (LIME-SHAP directional disagreement): PIMA-0486 - 2 feature(s) with opposite signs: Glucose;Insulin