# ClinXAI case studies (held-out patients)

Selected automatically by the pre-declared rules in `src/case_studies.py`. Probabilities are uncalibrated XGBoost scores; SHAP values are log-odds; LIME weights are local-surrogate coefficients.

## Case 1: High-confidence diabetes-positive prediction

*Rule:* Highest P(positive) among patients with P >= 0.85

*Patient:* PIMA-0245 · *Selection:* P(positive) = 0.934

- XGBoost score P(positive) = 0.934 (uncalibrated) -> predicted diabetes-positive; recorded outcome diabetes-positive (correct).
- Largest SHAP contribution: Glucose (+1.799 log-odds, toward positive); largest LIME weight: Glucose (+0.440, rule 'Glucose > 140.00').
- Agreement: top-3 overlap 1.00, top-5 overlap 0.80, top-3 Jaccard 1.00, Spearman rank rho 0.93, sign agreement 1.00.
- No material directional disagreement between LIME and SHAP for this patient.
- LIME surrogate weighted R² = 0.591 (moderate); surrogate-vs-model gap = 0.064.
- Across 10 LIME seeds: mean top-3 Jaccard 1.00, mean rank rho 0.94; material sign flips (|w| >= 0.01 on both sides): none; any sign change incl. near-zero weights: none.
- 1 of this patient's values were zero placeholders replaced by training medians; attributions to those features describe the median, not a measurement.

| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight | LIME rank | Material sign conflict |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 9 | 9 | +0.348 | 4 | `Pregnancies > 6.00` | +0.0677 | 4 |  |
| Glucose | 184 | 184 | +1.799 | 1 | `Glucose > 140.00` | +0.4401 | 1 |  |
| BloodPressure | 85 | 85 | +0.005 | 8 | `BloodPressure > 80.00` | +0.0022 | 8 |  |
| SkinThickness | 15 | 15 | -0.044 | 7 | `SkinThickness <= 25.00` | -0.0213 | 6 |  |
| Insulin | 0 | 125 (imputed) | +0.116 | 6 | `120.00 < Insulin <= 125.00` | +0.0447 | 5 |  |
| BMI | 30 | 30 | +0.168 | 5 | `27.62 < BMI <= 32.40` | +0.0147 | 7 |  |
| DiabetesPedigreeFunction | 1.213 | 1.213 | +0.393 | 3 | `DiabetesPedigreeFunction > 0.64` | +0.0848 | 3 |  |
| Age | 49 | 49 | +0.482 | 2 | `Age > 41.00` | +0.0923 | 2 |  |

SHAP: base -0.622 + sum = margin +2.645 log-odds -> P = 0.934. LIME: R² 0.591, surrogate prediction 0.870.

## Case 2: High-confidence diabetes-negative prediction

*Rule:* Lowest P(positive) among patients with P <= 0.15

*Patient:* PIMA-0032 · *Selection:* P(positive) = 0.011

- XGBoost score P(positive) = 0.011 (uncalibrated) -> predicted diabetes-negative; recorded outcome diabetes-negative (correct).
- Largest SHAP contribution: Glucose (-1.506 log-odds, toward negative); largest LIME weight: Glucose (-0.281, rule 'Glucose <= 99.00').
- Agreement: top-3 overlap 0.67, top-5 overlap 0.80, top-3 Jaccard 0.50, Spearman rank rho 0.83, sign agreement 1.00.
- No material directional disagreement between LIME and SHAP for this patient.
- LIME surrogate weighted R² = 0.391 (moderate); surrogate-vs-model gap = 0.212.
- Across 10 LIME seeds: mean top-3 Jaccard 0.73, mean rank rho 0.91; material sign flips (|w| >= 0.01 on both sides): none; any sign change incl. near-zero weights: none.

| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight | LIME rank | Material sign conflict |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 3 | 3 | -0.083 | 8 | `1.00 < Pregnancies <= 3.00` | -0.0259 | 5 |  |
| Glucose | 88 | 88 | -1.506 | 1 | `Glucose <= 99.00` | -0.2813 | 1 |  |
| BloodPressure | 58 | 58 | -0.109 | 7 | `BloodPressure <= 64.00` | -0.0037 | 8 |  |
| SkinThickness | 11 | 11 | -0.151 | 6 | `SkinThickness <= 25.00` | -0.0139 | 7 |  |
| Insulin | 54 | 54 | -0.405 | 4 | `Insulin <= 120.00` | -0.0870 | 3 |  |
| BMI | 24.8 | 24.8 | -1.000 | 2 | `BMI <= 27.62` | -0.2050 | 2 |  |
| DiabetesPedigreeFunction | 0.267 | 0.267 | -0.219 | 5 | `0.24 < DiabetesPedigreeFunction <= 0.38` | -0.0216 | 6 |  |
| Age | 22 | 22 | -0.429 | 3 | `Age <= 24.00` | -0.0777 | 4 |  |

SHAP: base -0.622 + sum = margin -4.525 log-odds -> P = 0.011. LIME: R² 0.391, surrogate prediction -0.202.

## Case 3: Borderline prediction

*Rule:* P(positive) closest to the 0.5 threshold within [0.4, 0.6]

*Patient:* PIMA-0078 · *Selection:* P(positive) = 0.508, |P - 0.5| = 0.008

- XGBoost score P(positive) = 0.508 (uncalibrated) -> predicted diabetes-positive; recorded outcome diabetes-positive (correct).
- Largest SHAP contribution: BMI (+0.777 log-odds, toward positive); largest LIME weight: BMI (+0.135, rule 'BMI > 36.50').
- Agreement: top-3 overlap 0.67, top-5 overlap 1.00, top-3 Jaccard 0.50, Spearman rank rho 0.74, sign agreement 0.75.
- Opposite directions for Glucose. LIME's sign reflects the patient's quartile bin versus other bins in the perturbed neighbourhood; SHAP's sign reflects the value versus the tree-path expectation. A sign difference shows the two baselines differ; it does not establish which method is 'right'.
- LIME surrogate weighted R² = 0.074 (low - the linear surrogate captures little of the model's local behaviour); surrogate-vs-model gap = 0.122.
- Across 10 LIME seeds: mean top-3 Jaccard 0.90, mean rank rho 0.86; material sign flips (|w| >= 0.01 on both sides): none; any sign change incl. near-zero weights: BloodPressure;SkinThickness.
- 3 of this patient's values were zero placeholders replaced by training medians; attributions to those features describe the median, not a measurement.

| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight | LIME rank | Material sign conflict |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 0 | 0 | -0.093 | 6 | `Pregnancies <= 1.00` | -0.0077 | 8 |  |
| Glucose | 131 | 131 | +0.250 | 3 | `117.00 < Glucose <= 140.00` | -0.0220 | 5 | yes |
| BloodPressure | 0 | 72 (imputed) | +0.059 | 7 | `64.00 < BloodPressure <= 72.00` | +0.0106 | 7 |  |
| SkinThickness | 0 | 29 (imputed) | +0.005 | 8 | `25.00 < SkinThickness <= 29.00` | -0.0140 | 6 |  |
| Insulin | 0 | 125 (imputed) | +0.140 | 5 | `120.00 < Insulin <= 125.00` | +0.0600 | 2 |  |
| BMI | 43.2 | 43.2 | +0.777 | 1 | `BMI > 36.50` | +0.1352 | 1 |  |
| DiabetesPedigreeFunction | 0.27 | 0.27 | -0.194 | 4 | `0.24 < DiabetesPedigreeFunction <= 0.38` | -0.0246 | 4 |  |
| Age | 26 | 26 | -0.287 | 2 | `24.00 < Age <= 29.00` | -0.0519 | 3 |  |

SHAP: base -0.622 + sum = margin +0.033 log-odds -> P = 0.508. LIME: R² 0.074, surrogate prediction 0.387.

## Case 4: LIME-SHAP feature-ranking disagreement

*Rule:* Lowest LIME-SHAP Spearman rank correlation among patients whose top-3 sets share at most 2 features (Jaccard <= 0.5); ties broken by lower top-3 Jaccard

*Patient:* PIMA-0351 · *Selection:* Spearman rho = -0.238, top-3 Jaccard = 0.00 (LIME top-3: Insulin;Age;Pregnancies; SHAP top-3: Glucose;DiabetesPedigreeFunction;BMI)

- XGBoost score P(positive) = 0.455 (uncalibrated) -> predicted diabetes-negative; recorded outcome diabetes-negative (correct).
- Largest SHAP contribution: Glucose (+0.421 log-odds, toward positive); largest LIME weight: Insulin (+0.064, rule '120.00 < Insulin <= 125.00').
- Agreement: top-3 overlap 0.00, top-5 overlap 0.60, top-3 Jaccard 0.00, Spearman rank rho -0.24, sign agreement 0.62.
- Opposite directions for Glucose. LIME's sign reflects the patient's quartile bin versus other bins in the perturbed neighbourhood; SHAP's sign reflects the value versus the tree-path expectation. A sign difference shows the two baselines differ; it does not establish which method is 'right'.
- LIME surrogate weighted R² = 0.029 (low - the linear surrogate captures little of the model's local behaviour); surrogate-vs-model gap = 0.139.
- Across 10 LIME seeds: mean top-3 Jaccard 0.90, mean rank rho 0.72; material sign flips (|w| >= 0.01 on both sides): none; any sign change incl. near-zero weights: Pregnancies;Glucose;BloodPressure;SkinThickness.
- 2 of this patient's values were zero placeholders replaced by training medians; attributions to those features describe the median, not a measurement.

| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight | LIME rank | Material sign conflict |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 4 | 4 | -0.097 | 5 | `3.00 < Pregnancies <= 6.00` | -0.0383 | 3 |  |
| Glucose | 137 | 137 | +0.421 | 1 | `117.00 < Glucose <= 140.00` | -0.0134 | 6 | yes |
| BloodPressure | 84 | 84 | -0.009 | 7 | `BloodPressure > 80.00` | -0.0131 | 7 |  |
| SkinThickness | 0 | 29 (imputed) | +0.002 | 8 | `25.00 < SkinThickness <= 29.00` | -0.0209 | 4 |  |
| Insulin | 0 | 125 (imputed) | +0.121 | 4 | `120.00 < Insulin <= 125.00` | +0.0636 | 1 |  |
| BMI | 31.2 | 31.2 | +0.220 | 3 | `27.62 < BMI <= 32.40` | -0.0095 | 8 |  |
| DiabetesPedigreeFunction | 0.252 | 0.252 | -0.238 | 2 | `0.24 < DiabetesPedigreeFunction <= 0.38` | -0.0204 | 5 |  |
| Age | 30 | 30 | +0.021 | 6 | `29.00 < Age <= 41.00` | +0.0457 | 2 |  |

SHAP: base -0.622 + sum = margin -0.181 log-odds -> P = 0.455. LIME: R² 0.029, surrogate prediction 0.315.

## Case 5: LIME-SHAP directional disagreement

*Rule:* Most same-feature sign disagreements (both attributions above tolerance); ties broken by the SHAP magnitude of the disagreeing features

*Patient:* PIMA-0486 · *Selection:* 2 feature(s) with opposite signs: Glucose;Insulin

- XGBoost score P(positive) = 0.459 (uncalibrated) -> predicted diabetes-negative; recorded outcome diabetes-negative (correct).
- Largest SHAP contribution: Glucose (+0.483 log-odds, toward positive); largest LIME weight: BMI (+0.140, rule 'BMI > 36.50').
- Agreement: top-3 overlap 0.33, top-5 overlap 0.60, top-3 Jaccard 0.20, Spearman rank rho 0.19, sign agreement 0.62.
- Opposite directions for Glucose;Insulin. LIME's sign reflects the patient's quartile bin versus other bins in the perturbed neighbourhood; SHAP's sign reflects the value versus the tree-path expectation. A sign difference shows the two baselines differ; it does not establish which method is 'right'.
- LIME surrogate weighted R² = 0.086 (low - the linear surrogate captures little of the model's local behaviour); surrogate-vs-model gap = 0.060.
- Across 10 LIME seeds: mean top-3 Jaccard 0.58, mean rank rho 0.60; material sign flips (|w| >= 0.01 on both sides): none; any sign change incl. near-zero weights: BloodPressure;Insulin.

| Feature | Recorded | Model input | SHAP (log-odds) | SHAP rank | LIME rule | LIME weight | LIME rank | Material sign conflict |
|---|---|---|---|---|---|---|---|---|
| Pregnancies | 1 | 1 | -0.126 | 6 | `Pregnancies <= 1.00` | -0.0165 | 5 |  |
| Glucose | 139 | 139 | +0.483 | 1 | `117.00 < Glucose <= 140.00` | -0.0112 | 7 | yes |
| BloodPressure | 62 | 62 | +0.043 | 8 | `BloodPressure <= 64.00` | -0.0091 | 8 |  |
| SkinThickness | 41 | 41 | +0.126 | 7 | `SkinThickness > 32.00` | +0.0167 | 4 |  |
| Insulin | 480 | 480 | -0.184 | 5 | `Insulin > 130.00` | +0.0325 | 3 | yes |
| BMI | 40.7 | 40.7 | +0.225 | 4 | `BMI > 36.50` | +0.1396 | 1 |  |
| DiabetesPedigreeFunction | 0.536 | 0.536 | +0.284 | 3 | `0.38 < DiabetesPedigreeFunction <= 0.64` | +0.0144 | 6 |  |
| Age | 21 | 21 | -0.394 | 2 | `Age <= 24.00` | -0.0847 | 2 |  |

SHAP: base -0.622 + sum = margin -0.166 log-odds -> P = 0.459. LIME: R² 0.086, surrogate prediction 0.399.
