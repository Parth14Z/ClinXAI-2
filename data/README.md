# Data provenance

| Item | Value |
|---|---|
| Dataset | Pima Indians Diabetes Database |
| Obtained from | OpenML dataset 37, version 1: <https://www.openml.org/d/37> |
| File | `raw/diabetes_openml37.arff` (downloaded from `https://openml.org/data/v1/download/37/diabetes.arff`) |
| Integrity | MD5 `3cbaa3e54586aa88cf6aacb4033e4470`, identical to the checksum OpenML publishes. `train_model.py` re-checks it and stops if it does not match. |
| Original owner | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Donor | Vincent Sigillito, Applied Physics Laboratory, Johns Hopkins University (received 9 May 1990) |
| Original repository | UCI Machine Learning Repository (no longer listed on the current UCI site) |
| Licence | OpenML's licence field says "Public". UCI asks users to cite the original work. No other licence terms are published. |
| Citation | Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proc. Symposium on Computer Applications and Medical Care, 261–265. IEEE Computer Society Press. |

`raw/provenance.json` holds the machine-readable record, including the SHA-256 hash and the retrieval date. `raw/openml_37_description.json` is OpenML's metadata as downloaded.

## Population

All 768 records are **women aged 21 or older of Pima Indian heritage** living near Phoenix, Arizona, USA. The records were collected decades ago. No results from this project should be applied to men, to other ancestries or age groups, or to today's clinical populations.

## Feature definitions (from the source documentation)

| Column | ARFF name | Definition | Unit |
|---|---|---|---|
| Pregnancies | preg | Number of times pregnant | count |
| Glucose | plas | Plasma glucose concentration 2 hours into an oral glucose tolerance test | mg/dL (the usual unit; the source does not state one) |
| BloodPressure | pres | Diastolic blood pressure | mm Hg |
| SkinThickness | skin | Triceps skin fold thickness | mm |
| Insulin | insu | 2-hour serum insulin | µU/mL |
| BMI | mass | Body mass index | kg/m² |
| DiabetesPedigreeFunction | pedi | Diabetes pedigree function (a score based on family history) | unitless |
| Age | age | Age | years |
| Outcome | class | `tested_positive` → 1, `tested_negative` → 0 | — |

**Label definition (source):** a record is positive if the patient "shows signs of diabetes according to World Health Organization criteria (i.e., if the 2 hour post-load plasma glucose was at least 200 mg/dl at any survey examination or if found during routine medical care)". The class counts are 500 negative and 268 positive.

## Missing values

The source documentation says there are no missing values. However, some values are zero where zero is physiologically impossible:

| Feature | Zeros | % of 768 |
|---|---|---|
| Glucose | 5 | 0.7 |
| BloodPressure | 35 | 4.6 |
| SkinThickness | 227 | 29.6 |
| Insulin | 374 | 48.7 |
| BMI | 11 | 1.4 |

These zeros are converted to missing values. They are then imputed with **medians fitted on the training split only**. A zero in `Pregnancies` is a real value and is kept.

## Derived files (`processed/`)

- `pima_diabetes_clean_names.csv`: the full dataset with descriptive column names and `patient_id` (the zero-based row number in the original file).
- `train.csv` and `test.csv`: the stratified 80/20 split (seed 42) with raw values. Zeros are kept here; the model pipeline handles them.
