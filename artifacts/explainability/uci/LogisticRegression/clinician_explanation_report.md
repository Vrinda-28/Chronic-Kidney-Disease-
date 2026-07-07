# Clinician Explanation Report

**Model:** LogisticRegression  
**Task:** CKD Prediction (UCI Dataset)  
**Scope:** Patient-level SHAP explanations for selected test cases  

> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.
> All predictions must be reviewed by a qualified clinician.

---

## Top Features (Global SHAP Importance)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------| 
| 1 | `white_blood_cell_count` | 59.8161 |
| 2 | `blood_glucose_random` | 40.2662 |
| 3 | `urea_creatinine_product` | 25.2880 |
| 4 | `hemoglobin_creatinine_ratio` | 16.4377 |
| 5 | `blood_urea` | 15.7815 |
| 6 | `potassium` | 11.9582 |
| 7 | `albumin_specific_gravity_interaction` | 10.8341 |
| 8 | `albumin` | 10.7582 |
| 9 | `specific_gravity` | 10.3619 |
| 10 | `hemoglobin` | 9.6037 |

---

## Patient-Level Explanations

### ✓ Correctly Predicted CKD  (Patient #78)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `white_blood_cell_count` = **1.28e+04** — this strongly **increased** the CKD probability (SHAP = +210.230).
2. `urea_creatinine_product` = **156** — this strongly **decreased** the CKD probability (SHAP = -46.355).
3. `blood_urea` = **60** — this strongly **decreased** the CKD probability (SHAP = -28.726).

---

### ✓ Correctly Predicted CKD  (Patient #16)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `white_blood_cell_count` = **8e+03** — this strongly **decreased** the CKD probability (SHAP = -37.046).
2. `blood_glucose_random` = **123** — this strongly **decreased** the CKD probability (SHAP = -22.237).
3. `diabetes_mellitus` = **1** — this strongly **increased** the CKD probability (SHAP = +19.593).

---

### ⚠ Over-predicted CKD (False Positive)  (Patient #79)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `blood_glucose_random` = **91** — this strongly **decreased** the CKD probability (SHAP = -20.512).
2. `white_blood_cell_count` = **8.6e+03** — this strongly **increased** the CKD probability (SHAP = +19.858).
3. `packed_cell_volume` = **48** — this strongly **decreased** the CKD probability (SHAP = -14.936).

> ⚠️ **Clinical Note:** This patient was **over-flagged**.
> The model predicted CKD (100.0%) but the true diagnosis is notCKD.

---

### ⚠ Over-predicted CKD (False Positive)  (Patient #76)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `white_blood_cell_count` = **5.8e+03** — this strongly **decreased** the CKD probability (SHAP = -106.348).
2. `blood_glucose_random` = **86** — this strongly **decreased** the CKD probability (SHAP = -22.247).
3. `packed_cell_volume` = **51** — this strongly **decreased** the CKD probability (SHAP = -20.335).

> ⚠️ **Clinical Note:** This patient was **over-flagged**.
> The model predicted CKD (100.0%) but the true diagnosis is notCKD.

---

## Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| Positive SHAP | Pushes prediction *towards* CKD |
| Negative SHAP | Pushes prediction *away from* CKD |
| Large \|SHAP\| | Feature was influential for this patient |