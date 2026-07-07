# Clinician Explanation Report

**Model:** LightGBM  
**Task:** CKD Prediction (UCI Dataset)  
**Scope:** Patient-level SHAP explanations for selected test cases  

> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.
> All predictions must be reviewed by a qualified clinician.

---

## Top Features (Global SHAP Importance)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------| 
| 1 | `anemia_risk_score` | 1.5049 |
| 2 | `cardiovascular_burden_score` | 1.4309 |
| 3 | `specific_gravity` | 1.1885 |
| 4 | `hemoglobin_creatinine_ratio` | 1.0095 |
| 5 | `albumin` | 0.6205 |
| 6 | `hemoglobin` | 0.4852 |
| 7 | `bun_creatinine_ratio` | 0.2523 |
| 8 | `blood_glucose_random` | 0.2164 |
| 9 | `red_blood_cell_count` | 0.2032 |
| 10 | `albumin_specific_gravity_interaction` | 0.1720 |

---

## Patient-Level Explanations

### ✓ Correctly Predicted CKD  (Patient #78)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `cardiovascular_burden_score` = **1** — this strongly **increased** the CKD probability (SHAP = +1.437).
2. `anemia_risk_score` = **1.22** — this strongly **increased** the CKD probability (SHAP = +1.432).
3. `hemoglobin_creatinine_ratio` = **3.35** — this strongly **increased** the CKD probability (SHAP = +1.189).

---

### ✓ Correctly Predicted CKD  (Patient #36)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `cardiovascular_burden_score` = **2** — this strongly **increased** the CKD probability (SHAP = +1.296).
2. `anemia_risk_score` = **0.304** — this strongly **increased** the CKD probability (SHAP = +1.239).
3. `specific_gravity` = **1.01** — this strongly **increased** the CKD probability (SHAP = +1.174).

---

### ✓ Correctly Predicted notCKD  (Patient #79)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.12** — this strongly **decreased** the CKD probability (SHAP = -1.799).
2. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -1.752).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -1.374).

---

### ✓ Correctly Predicted notCKD  (Patient #76)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -1.705).
2. `hemoglobin_creatinine_ratio` = **22.7** — this strongly **decreased** the CKD probability (SHAP = -1.370).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -1.344).

---

### ⚠ Missed CKD (False Negative)  (Patient #23)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.308** — this strongly **increased** the CKD probability (SHAP = +1.801).
2. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -1.295).
3. `hemoglobin_creatinine_ratio` = **12.7** — this strongly **decreased** the CKD probability (SHAP = -1.292).

> ⚠️ **Clinical Note:** This CKD patient was **missed**.
> Despite having CKD, the model assigned only 0.0% probability.

---

## Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| Positive SHAP | Pushes prediction *towards* CKD |
| Negative SHAP | Pushes prediction *away from* CKD |
| Large \|SHAP\| | Feature was influential for this patient |