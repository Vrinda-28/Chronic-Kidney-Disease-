# Clinician Explanation Report

**Model:** RandomForest  
**Task:** CKD Prediction (UCI Dataset)  
**Scope:** Patient-level SHAP explanations for selected test cases  

> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.
> All predictions must be reviewed by a qualified clinician.

---

## Top Features (Global SHAP Importance)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------| 
| 1 | `anemia_risk_score` | 0.0997 |
| 2 | `hemoglobin` | 0.0545 |
| 3 | `hemoglobin_creatinine_ratio` | 0.0520 |
| 4 | `specific_gravity` | 0.0471 |
| 5 | `packed_cell_volume` | 0.0423 |
| 6 | `cardiovascular_burden_score` | 0.0422 |
| 7 | `age_creatinine_interaction` | 0.0270 |
| 8 | `serum_creatinine` | 0.0256 |
| 9 | `albumin` | 0.0238 |
| 10 | `red_blood_cell_count` | 0.0211 |

---

## Patient-Level Explanations

### ✓ Correctly Predicted CKD  (Patient #78)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **1.22** — this moderately **increased** the CKD probability (SHAP = +0.104).
2. `hemoglobin` = **8.7** — this slightly **increased** the CKD probability (SHAP = +0.057).
3. `hemoglobin_creatinine_ratio` = **3.35** — this slightly **increased** the CKD probability (SHAP = +0.050).

---

### ✓ Correctly Predicted CKD  (Patient #77)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.771** — this slightly **increased** the CKD probability (SHAP = +0.077).
2. `specific_gravity` = **1.01** — this slightly **increased** the CKD probability (SHAP = +0.054).
3. `cardiovascular_burden_score` = **2** — this marginally **increased** the CKD probability (SHAP = +0.046).

---

### ✓ Correctly Predicted notCKD  (Patient #79)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.12** — this moderately **decreased** the CKD probability (SHAP = -0.118).
2. `packed_cell_volume` = **48** — this slightly **decreased** the CKD probability (SHAP = -0.060).
3. `hemoglobin` = **13.5** — this slightly **decreased** the CKD probability (SHAP = -0.057).

---

### ✓ Correctly Predicted notCKD  (Patient #45)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.247** — this slightly **increased** the CKD probability (SHAP = +0.083).
2. `hemoglobin_creatinine_ratio` = **27.6** — this slightly **decreased** the CKD probability (SHAP = -0.062).
3. `hemoglobin` = **13.8** — this slightly **decreased** the CKD probability (SHAP = -0.055).

---

### ⚠ Over-predicted CKD (False Positive)  (Patient #27)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.308** — this moderately **increased** the CKD probability (SHAP = +0.182).
2. `hemoglobin` = **12.7** — this slightly **increased** the CKD probability (SHAP = +0.085).
3. `packed_cell_volume` = **40** — this slightly **increased** the CKD probability (SHAP = +0.062).

> ⚠️ **Clinical Note:** This patient was **over-flagged**.
> The model predicted CKD (100.0%) but the true diagnosis is notCKD.

---

## Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| Positive SHAP | Pushes prediction *towards* CKD |
| Negative SHAP | Pushes prediction *away from* CKD |
| Large \|SHAP\| | Feature was influential for this patient |