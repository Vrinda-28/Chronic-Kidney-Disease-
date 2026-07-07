# Clinician Explanation Report

**Model:** CatBoost  
**Task:** CKD Prediction (UCI Dataset)  
**Scope:** Patient-level SHAP explanations for selected test cases  

> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.
> All predictions must be reviewed by a qualified clinician.

---

## Top Features (Global SHAP Importance)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------| 
| 1 | `anemia_risk_score` | 1.4438 |
| 2 | `specific_gravity` | 1.2315 |
| 3 | `cardiovascular_burden_score` | 1.0473 |
| 4 | `hemoglobin_creatinine_ratio` | 0.9842 |
| 5 | `hemoglobin` | 0.5303 |
| 6 | `albumin` | 0.3767 |
| 7 | `albumin_specific_gravity_interaction` | 0.3472 |
| 8 | `urea_creatinine_product` | 0.2879 |
| 9 | `age_creatinine_interaction` | 0.2770 |
| 10 | `packed_cell_volume` | 0.2701 |

---

## Patient-Level Explanations

### ✓ Correctly Predicted CKD  (Patient #78)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **1.22** — this strongly **increased** the CKD probability (SHAP = +1.941).
2. `cardiovascular_burden_score` = **1** — this strongly **increased** the CKD probability (SHAP = +1.131).
3. `hemoglobin_creatinine_ratio` = **3.35** — this strongly **increased** the CKD probability (SHAP = +1.051).

---

### ✓ Correctly Predicted CKD  (Patient #77)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.771** — this strongly **increased** the CKD probability (SHAP = +1.431).
2. `specific_gravity` = **1.01** — this strongly **increased** the CKD probability (SHAP = +1.289).
3. `cardiovascular_burden_score` = **2** — this strongly **increased** the CKD probability (SHAP = +1.167).

---

### ✓ Correctly Predicted notCKD  (Patient #79)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.12** — this strongly **decreased** the CKD probability (SHAP = -1.833).
2. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -0.967).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -0.782).

---

### ✓ Correctly Predicted notCKD  (Patient #76)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `hemoglobin_creatinine_ratio` = **22.7** — this strongly **decreased** the CKD probability (SHAP = -1.154).
2. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -0.852).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -0.742).

---

## Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| Positive SHAP | Pushes prediction *towards* CKD |
| Negative SHAP | Pushes prediction *away from* CKD |
| Large \|SHAP\| | Feature was influential for this patient |