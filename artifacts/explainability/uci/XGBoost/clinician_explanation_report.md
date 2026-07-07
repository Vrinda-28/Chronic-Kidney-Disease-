# Clinician Explanation Report

**Model:** XGBoost  
**Task:** CKD Prediction (UCI Dataset)  
**Scope:** Patient-level SHAP explanations for selected test cases  

> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.
> All predictions must be reviewed by a qualified clinician.

---

## Top Features (Global SHAP Importance)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------| 
| 1 | `anemia_risk_score` | 1.7185 |
| 2 | `cardiovascular_burden_score` | 0.8386 |
| 3 | `specific_gravity` | 0.7423 |
| 4 | `hemoglobin_creatinine_ratio` | 0.5713 |
| 5 | `age_creatinine_interaction` | 0.4975 |
| 6 | `albumin` | 0.3944 |
| 7 | `hemoglobin` | 0.3536 |
| 8 | `blood_glucose_random` | 0.2768 |
| 9 | `bun_creatinine_ratio` | 0.2535 |
| 10 | `serum_creatinine` | 0.2065 |

---

## Patient-Level Explanations

### ✓ Correctly Predicted CKD  (Patient #78)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **1.22** — this strongly **increased** the CKD probability (SHAP = +1.764).
2. `cardiovascular_burden_score` = **1** — this strongly **increased** the CKD probability (SHAP = +1.138).
3. `hemoglobin_creatinine_ratio` = **3.35** — this strongly **increased** the CKD probability (SHAP = +0.717).

---

### ✓ Correctly Predicted CKD  (Patient #36)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.304** — this strongly **increased** the CKD probability (SHAP = +1.591).
2. `cardiovascular_burden_score` = **2** — this strongly **increased** the CKD probability (SHAP = +1.034).
3. `specific_gravity` = **1.01** — this strongly **increased** the CKD probability (SHAP = +0.901).

---

### ✓ Correctly Predicted notCKD  (Patient #79)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.12** — this strongly **decreased** the CKD probability (SHAP = -1.797).
2. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -0.647).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -0.556).

---

### ✓ Correctly Predicted notCKD  (Patient #45)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | notCKD |
| CKD Probability | 0.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.247** — this strongly **increased** the CKD probability (SHAP = +0.919).
2. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -0.576).
3. `cardiovascular_burden_score` = **0** — this strongly **decreased** the CKD probability (SHAP = -0.561).

---

### ⚠ Missed CKD (False Negative)  (Patient #12)

| Field | Value |
|-------|-------|
| True Diagnosis | CKD |
| Model Prediction | notCKD |
| CKD Probability | 7.5% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.113** — this strongly **decreased** the CKD probability (SHAP = -1.841).
2. `specific_gravity` = **1.01** — this strongly **increased** the CKD probability (SHAP = +1.617).
3. `hemoglobin_creatinine_ratio` = **6.62** — this strongly **increased** the CKD probability (SHAP = +0.925).

> ⚠️ **Clinical Note:** This CKD patient was **missed**.
> Despite having CKD, the model assigned only 7.5% probability.

---

### ⚠ Over-predicted CKD (False Positive)  (Patient #27)

| Field | Value |
|-------|-------|
| True Diagnosis | notCKD |
| Model Prediction | CKD |
| CKD Probability | 100.0% |

**Key drivers for this prediction:**

1. `anemia_risk_score` = **0.308** — this strongly **increased** the CKD probability (SHAP = +2.187).
2. `hemoglobin_creatinine_ratio` = **18.1** — this strongly **decreased** the CKD probability (SHAP = -0.576).
3. `specific_gravity` = **1.02** — this strongly **decreased** the CKD probability (SHAP = -0.527).

> ⚠️ **Clinical Note:** This patient was **over-flagged**.
> The model predicted CKD (100.0%) but the true diagnosis is notCKD.

---

## Interpretation Guide

| Symbol | Meaning |
|--------|---------|
| Positive SHAP | Pushes prediction *towards* CKD |
| Negative SHAP | Pushes prediction *away from* CKD |
| Large \|SHAP\| | Feature was influential for this patient |