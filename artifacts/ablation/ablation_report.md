# Ablation Study Report

**Dataset:** UCI CKD  
**Approach:** Mask-based ablation — non-selected features replaced with training-set mean  
**Feature ranking:** SHAP importance (positive class)  

> This study evaluates whether a simpler model with fewer features
> can achieve performance comparable to the full-feature model.

---

## Results by Model

### CatBoost

| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |
|----------------|-----------|---------|-----|-----|----------|-------------|-------------|
| **All Features** | 23 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Top 20 SHAP** | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Top 15 SHAP** | 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Top 10 SHAP** | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Clinical Baseline** | 7 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 5 SHAP** | 5 | 0.9833 | 0.9901 | 0.9735 | 0.9833 | 1.0000 | 0.9667 |
| **Top 3 SHAP** | 3 | 0.8833 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |

**Performance retention vs. All Features:**

- **Top 20 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 15 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 10 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Clinical Baseline**: retention=50.0% (Δ=+0.5000) ❌
- **Top 5 SHAP**: retention=98.3% (Δ=+0.0167) ⚠️
- **Top 3 SHAP**: retention=88.3% (Δ=+0.1167) ❌

---

### XGBoost

| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |
|----------------|-----------|---------|-----|-----|----------|-------------|-------------|
| **All Features** | 23 | 0.9830 | 0.9800 | 0.9467 | 0.9733 | 0.9800 | 0.9667 |
| **Top 20 SHAP** | 20 | 0.9830 | 0.9800 | 0.9467 | 0.9733 | 0.9800 | 0.9667 |
| **Top 15 SHAP** | 15 | 0.9830 | 0.9800 | 0.9467 | 0.9733 | 0.9800 | 0.9667 |
| **Top 10 SHAP** | 10 | 0.9830 | 0.9800 | 0.9467 | 0.9733 | 0.9800 | 0.9667 |
| **Clinical Baseline** | 7 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 5 SHAP** | 5 | 0.9667 | 0.9709 | 0.9214 | 0.9500 | 1.0000 | 0.9000 |
| **Top 3 SHAP** | 3 | 0.9500 | 0.9346 | 0.8201 | 0.8833 | 1.0000 | 0.7667 |

**Performance retention vs. All Features:**

- **Top 20 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 15 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 10 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Clinical Baseline**: retention=50.9% (Δ=+0.4830) ❌
- **Top 5 SHAP**: retention=98.3% (Δ=+0.0163) ⚠️
- **Top 3 SHAP**: retention=96.6% (Δ=+0.0330) ❌

---

### RandomForest

| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |
|----------------|-----------|---------|-----|-----|----------|-------------|-------------|
| **All Features** | 24 | 0.9830 | 0.9901 | 0.9735 | 0.9833 | 1.0000 | 0.9667 |
| **Top 20 SHAP** | 20 | 0.9830 | 0.9901 | 0.9735 | 0.9833 | 1.0000 | 0.9667 |
| **Top 15 SHAP** | 15 | 0.9830 | 0.9901 | 0.9735 | 0.9833 | 1.0000 | 0.9667 |
| **Top 10 SHAP** | 10 | 0.9833 | 0.9804 | 0.9473 | 0.9667 | 1.0000 | 0.9333 |
| **Clinical Baseline** | 7 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 5 SHAP** | 5 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 3 SHAP** | 3 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |

**Performance retention vs. All Features:**

- **Top 20 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 15 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 10 SHAP**: retention=100.0% (Δ=-0.0003) ✅
- **Clinical Baseline**: retention=50.9% (Δ=+0.4830) ❌
- **Top 5 SHAP**: retention=50.9% (Δ=+0.4830) ❌
- **Top 3 SHAP**: retention=50.9% (Δ=+0.4830) ❌

---

### LightGBM

| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |
|----------------|-----------|---------|-----|-----|----------|-------------|-------------|
| **All Features** | 23 | 0.9900 | 0.9899 | 0.9739 | 0.9900 | 0.9800 | 1.0000 |
| **Top 20 SHAP** | 20 | 0.9900 | 0.9899 | 0.9739 | 0.9900 | 0.9800 | 1.0000 |
| **Top 15 SHAP** | 15 | 0.9900 | 0.9899 | 0.9739 | 0.9900 | 0.9800 | 1.0000 |
| **Top 10 SHAP** | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Clinical Baseline** | 7 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 5 SHAP** | 5 | 0.9667 | 0.9804 | 0.9473 | 0.9667 | 1.0000 | 0.9333 |
| **Top 3 SHAP** | 3 | 0.8833 | 0.8475 | 0.5423 | 0.7000 | 1.0000 | 0.4000 |

**Performance retention vs. All Features:**

- **Top 20 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 15 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 10 SHAP**: retention=101.0% (Δ=-0.0100) ✅
- **Clinical Baseline**: retention=50.5% (Δ=+0.4900) ❌
- **Top 5 SHAP**: retention=97.6% (Δ=+0.0233) ❌
- **Top 3 SHAP**: retention=89.2% (Δ=+0.1067) ❌

---

### LogisticRegression

| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |
|----------------|-----------|---------|-----|-----|----------|-------------|-------------|
| **All Features** | 24 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 20 SHAP** | 20 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 15 SHAP** | 15 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 10 SHAP** | 10 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Clinical Baseline** | 7 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 5 SHAP** | 5 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |
| **Top 3 SHAP** | 3 | 0.5000 | 0.7692 | 0.0000 | 0.5000 | 1.0000 | 0.0000 |

**Performance retention vs. All Features:**

- **Top 20 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 15 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 10 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Clinical Baseline**: retention=100.0% (Δ=+0.0000) ✅
- **Top 5 SHAP**: retention=100.0% (Δ=+0.0000) ✅
- **Top 3 SHAP**: retention=100.0% (Δ=+0.0000) ✅

---

## Interpretation

| Symbol | Meaning |
|--------|---------|
| ✅ | ROC-AUC drop < 0.005 — effectively equivalent |
| ⚠️ | ROC-AUC drop 0.005–0.020 — slight trade-off |
| ❌ | ROC-AUC drop > 0.020 — meaningful performance loss |

## Top 15 SHAP-Ranked Features

| Rank | Feature | Mean |SHAP| |
|------|---------|----------|
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
| 11 | `serum_creatinine` | 0.2187 |
| 12 | `diabetes_mellitus` | 0.2165 |
| 13 | `hypertension` | 0.1730 |
| 14 | `bun_creatinine_ratio` | 0.1250 |
| 15 | `red_blood_cell_count` | 0.1240 |

---

## Methodology Note

Non-selected features are replaced with their **training-set mean** before
prediction. The model is not retrained. This is leakage-safe because the
imputation values come from the training set only.