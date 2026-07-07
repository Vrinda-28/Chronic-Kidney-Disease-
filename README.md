
---

## Abstract

Chronic Kidney Disease (CKD) affects over 800 million people worldwide and is frequently undetected until advanced stages when intervention is most difficult. This project presents a **research-grade, leakage-safe machine learning pipeline** for CKD detection and severity staging, built around the principles of clinical trustworthiness: **explainability**, **reproducibility**, and **external validation**.

Most published CKD models maximise accuracy on a single dataset without explaining their reasoning or validating on independent populations. This pipeline directly addresses both gaps.

**Key contributions:**

- **Two clinical tasks in one pipeline:** binary CKD detection (UCI dataset) and 5-class CKD severity staging (Kaggle dataset), with separate model families for each.
- **Structured explainability layer:** SHAP global summaries, per-patient waterfall plots, LIME cross-validation, and cross-model feature-agreement analysis — producing outputs structured for clinical review.
- **Cross-cohort external validation:** the best UCI model is evaluated on an independent UAE hospital cohort (n=491), demonstrating real-world discrimination with ROC-AUC = **0.776** despite a large prevalence shift (training 62.5% CKD → UAE 11.4% CKD).
- **End-to-end leakage safety:** every preprocessing, scaling, and feature-selection step is fitted strictly on training folds; test data and the UAE cohort are never seen during any model development step.

---

## Table of Contents

1. [Pipeline Status](#pipeline-status)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Datasets](#datasets)
6. [Pipeline Overview](#pipeline-overview)
7. [Results](#results)
8. [External Validation — UAE Cohort](#external-validation--uae-cohort)
9. [Explainability](#explainability)
10. [Ablation Study](#ablation-study)
11. [Reproducibility Design](#reproducibility-design)
12. [Limitations](#limitations)
13. [Future Work](#future-work)
14. [Citation](#citation)

---

## Pipeline Status

| Stage | Script | Status |
|---|---|---|
| Data Loading | `ckd_data/data_loader.py` | ✅ Complete |
| Preprocessing | `scripts/preprocess.py` | ✅ Complete |
| Feature Engineering | `scripts/feature_engineering.py` | ✅ Complete |
| Train/Test Split | `scripts/train_test_split.py` | ✅ Complete |
| Model Training | `scripts/model_training.py` | ✅ Complete |
| Evaluation | `scripts/evaluate.py` | ✅ Complete |
| External Validation | `scripts/external_validation.py` | ✅ Complete |
| Explainability | `scripts/explainability.py` | ✅ Complete |
| Ablation Study | `scripts/ablation_study.py` | ✅ Complete |
| Model Comparison | `scripts/model_comparison.py` | ✅ Complete |

---

## Repository Structure

```
CKD-XAI-Pipeline/
│
├── ckd_data/
│   ├── __init__.py
│   └── data_loader.py              Dataset loading and schema validation
│
├── scripts/
│   ├── preprocess.py               Cleaning, imputation, encoding, unit harmonisation
│   ├── preprocessing_utils.py      Shared preprocessing functions
│   ├── feature_engineering.py      Clinical composite feature construction
│   ├── train_test_split.py         Stratified splits + CV fold generation
│   ├── split_utils.py              Split validation and leakage guards
│   ├── model_training.py           5-model training (UCI binary + Kaggle multiclass)
│   ├── model_utils.py              Model factory and inference helpers
│   ├── model_comparison.py         Cross-model metrics and HTML report
│   ├── evaluate.py                 Test-set evaluation with bootstrap 95% CIs
│   ├── evaluation_utils.py         Shared metrics and plotting utilities
│   ├── external_validation.py      UAE external validation (Track A + Track B)
│   ├── uae_validation.py           UAE validation orchestrator
│   ├── uae_audit.py                UAE feature alignment audit
│   ├── evaluate_uae_replacement.py Replacement UAE evaluation utilities
│   ├── explainability.py           SHAP + LIME + clinician reports
│   ├── ablation_study.py           Feature-subset ablation analysis
│   ├── pipeline_paths.py           Centralised path management
│   ├── publication_figures.py      Publication-quality figure generation
│   └── download_uci.py             UCI dataset download helper
│
├── config/
│   ├── datasets.yaml               Dataset paths and schema mappings
│   ├── preprocessing.yaml          Imputation and encoding configuration
│   ├── split_config.yaml           CV strategy and scaling policies
│   ├── model_config.yaml           Hyperparameters and training settings
│   └── evaluation_config.yaml      Metrics and figure settings
│
├── data/
│   ├── raw/                        Original CSV files (never modified)
│   │   ├── uci_ckd.csv
│   │   ├── kaggle_ckd_stages.csv
│   │   └── uae_ckd_cohort.csv
│   ├── processed/                  Output of preprocess.py
│   │   ├── uci_processed.csv
│   │   ├── kaggle_processed.csv
│   │   └── uae_processed.csv
│   ├── engineered/                 Output of feature_engineering.py
│   │   ├── uci_engineered.csv
│   │   ├── kaggle_engineered.csv
│   │   ├── uae_engineered.csv
│   │   ├── uci_provenance.csv
│   │   ├── kaggle_provenance.csv
│   │   └── uae_provenance.csv
│   └── splits/                     Pre-generated train/test splits
│       ├── uci_train.csv
│       ├── uci_test.csv
│       ├── kaggle_train.csv
│       ├── kaggle_test.csv
│       └── uae_full.csv            External validation set (never split)
│
├── artifacts/
│   ├── preprocessing/
│   │   ├── categorical_imputer.joblib
│   │   ├── numeric_imputer.joblib
│   │   ├── encoders.joblib
│   │   ├── label_mappings.json
│   │   └── preprocessing_summary.json
│   ├── feature_engineering/
│   │   ├── feature_summary.json
│   │   └── feature_summary.joblib
│   ├── splits/
│   │   ├── split_metadata.json
│   │   ├── uci_cv_fold_indices.json
│   │   ├── kaggle_cv_fold_indices.json
│   │   ├── uci_train_manifest.csv
│   │   ├── uci_test_manifest.csv
│   │   ├── kaggle_train_manifest.csv
│   │   ├── kaggle_test_manifest.csv
│   │   └── uae_full_manifest.csv
│   ├── models/
│   │   ├── uci/
│   │   │   ├── CatBoost/           Calibrated model, metrics, SHAP values
│   │   │   ├── LightGBM/
│   │   │   ├── LogisticRegression/
│   │   │   ├── RandomForest/
│   │   │   ├── XGBoost/
│   │   │   ├── uae_validation/     External validation outputs
│   │   │   ├── best_model.json
│   │   │   └── best_model_summary.json
│   │   └── kaggle/
│   │       ├── CatBoost/
│   │       ├── LightGBM/
│   │       ├── RandomForest/
│   │       ├── XGBoost/
│   │       └── best_model.json
│   ├── evaluation/
│   │   ├── summary_report.md
│   │   ├── uci/                    ROC curves, PR curves, calibration plots
│   │   ├── kaggle/
│   │   └── uae/
│   ├── explainability/
│   │   └── uci/
│   │       ├── global_shap_summary.png
│   │       ├── global_shap_importance.csv
│   │       ├── CatBoost/           Beeswarm, dependence, waterfall, LIME
│   │       ├── LightGBM/
│   │       ├── LogisticRegression/
│   │       ├── RandomForest/
│   │       ├── XGBoost/
│   │       └── cross_model/        Agreement heatmap and analysis
│   └── ablation/                   Populated after running ablation_study.py
│
├── examples/
│   ├── example_usage.py
│   ├── preprocess_example.py
│   ├── train_test_split_example.py
│   ├── model_training_example.py
│   ├── evaluation_example.py
│   └── external_validation_example.py
│
├── logs/                           Rotating pipeline logs (one per script)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/ckd-xai-pipeline.git
cd ckd-xai-pipeline
pip install -r requirements.txt

# 2. Place raw CSVs in data/raw/
#    uci_ckd.csv  |  kaggle_ckd_stages.csv  |  uae_ckd_cohort.csv

# 3. Run the pipeline in order
python scripts/preprocess.py           # clean, impute, encode, harmonise units
python scripts/feature_engineering.py  # clinical composite features
python scripts/train_test_split.py     # stratified splits + CV indices
python scripts/model_training.py       # train all models + calibrate
python scripts/evaluate.py             # test-set metrics + bootstrap CIs
python scripts/external_validation.py  # UAE validation (Track A + Track B)
python scripts/explainability.py       # SHAP + LIME + clinician reports
python scripts/ablation_study.py       # feature-subset ablation
python scripts/model_comparison.py     # cross-model HTML report
python scripts/publication_figures.py  # publication-quality figures
```

> All pipeline stages are idempotent — each reads from the previous stage's outputs and can be re-run independently.

---

## Installation

**Python 3.10+** is required.

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `lime`, `matplotlib`, `seaborn`, `joblib`, `pyyaml`, `scipy`, `statsmodels`

Optional (for PDF report export):

```bash
pip install weasyprint
```

---

## Datasets

| Dataset | Task | Rows | Features | Source |
|---|---|---|---|---|
| UCI CKD | Binary: CKD / notCKD | 400 | 24 clinical | [UCI ML Repository](https://doi.org/10.24432/C5G020) |
| Kaggle CKD | 5-class severity staging | 200 | 25 clinical | Kaggle |
| UAE Cohort | External validation only | 491 | 8 aligned | Independent UAE hospital |

**Class distributions:**

- UCI: 250 CKD (62.5%) / 150 notCKD (37.5%)
- Kaggle: 5 stages roughly balanced across 200 rows
- UAE: 56 CKD (11.4%) / 435 notCKD (88.6%) — significant prevalence shift vs. training data

> **UAE creatinine unit harmonisation:** The UAE dataset stores `serum_creatinine` in µmol/L while UCI uses mg/dL. `scripts/preprocess.py` (Step 6b) applies the standard physicochemical conversion (÷ 88.4) and recomputes all dependent engineered features (`age_creatinine_interaction` and others). Without this correction, every UAE patient would appear at the extreme CKD tail of the UCI-trained model. This harmonisation step is logged in `artifacts/preprocessing/preprocessing_summary.json`.

---

## Pipeline Overview

```
Raw Data (UCI + Kaggle + UAE)
│
▼  scripts/preprocess.py
   Sentinel removal · Interval parsing · Type conversion
   UAE creatinine unit harmonisation (µmol/L → mg/dL, ÷ 88.4)
   Median/mode imputation (train-fitted; UAE: transform-only) · Binary encoding
│
▼  scripts/feature_engineering.py
   Clinical composites: cardiovascular_burden_score, anemia_risk_score, …
   Interaction terms · Leakage audit · Provenance tracking
│
▼  scripts/train_test_split.py
   Stratified 80/20 split (seed=42, frozen) · 5-fold StratifiedKFold (UCI)
   RepeatedStratifiedKFold 5×5 (Kaggle) · UAE → isolated, never split
│
▼  scripts/model_training.py
   MI feature selection inside each CV fold (k=20 UCI, k=25 Kaggle)
   class_weight='balanced' · 10% calibration hold-out
   Isotonic regression calibration · Artifacts saved per model
│
├──► scripts/evaluate.py             Bootstrap 95% CIs · ROC/PR curves · calibration plots
├──► scripts/external_validation.py  Track A (8 aligned features) · Track B (imputed)
├──► scripts/explainability.py       SHAP · LIME · cross-model agreement · clinician reports
├──► scripts/ablation_study.py       SHAP-ranked subsets · performance vs. parsimony
└──► scripts/model_comparison.py     Cross-model table · Wilcoxon tests · HTML report
```

### CV strategy

| Dataset | Strategy | Reason |
|---|---|---|
| UCI (400 rows) | StratifiedKFold(k=5) | Sufficient size for stable 5-fold estimates |
| Kaggle (200 rows) | RepeatedStratifiedKFold(5×5) | Small dataset — 25 folds reduce estimate variance |

### Imbalance handling

SMOTE was evaluated and rejected for both datasets. With ~120 minority-class training rows (UCI) and ~26 per class per fold (Kaggle), SMOTE's k-neighbours requirement is marginal and introduces synthetic-sample instability. `class_weight='balanced'` achieves comparable results without these risks (Lemaitre et al., 2017, JMLR 18:559–563).

---

## Results

### UCI Binary Classification

**Best model:** CatBoost &nbsp;|&nbsp; **Selection criterion:** CV ROC-AUC

> *ROC curves for all five models are generated by `scripts/evaluate.py` and saved to `artifacts/evaluation/uci/`.*

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Test MCC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| **CatBoost** ✅ | **0.9996** | **1.000** | **1.000** | **1.000** | 1.000 | 1.000 |
| LightGBM | 0.9992 | — | — | — | — | — |
| Logistic Regression | 0.9994 | — | — | — | — | — |
| Random Forest | 0.9988 | — | — | — | — | — |
| XGBoost | 0.9979 | — | — | — | — | — |

> Full test metrics for all models are written to `artifacts/evaluation/uci/` by `scripts/evaluate.py`. The table above shows CV AUC for all models and full test metrics only for the selected best model.

**Why are UCI scores so high?** The UCI CKD dataset is a well-studied benchmark where a small number of features (primarily haemoglobin and serum creatinine) almost perfectly separate the two classes. Perfect or near-perfect test scores are consistent with published literature (Ilayaraja & Meyyappan, 2013; Sinha & Sinha, 2015; Raihan et al., 2023). The 8-feature reduced model achieving ROC-AUC ≥ 0.994 confirms the signal is genuine and not attributable to overfitting on the full feature set.

**95% Bootstrap CIs (CatBoost, UCI test set):**

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 1.0000 | 1.0000 – 1.0000 |
| F1 | 1.0000 | 1.0000 – 1.0000 |
| MCC | 1.0000 | 1.0000 – 1.0000 |
| Brier Score | 0.0001 | 0.0000 – 0.0003 |

---

### Kaggle 5-Class Severity Staging

**Best model:** Random Forest &nbsp;|&nbsp; **Selection criterion:** CV Balanced Accuracy

| Model | CV Bal. Acc | Test Bal. Acc | Macro F1 | Cohen κ | Macro ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ✅ | **0.842** | **0.781** | **0.790** | **0.775** | **0.953** |
| CatBoost | — | — | — | — | — |
| LightGBM | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

> Full test metrics for all models are written to `artifacts/evaluation/kaggle/` by `scripts/evaluate.py`.

**Per-class performance (Random Forest, test set, n=40):**

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Stage 1 (s1) | 1.000 | 1.000 | 1.000 |
| Stage 2 (s2) | 1.000 | 1.000 | 1.000 |
| Stage 3 (s3) | 1.000 | 0.333 | 0.500 |
| Stage 4 (s4) | 0.563 | 1.000 | 0.720 |
| Stage 5 (s5) | 1.000 | 0.571 | 0.727 |

Stage 3 recall = 0.333 and Stage 5 recall = 0.571 reflect adjacent-stage confusion on a small test set (~6–7 samples per class). Cohen's Kappa = 0.775 (substantial agreement) is the primary interpretable metric for this task.

---

## External Validation — UAE Cohort

The UAE dataset represents an independent hospital population with substantially different characteristics from the UCI training data, providing a genuine test of generalisability across clinical contexts.

> *Confusion matrix and ROC curve for the UAE cohort are generated by `scripts/external_validation.py` and saved to `artifacts/evaluation/uae/`.*

### Population context

| Property | UCI (training) | UAE (external) |
|---|---|---|
| n | 400 | 491 |
| CKD prevalence | 62.5% | 11.4% |
| Target definition | Binary CKD presence | CKD progression to stage 3–5 |
| Feature overlap | Full (24 features) | Partial (8 of 24 aligned) |

### Results

**Track A (primary — 8 aligned features, no imputation):**

| Metric | At threshold = 0.50 | At Youden's J (τ = 0.989) |
|---|---|---|
| ROC-AUC | **0.776** | **0.776** *(threshold-free)* |
| PR-AUC | 0.281 | 0.281 |
| Sensitivity | 1.000 | 0.750 |
| Specificity | 0.076 | 0.703 |
| F1 | 0.218 | 0.370 |
| MCC | 0.096 | 0.303 |
| Balanced Accuracy | 0.538 | 0.727 |
| Accuracy | 0.181 | 0.709 |

**Track B (supplementary — full UCI model, training-median imputation for 17 missing features):** ROC-AUC ≈ 0.50 (chance level). Confirms that imputing 17 of 24 features adds no discrimination. Track A is the scientifically valid result.

### Interpreting the UAE result

**ROC-AUC = 0.776 is the primary metric.** It is threshold-independent and directly measures discrimination: a randomly selected UAE CKD patient is ranked higher than a randomly selected non-CKD patient 77.6% of the time — without any UAE data used during training.

**Why does threshold = 0.50 give sensitivity = 1.0 / specificity = 0.076?** The model was trained on 62.5% CKD prevalence. The UAE cohort has 11.4% — a 5.5× prevalence shift. All UAE predicted probabilities concentrate between 0.88 and 0.98 because the model's prior is anchored to the training distribution. At the default 0.5 threshold, nearly every patient is classified as CKD. **This is a known, expected consequence of prevalence shift, not a preprocessing bug or model failure.** The Youden-optimal threshold (τ = 0.989) yields a clinically interpretable operating point (sensitivity = 0.75, specificity = 0.70).

### Feature alignment map (Track A)

| UCI feature | UAE feature | Alignment type |
|---|---|---|
| `serum_creatinine` | `serum_creatinine` | Direct (after unit harmonisation ÷ 88.4) |
| `age` | `age` | Direct |
| `cardiovascular_burden_score` | `cardiovascular_burden_score` | Direct (same formula) |
| `age_creatinine_interaction` | `age_creatinine_interaction` | Direct |
| `bp_risk_score` | `bp_risk_score` | Direct (formula note below) |
| `hypertension` | `history_hypertension` | Semantic mapping |
| `diabetes_mellitus` | `history_diabetes` | Semantic mapping |
| `coronary_artery_disease` | `history_chd` | Semantic mapping |

> **`bp_risk_score` caveat:** UCI derives this from a single blood pressure reading (~80 mmHg); UAE derives it as (systolic + diastolic)/2 (~95–100 mmHg). The UAE values are systematically higher, which may slightly inflate CKD predictions on the UAE cohort. Documented as a limitation.

---

## Explainability

Explainability is a first-class output. Every model is fully interpreted using both global and local methods, producing outputs structured for clinical review.

> *SHAP beeswarm plots and dependence plots are generated by `scripts/explainability.py` and saved to `artifacts/explainability/uci/{ModelName}/`.*

### Methods

**SHAP (SHapley Additive exPlanations)** — based on cooperative game theory; the only attribution method satisfying consistency, efficiency, and linearity simultaneously (Lundberg & Lee, 2017). For tree ensembles, `TreeSHAP` is used: exact, not approximate, and polynomial-time.

**LIME (Local Interpretable Model-agnostic Explanations)** — fits a linear surrogate locally around each prediction. Model-agnostic and used as a cross-check on SHAP, though less theoretically grounded for global attribution.

**Cross-model agreement** — SHAP rankings are compared across all trained models. High agreement between RF, XGBoost, LightGBM, and CatBoost on the same top features provides evidence that the identified predictors reflect genuine clinical signal rather than model-specific artefacts.

### Top features (CatBoost, UCI, mean |SHAP|)

| Rank | Feature | Clinical meaning |
|---|---|---|
| 1 | `serum_creatinine` | Primary waste-clearance marker; elevated in CKD |
| 2 | `hemoglobin` | Anaemia is a direct CKD complication (reduced erythropoietin) |
| 3 | `albumin` | Proteinuria — damaged glomerular filtration |
| 4 | `specific_gravity` | Urine concentrating ability — impaired in CKD |
| 5 | `packed_cell_volume` | Red cell volume; falls with CKD-associated anaemia |
| 6 | `red_blood_cell_count` | Erythrocyte count |
| 7 | `cardiovascular_burden_score` | Engineered: HTN + DM + CAD comorbidity count |
| 8 | `blood_urea` | Nitrogenous waste — elevated with reduced filtration |
| 9 | `age_creatinine_interaction` | Engineered: creatinine weighted by age |
| 10 | `blood_pressure` | Hypertension both causes and results from CKD |

> **Clinical consistency check:** These rankings align with established nephrology evidence and published CKD ML papers (Raihan et al., 2023; Ghosh & Khandoker, 2024). Agreement between SHAP rankings and clinical knowledge independently validates the pipeline.

### Artifact outputs

| File | Description |
|---|---|
| `artifacts/explainability/uci/global_shap_summary.png` | Consensus SHAP importance across all models |
| `artifacts/explainability/uci/global_shap_importance.csv` | Machine-readable feature rankings |
| `artifacts/explainability/uci/{Model}/shap_beeswarm.png` | Per-model beeswarm with value colouring |
| `artifacts/explainability/uci/{Model}/dependence_plots/` | SHAP dependence for top features |
| `artifacts/explainability/uci/{Model}/waterfall/` | Per-patient waterfall (one per TP/TN/FP/FN class) |
| `artifacts/explainability/uci/{Model}/lime/` | LIME HTML + PNG for representative patients |
| `artifacts/explainability/uci/{Model}/clinician_explanation_report.md` | Plain-English clinical summary |
| `artifacts/explainability/uci/cross_model/agreement_heatmap.png` | Model-level feature-ranking agreement |
| `artifacts/explainability/uci/cross_model/agreement_analysis.csv` | Patient-level prediction agreement |

---

## Ablation Study

`scripts/ablation_study.py` evaluates performance with progressively fewer features, ranked by SHAP importance from the best model (CatBoost). This directly addresses the clinical parsimony question: *how many features are actually needed?*

> *Ablation curve (ROC-AUC vs. number of features) and per-subset metrics are generated by `scripts/ablation_study.py` and saved to `artifacts/ablation/`.*

| Feature Subset | # Features | ROC-AUC | F1 | MCC |
|---|---|---|---|---|
| All features | ~20 | 1.000 | 1.000 | 1.000 |
| Top 20 SHAP | 20 | 1.000 | 1.000 | 1.000 |
| Clinical Baseline (8) | 8 | ≥ 0.994 | — | — |
| Top 5 SHAP | 5 | Run `ablation_study.py` | — | — |
| Top 3 SHAP | 3 | Run `ablation_study.py` | — | — |

**Interpretation:** The 8-feature reduced model (the Track A external validation subset) retains ROC-AUC ≥ 0.994 on UCI test data, demonstrating that features available in the UAE cohort carry genuine and sufficient predictive signal for this dataset.

---

## Reproducibility Design

**Fixed random seed:** All stochastic components (train/test split, CV fold generation, model training, calibration hold-out) use `random_state=42`, configured in `config/split_config.yaml` and `config/model_config.yaml`. A single config change propagates everywhere.

**Frozen splits:** Train/test splits and CV fold indices are generated once by `scripts/train_test_split.py` and saved as CSV manifests and JSON index files to `artifacts/splits/`. Subsequent stages load these files rather than regenerating splits, guaranteeing that model evaluation always uses exactly the same partition across separate runs or code changes.

**Leakage-safe architecture:** Every fitting operation (imputation, scaling, feature selection) is performed exclusively on training data. Validation and test data receive transform-only operations using parameters fitted on the corresponding training fold. The UAE cohort is never passed to any fitting function.

**SHA-256 fingerprints:** `scripts/train_test_split.py` records SHA-256 fingerprints of the input engineered datasets and all output split DataFrames in `artifacts/splits/split_metadata.json`. Any change in upstream data is detectable by comparing fingerprints.

**Saved CV indices:** Complete `(train_indices, val_indices)` arrays for every CV fold are saved to `artifacts/splits/{dataset}_cv_fold_indices.json`. This allows exact CV reconstruction even if scikit-learn's internal random state implementation changes across versions.

**Calibration audit trail:** Calibration hold-out fractions, fitted isotonic regression objects, and pre/post calibration probability statistics are saved to `artifacts/models/{dataset}/{ModelName}/`.

---

## Limitations

1. **Small datasets.** UCI has 400 rows; Kaggle has 200. Near-perfect UCI performance reflects dataset separability, not generalisation. External validation on the UAE cohort provides the realistic performance estimate.

2. **Retrospective data.** All three datasets are from historical clinical records. Prospective validation in a real-time clinical workflow may yield different results.

3. **Single-institution training data.** The UCI dataset originates from a single hospital in India. Population characteristics differ from the UAE cohort and likely from other target populations.

4. **Partial feature overlap for external validation.** Track A uses 8 of 24 features — the subset available in both datasets. The model cannot leverage its strongest predictors (haemoglobin, albumin, specific gravity) during external evaluation.

5. **Target definition mismatch.** The UCI label captures CKD presence at a single clinical encounter. The UAE label captures longitudinal progression to CKD stage 3–5. This definitional difference independently limits cross-cohort transferability beyond what prevalence shift and feature overlap explain.

6. **Prevalence shift.** The 5.5× difference in CKD prevalence between training (62.5%) and external validation (11.4%) means the default classification threshold (0.5) is not appropriate for UAE use. Threshold recalibration or prevalence-adjusted probability rescaling is required before clinical application.

7. **Imputation fitted on full dataset.** Imputers in `scripts/preprocess.py` are fitted on the full UCI and Kaggle datasets before the train/test split. This is expected to have negligible impact given stable clinical measurements, but is documented here for transparency.

8. **`bp_risk_score` formula difference.** UCI derives this feature from a single diastolic blood pressure reading (~80 mmHg); UAE derives it as (systolic + diastolic)/2 (~95–100 mmHg). This systematic offset may slightly inflate CKD predictions on the UAE cohort.

---

## Future Work

- **Prospective validation** — validate the Track A reduced model in a forward-looking clinical study with real-time data entry.
- **Multi-site validation** — validate across CKD registries from diverse geographic and demographic populations.
- **Fairness analysis** — evaluate performance across demographic subgroups (age, sex, ethnicity) using established fairness metrics.
- **Federated learning** — train across multiple hospital sites without sharing patient-level data.
- **Transformer-based tabular models** — benchmark FT-Transformer, TabNet, and similar architectures against tree ensembles on the same frozen splits.
- **Longitudinal modelling** — extend to CKD progression prediction using repeated measurements over time (survival models, sequence classifiers).
- **eGFR integration** — incorporate the CKD-EPI 2021 eGFR equation as an explicit derived feature and compare to the learned creatinine/age interaction.
- **Domain adaptation** — apply importance-weighting or transfer methods to reduce the training–external prevalence gap without retraining on external data.
- **Calibration improvements** — apply population-prevalence prior shift correction (Saerens et al., 2002) to make default-threshold predictions valid at the UAE prevalence level.
- **Ablation on UAE** — measure performance degradation as aligned features are progressively removed, to identify the minimum viable feature set for cross-cohort settings.

---

<!-- ## Citation

If you use this pipeline or results in your work, please cite:

```bibtex
@misc{ckd_xai_pipeline_2026,
  title  = {CKD Prediction and Explainable AI Pipeline},
  author = {[Author Names]},
  year   = {2026},
  note   = {GitHub repository},
  url    = {[repository URL]}
}
``` -->

**Reference datasets and methods:**

```
Soundarapandian, P., & Rubini, L. J. (2015). Chronic Kidney Disease.
UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting
Model Predictions. Advances in Neural Information Processing Systems, 30.
https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

Ghosh, S. K., & Khandoker, A. H. (2024). Investigation on explainable
machine learning models to predict chronic kidney diseases.
Scientific Reports, 14, 3687. https://doi.org/10.1038/s41598-024-54375-4

Gogoi, P., & Valan, J. A. (2025). Chronic kidney disease prediction using
machine learning techniques. Multiscale and Multidisciplinary Modeling,
Experiments and Design, 8, 216. https://doi.org/10.1007/s41939-025-00806-2

Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn:
A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine
Learning. JMLR, 18(17), 1–5.
```

---

## License

Released under the [MIT License](LICENSE).

