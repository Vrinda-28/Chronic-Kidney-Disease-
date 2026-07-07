"""
uae_validation.py
=================

Scientifically rigorous UAE external validation for the CKD pipeline.

WHY THIS MODULE EXISTS
----------------------
The original _evaluate_uae() in model_training.py filled 19/23 missing UCI
features with 0.0. This is NOT neutral — it creates clinically impossible
values (blood_pressure=0, hemoglobin=0, sodium=0) that the UCI-trained model
maps to severe CKD, causing it to predict CKD for every UAE patient.
The reported 11.4% accuracy / TN=0 result is an artefact of this imputation
strategy, not a real generalization signal.

CORRECT APPROACH
----------------
Two complementary validation tracks are implemented:

  Track A — Reduced-feature UCI model (primary, publication-valid)
  ─────────────────────────────────────────────────────────────────
  Train a NEW CatBoost/RF/etc. model on the UCI training set using ONLY the
  8 features that genuinely align between UCI and UAE (via direct match or
  documented semantic mapping). Apply this model to UAE. Report as the
  external validation result.

  This is the standard approach in cross-cohort clinical ML studies
  (see: Rajpurkar et al. 2022, Nature Medicine "AI in health and medicine").
  A reduced-feature model that generalizes is stronger evidence than a
  full-feature model that cannot be validated externally.

  Track B — Feature-aligned full inference (supplementary, clearly caveated)
  ──────────────────────────────────────────────────────────────────────────
  For completeness, the original full model is also evaluated after applying
  principled semantic feature mapping (NOT zero-filling). Missing features
  that have no UAE equivalent (specific_gravity, albumin, etc.) are imputed
  with the TRAINING SET MEDIAN from UCI, not zero. This is still imperfect
  but is far less biased than zero-filling and gives an honest upper-bound
  estimate of what the full model would achieve on this cohort. Clearly
  labelled as "approximate / caveated" in all artifacts.

FEATURE ALIGNMENT MAP
---------------------
Direct overlaps (same column name, same clinical meaning):
  serum_creatinine, age, cardiovascular_burden_score,
  age_creatinine_interaction, bp_risk_score

Semantic mappings (different column name, same clinical construct,
both binary-encoded as 0/1):
  UCI: hypertension            → UAE: history_hypertension
  UCI: diabetes_mellitus       → UAE: history_diabetes
  UCI: coronary_artery_disease → UAE: history_chd

bp_risk_score caveat: UCI computes it from a single blood_pressure reading;
UAE computes it as (systolic_bp + diastolic_bp) / 2. Both are valid mean BP
proxies. Document in methods section.

cardiovascular_burden_score caveat: UAE uses history_* flag columns (past
diagnoses), UCI uses current binary flags. Semantically equivalent for
population-level prediction; note in paper's limitations section.

WHAT THIS MODULE DOES NOT DO
-----------------------------
- Does not fill missing features with 0
- Does not apply SMOTE to UAE
- Does not fit any imputer/scaler on UAE data
- Does not treat UAE validation as model selection criterion
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


logger = logging.getLogger("ckd_model_trainer")


# =============================================================================
# Feature alignment registry
# =============================================================================

# Direct feature overlaps: features that exist in UCI engineered output
# AND in UAE engineered output under the same column name.
UCI_UAE_DIRECT_OVERLAPS: List[str] = [
    "serum_creatinine",
    "age",
    "cardiovascular_burden_score",
    "age_creatinine_interaction",
    "bp_risk_score",
]

# Semantic feature mappings: UCI column name → UAE column name.
# Both are binary-encoded (0/1) with the same clinical meaning.
UCI_TO_UAE_SEMANTIC_MAP = {
    "hypertension": "history_hypertension",
    "diabetes_mellitus": "history_diabetes",
    "coronary_artery_disease": "history_chd",
}

# Caveats to document in paper for mapped/bridged features.
FEATURE_MAPPING_CAVEATS: Dict[str, str] = {
    "bp_risk_score": (
        "UCI: derived from single blood_pressure reading. "
        "UAE: derived as (systolic_bp + diastolic_bp) / 2. "
        "Both are mean BP proxies; formula difference noted as limitation."
    ),
    "cardiovascular_burden_score": (
        "UCI: uses current binary flags (hypertension, CAD, DM). "
        "UAE: uses history flags (history_hypertension, history_chd, history_diabetes). "
        "Semantically equivalent comorbidity counts; noted as limitation."
    ),
    "hypertension": (
        "UCI: current binary hypertension flag. "
        "UAE: history_hypertension (past diagnosis). Mapped as semantic equivalent."
    ),
    "diabetes_mellitus": (
        "UCI: current binary diabetes flag. "
        "UAE: history_diabetes (past diagnosis). Mapped as semantic equivalent."
    ),
    "coronary_artery_disease": (
        "UCI: current binary CAD flag. "
        "UAE: history_chd (past diagnosis). Mapped as semantic equivalent."
    ),
}

# The complete set of UCI features usable in the reduced model (in order).
REDUCED_FEATURE_SET: List[str] = UCI_UAE_DIRECT_OVERLAPS + list(UCI_TO_UAE_SEMANTIC_MAP.keys())

# Features in the original UCI model that have NO UAE equivalent.
# These are documented so the paper can state explicitly which variables
# were unavailable in the external cohort.
UCI_FEATURES_ABSENT_IN_UAE: List[str] = [
    "blood_pressure",
    "specific_gravity",
    "albumin",
    "blood_glucose_random",
    "blood_urea",
    "sodium",
    "potassium",
    "hemoglobin",
    "packed_cell_volume",
    "white_blood_cell_count",
    "red_blood_cell_count",
    "bun_creatinine_ratio",
    "sodium_potassium_ratio",
    "anemia_risk_score",
    "urea_creatinine_product",
    "hemoglobin_creatinine_ratio",
    "albumin_specific_gravity_interaction",
]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class UAEValidationReport:
    """Complete report from both UAE validation tracks."""

    # Track A: reduced-feature model
    track_a_valid: bool = False
    track_a_features_used: List[str] = field(default_factory=list)
    track_a_model_name: str = ""
    track_a_uci_cv_metrics: Dict[str, Any] = field(default_factory=dict)
    track_a_uci_test_metrics: Dict[str, Any] = field(default_factory=dict)
    track_a_uae_metrics: Dict[str, Any] = field(default_factory=dict)
    track_a_n_features: int = 0

    # Track B: full model with training-median imputation (caveated)
    track_b_valid: bool = False
    track_b_features_used: List[str] = field(default_factory=list)
    track_b_features_imputed_with_train_median: List[str] = field(default_factory=list)
    track_b_uae_metrics: Dict[str, Any] = field(default_factory=dict)

    # Alignment audit
    feature_alignment_map: Dict[str, str] = field(default_factory=dict)
    caveats: Dict[str, str] = field(default_factory=dict)
    uae_n_rows: int = 0
    uae_target_distribution: Dict[str, int] = field(default_factory=dict)

    # What went wrong with the original zero-fill approach (for paper narrative)
    original_zero_fill_diagnosis: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "validation_methodology": (
                "Track A (primary): reduced-feature UCI model trained on UCI "
                "training set using only features available in UAE. "
                "Track B (supplementary, caveated): full UCI model with "
                "training-median imputation for unavailable features (NOT "
                "zero-fill). Both tracks documented for transparency."
            ),
            "track_a": {
                "valid": self.track_a_valid,
                "description": (
                    "PRIMARY external validation result. UCI model retrained "
                    "using only UCI-UAE aligned features. No imputation needed. "
                    "This is the result to report in the paper."
                ),
                "n_features": self.track_a_n_features,
                "features_used": self.track_a_features_used,
                "model_name": self.track_a_model_name,
                "uci_cv_metrics": self.track_a_uci_cv_metrics,
                "uci_test_metrics": self.track_a_uci_test_metrics,
                "uae_metrics": self.track_a_uae_metrics,
            },
            "track_b": {
                "valid": self.track_b_valid,
                "description": (
                    "SUPPLEMENTARY result only. Full UCI model with "
                    "training-median imputation (NOT zero-fill) for missing "
                    "features. Report as approximate upper bound, with caveats. "
                    "DO NOT report as primary external validation."
                ),
                "features_used": self.track_b_features_used,
                "features_imputed_with_train_median": (
                    self.track_b_features_imputed_with_train_median
                ),
                "uae_metrics": self.track_b_uae_metrics,
            },
            "feature_alignment": {
                "direct_overlaps": UCI_UAE_DIRECT_OVERLAPS,
                "semantic_mappings": UCI_TO_UAE_SEMANTIC_MAP,
                "caveats": self.caveats,
                "features_absent_in_uae": UCI_FEATURES_ABSENT_IN_UAE,
            },
            "uae_cohort": {
                "n_rows": self.uae_n_rows,
                "target_distribution": self.uae_target_distribution,
            },
            "original_zero_fill_diagnosis": self.original_zero_fill_diagnosis,
        }


# =============================================================================
# Feature alignment utilities
# =============================================================================

def build_aligned_uae_matrix(
    uae_df: pd.DataFrame,
    uci_features_requested: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build a UAE feature matrix aligned to UCI feature names.

    For each UCI feature requested:
      1. If the feature exists in UAE with the same name → use it directly.
      2. If a semantic mapping exists (UCI name → UAE name) → rename the
         UAE column to the UCI feature name.
      3. If no mapping exists → mark as unmappable (do NOT fill with 0).

    Returns
    -------
    aligned_df: DataFrame with UCI column names where available.
    available_features: UCI feature names that were successfully aligned.
    unavailable_features: UCI feature names with no UAE equivalent.
    """
    aligned_cols: Dict[str, pd.Series] = {}
    available: List[str] = []
    unavailable: List[str] = []

    for uci_feat in uci_features_requested:
        if uci_feat in uae_df.columns:
            # Direct overlap
            aligned_cols[uci_feat] = uae_df[uci_feat].copy()
            available.append(uci_feat)
        elif uci_feat in UCI_TO_UAE_SEMANTIC_MAP:
            uae_col = UCI_TO_UAE_SEMANTIC_MAP[uci_feat]
            if uae_col in uae_df.columns:
                aligned_cols[uci_feat] = uae_df[uae_col].copy()
                available.append(uci_feat)
                logger.info(
                    "[UAE Alignment] Mapped UCI '%s' → UAE '%s' (semantic equivalence).",
                    uci_feat, uae_col,
                )
            else:
                unavailable.append(uci_feat)
                logger.warning(
                    "[UAE Alignment] Semantic mapping target '%s' (for UCI '%s') "
                    "not found in UAE — feature unavailable.",
                    uae_col, uci_feat,
                )
        else:
            unavailable.append(uci_feat)

    aligned_df = pd.DataFrame(aligned_cols)
    return aligned_df, available, unavailable


def compute_uci_train_medians(
    X_train: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Compute column medians from UCI training data for Track B imputation.
    Only called on the UCI training set — never on UAE data.
    """
    medians: Dict[str, float] = {}
    for col in feature_cols:
        if col in X_train.columns:
            medians[col] = float(X_train[col].median())
    return medians


# =============================================================================
# Metric computation (binary, with diagnostics)
# =============================================================================

def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    prefix: str = "",
) -> Dict[str, Any]:
    p = prefix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m: Dict[str, Any] = {
        f"{p}accuracy":          round(float(accuracy_score(y_true, y_pred)), 6),
        f"{p}balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        f"{p}precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}recall":            round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}sensitivity":       round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}specificity":       round(float(tn / max(tn + fp, 1)), 6),
        f"{p}f1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}mcc":               round(float(matthews_corrcoef(y_true, y_pred)), 6),
        f"{p}tp": int(tp), f"{p}tn": int(tn), f"{p}fp": int(fp), f"{p}fn": int(fn),
    }
    if y_proba is not None:
        try:
            m[f"{p}roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        except ValueError:
            m[f"{p}roc_auc"] = None
        try:
            m[f"{p}pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 6)
        except ValueError:
            m[f"{p}pr_auc"] = None
    return m


# =============================================================================
# Track A: reduced-feature model
# =============================================================================

# ---------------------------------------------------------------------------
# Inline model builder (used when model_utils is unavailable)
# When used inside the full pipeline, model_utils.build_model is preferred.
# ---------------------------------------------------------------------------
def _build_model_inline(model_name: str, params: Dict[str, Any],
                         task_type: str, y_train: Optional[np.ndarray] = None) -> Any:
    p = dict(params)
    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**p)
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        p.pop("use_label_encoder", None)
        if task_type == "binary" and y_train is not None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            if n_pos > 0:
                p["scale_pos_weight"] = round(n_neg / n_pos, 4)
        elif task_type == "multiclass" and y_train is not None:
            p["num_class"] = int(len(np.unique(y_train)))
        return XGBClassifier(**p)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        if task_type == "multiclass" and y_train is not None:
            p.setdefault("objective", "multiclass")
            p["num_class"] = int(len(np.unique(y_train)))
        return LGBMClassifier(**p)
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        p["auto_class_weights"] = "Balanced"
        if task_type == "multiclass":
            p.setdefault("loss_function", "MultiClass")
        return CatBoostClassifier(**p)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_build_model():
    try:
        from model_utils import build_model
        return build_model
    except ImportError:
        return _build_model_inline


def _get_compute_binary_metrics():
    try:
        from model_utils import compute_binary_metrics
        return compute_binary_metrics
    except ImportError:
        return _binary_metrics


def run_track_a(
    best_model_name: str,
    model_cfg_params: Dict[str, Any],
    task_type: str,
    X_train_full: pd.DataFrame,
    y_train: np.ndarray,
    X_test_full: pd.DataFrame,
    y_test: np.ndarray,
    cv_fold_indices: List[Dict[str, Any]],
    uae_df: pd.DataFrame,
    y_uae: np.ndarray,
    random_seed: int,
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Train a reduced-feature UCI model using only UCI-UAE aligned features,
    evaluate it on the UCI test set (to show it still performs reasonably
    on internal data), then evaluate it on UAE.

    Returns
    -------
    cv_metrics: CV performance of the reduced model on UCI folds.
    uci_test_metrics: Test-set metrics for the reduced model.
    uae_metrics: UAE external validation metrics.
    """
    build_model = _get_build_model()
    compute_binary_metrics = _get_compute_binary_metrics()

    # Determine which REDUCED features are actually present in UCI training data
    reduced_features_available = [
        f for f in REDUCED_FEATURE_SET if f in X_train_full.columns
    ]

    if len(reduced_features_available) < 3:
        raise ValueError(
            f"Track A: only {len(reduced_features_available)} reduced features "
            f"found in UCI training data ({reduced_features_available}). "
            f"Minimum 3 required for a meaningful model."
        )

    logger.info(
        "[UAE Track A] Reduced feature set (%d features): %s",
        len(reduced_features_available), reduced_features_available,
    )

    X_train_reduced = X_train_full[reduced_features_available]
    X_test_reduced = X_test_full[reduced_features_available]

    # ── CV evaluation of reduced model ────────────────────────────────────
    fold_metrics_reduced: List[Dict[str, Any]] = []
    for fold in cv_fold_indices:
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]

        X_ft = X_train_reduced.iloc[train_idx].values
        y_ft = y_train[train_idx]
        X_fv = X_train_reduced.iloc[val_idx].values
        y_fv = y_train[val_idx]

        m = build_model(best_model_name, model_cfg_params, task_type, y_train=y_ft)
        m.fit(X_ft, y_ft)
        y_pred_v = m.predict(X_fv)
        y_proba_v = m.predict_proba(X_fv)[:, 1] if hasattr(m, "predict_proba") else None
        fm = compute_binary_metrics(y_fv, y_pred_v, y_proba_v)
        fm["fold_num"] = fold["fold_num"]
        fold_metrics_reduced.append(fm)

    # CV summary
    numeric_keys = [k for k in fold_metrics_reduced[0] if isinstance(fold_metrics_reduced[0][k], (int, float))]
    cv_summary: Dict[str, Any] = {}
    for key in numeric_keys:
        vals = [fm[key] for fm in fold_metrics_reduced if key in fm]
        arr = np.array(vals, dtype=float)
        cv_summary[key] = {
            "mean": round(float(arr.mean()), 6),
            "std":  round(float(arr.std()), 6),
        }

    logger.info(
        "[UAE Track A] Reduced model CV ROC-AUC: %.4f ± %.4f",
        cv_summary.get("roc_auc", {}).get("mean", 0),
        cv_summary.get("roc_auc", {}).get("std", 0),
    )

    # ── Final refit on full UCI training data ──────────────────────────────
    final_model_reduced = build_model(
        best_model_name, model_cfg_params, task_type, y_train=y_train
    )
    final_model_reduced.fit(X_train_reduced.values, y_train)

    # ── UCI test set evaluation ────────────────────────────────────────────
    y_test_pred = final_model_reduced.predict(X_test_reduced.values)
    y_test_proba = (
        final_model_reduced.predict_proba(X_test_reduced.values)[:, 1]
        if hasattr(final_model_reduced, "predict_proba") else None
    )
    uci_test_metrics = compute_binary_metrics(
        y_test, y_test_pred, y_test_proba, prefix="reduced_test_"
    )
    logger.info(
        "[UAE Track A] Reduced model UCI test ROC-AUC: %.4f | F1: %.4f",
        uci_test_metrics.get("reduced_test_roc_auc", 0),
        uci_test_metrics.get("reduced_test_f1", 0),
    )

    # ── Build aligned UAE matrix ───────────────────────────────────────────
    uae_aligned, available_uae, unavailable_uae = build_aligned_uae_matrix(
        uae_df, reduced_features_available
    )

    if len(available_uae) < 3:
        raise ValueError(
            f"Track A: only {len(available_uae)} features aligned in UAE "
            f"({available_uae}). Cannot run meaningful validation."
        )

    if unavailable_uae:
        logger.warning(
            "[UAE Track A] %d reduced features unavailable in UAE even after "
            "mapping (will be dropped from this evaluation): %s",
            len(unavailable_uae), unavailable_uae,
        )

    # For features in reduced_features_available not in uae_aligned,
    # impute with UCI training median (only for Track A if needed)
    final_features_for_uae = [f for f in reduced_features_available if f in uae_aligned.columns]
    final_model_for_uae = build_model(
        best_model_name, model_cfg_params, task_type, y_train=y_train
    )
    final_model_for_uae.fit(
        X_train_full[final_features_for_uae].values, y_train
    )

    X_uae_a = uae_aligned[final_features_for_uae].values

    y_uae_pred_a = final_model_for_uae.predict(X_uae_a)
    y_uae_proba_a = (
        final_model_for_uae.predict_proba(X_uae_a)[:, 1]
        if hasattr(final_model_for_uae, "predict_proba") else None
    )

    uae_metrics = _binary_metrics(y_uae, y_uae_pred_a, y_uae_proba_a, prefix="uae_")
    uae_metrics["uae_n_rows"] = len(uae_df)
    uae_metrics["uae_features_used"] = final_features_for_uae
    uae_metrics["uae_n_features"] = len(final_features_for_uae)
    uae_metrics["uae_unavailable_features"] = unavailable_uae
    uae_metrics["uae_model_name"] = f"{best_model_name}_reduced"

    logger.info(
        "[UAE Track A] ✔ UAE external validation — "
        "Acc: %.4f | ROC-AUC: %.4f | F1: %.4f | "
        "Sens: %.4f | Spec: %.4f | MCC: %.4f",
        uae_metrics.get("uae_accuracy", 0),
        uae_metrics.get("uae_roc_auc", 0),
        uae_metrics.get("uae_f1", 0),
        uae_metrics.get("uae_sensitivity", 0),
        uae_metrics.get("uae_specificity", 0),
        uae_metrics.get("uae_mcc", 0),
    )

    # Save model and artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model_for_uae, artifacts_dir / "track_a_reduced_model.joblib")

    with open(artifacts_dir / "track_a_feature_list.json", "w") as fh:
        json.dump({
            "reduced_features_in_uci_training": reduced_features_available,
            "features_actually_used_for_uae": final_features_for_uae,
            "features_unavailable_in_uae": unavailable_uae,
            "feature_mapping_applied": {
                k: v for k, v in UCI_TO_UAE_SEMANTIC_MAP.items()
                if k in final_features_for_uae
            },
            "caveats": {k: FEATURE_MAPPING_CAVEATS[k]
                        for k in final_features_for_uae if k in FEATURE_MAPPING_CAVEATS},
        }, fh, indent=2)

    preds_df = pd.DataFrame({
        "row_position": range(len(uae_df)),
        "y_true": y_uae,
        "y_pred": y_uae_pred_a,
        "y_proba_ckd": y_uae_proba_a if y_uae_proba_a is not None else [None] * len(uae_df),
    })
    preds_df.to_csv(artifacts_dir / "track_a_uae_predictions.csv", index=False)

    return cv_summary, uci_test_metrics, uae_metrics


# =============================================================================
# Track B: full model with training-median imputation (caveated)
# =============================================================================

def run_track_b(
    full_model: Any,
    union_features: List[str],
    X_train_full: pd.DataFrame,
    uae_df: pd.DataFrame,
    y_uae: np.ndarray,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """
    Apply the original full UCI model to UAE, but replace zero-fill with
    UCI training-median imputation for missing features. This is still
    an approximation (the model was not trained on a UAE-like feature
    distribution) but it is far less biased than zero-filling and
    gives an honest secondary estimate.

    Clearly caveated: the features imputed from training medians are
    documented explicitly. The result is labelled "supplementary / caveated"
    in all artifacts.
    """
    logger.info("[UAE Track B] Running full model with training-median imputation …")

    # Compute UCI training-set medians for imputation (never use UAE data)
    train_medians = compute_uci_train_medians(X_train_full, union_features)

    # Build UAE aligned matrix with training-median fallback
    aligned_parts: Dict[str, pd.Series] = {}
    imputed_with_median: List[str] = []
    aligned_from_uae: List[str] = []

    for uci_feat in union_features:
        if uci_feat in uae_df.columns:
            aligned_parts[uci_feat] = uae_df[uci_feat].copy()
            aligned_from_uae.append(uci_feat)
        elif uci_feat in UCI_TO_UAE_SEMANTIC_MAP:
            uae_col = UCI_TO_UAE_SEMANTIC_MAP[uci_feat]
            if uae_col in uae_df.columns:
                aligned_parts[uci_feat] = uae_df[uae_col].copy()
                aligned_from_uae.append(uci_feat)
            else:
                median_val = train_medians.get(uci_feat, 0.0)
                aligned_parts[uci_feat] = pd.Series(
                    [median_val] * len(uae_df), index=uae_df.index
                )
                imputed_with_median.append(uci_feat)
        else:
            median_val = train_medians.get(uci_feat, 0.0)
            aligned_parts[uci_feat] = pd.Series(
                [median_val] * len(uae_df), index=uae_df.index
            )
            imputed_with_median.append(uci_feat)

    X_uae_b = pd.DataFrame(aligned_parts)[union_features].values

    y_uae_pred_b = full_model.predict(X_uae_b)
    y_uae_proba_b = (
        full_model.predict_proba(X_uae_b)[:, 1]
        if hasattr(full_model, "predict_proba") else None
    )

    uae_metrics_b = _binary_metrics(y_uae, y_uae_pred_b, y_uae_proba_b, prefix="uae_b_")
    uae_metrics_b["uae_n_rows"] = len(uae_df)
    uae_metrics_b["features_from_uae"] = aligned_from_uae
    uae_metrics_b["features_imputed_from_uci_train_median"] = imputed_with_median
    uae_metrics_b["imputation_note"] = (
        "CAVEATED: Features imputed from UCI training-set median. "
        "The model was not trained expecting median-valued inputs for these "
        "clinical variables. Do NOT report this as the primary external "
        "validation result. Use Track A instead."
    )

    logger.info(
        "[UAE Track B] Caveated full-model UAE — "
        "Acc: %.4f | ROC-AUC: %.4f | Sens: %.4f | Spec: %.4f",
        uae_metrics_b.get("uae_b_accuracy", 0),
        uae_metrics_b.get("uae_b_roc_auc", 0),
        uae_metrics_b.get("uae_b_sensitivity", 0),
        uae_metrics_b.get("uae_b_specificity", 0),
    )

    # Save predictions
    preds_df_b = pd.DataFrame({
        "row_position": range(len(uae_df)),
        "y_true": y_uae,
        "y_pred": y_uae_pred_b,
        "y_proba_ckd": y_uae_proba_b if y_uae_proba_b is not None else [None] * len(uae_df),
    })
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preds_df_b.to_csv(artifacts_dir / "track_b_uae_predictions.csv", index=False)

    return uae_metrics_b


# =============================================================================
# Diagnosis: why the zero-fill approach failed
# =============================================================================

def diagnose_zero_fill_failure(
    full_model: Any,
    union_features: List[str],
    uae_df: pd.DataFrame,
    y_uae: np.ndarray,
    X_train_full: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Reproduce and explain the zero-fill result so it can be documented in
    the paper as a negative finding / methodological note.
    """
    zero_matrix = np.zeros((len(uae_df), len(union_features)), dtype=float)
    for i, feat in enumerate(union_features):
        if feat in uae_df.columns:
            zero_matrix[:, i] = uae_df[feat].values
        elif feat in UCI_TO_UAE_SEMANTIC_MAP:
            uae_col = UCI_TO_UAE_SEMANTIC_MAP[feat]
            if uae_col in uae_df.columns:
                zero_matrix[:, i] = uae_df[uae_col].values
            # else leave as 0

    y_zero_pred = full_model.predict(zero_matrix)
    tn, fp, fn, tp = confusion_matrix(y_uae, y_zero_pred, labels=[0, 1]).ravel()

    # Measure how different zero-filled values are from training medians
    train_medians = compute_uci_train_medians(X_train_full, union_features)
    zero_vs_median: Dict[str, Dict[str, float]] = {}
    for feat in union_features:
        if feat not in uae_df.columns and feat not in UCI_TO_UAE_SEMANTIC_MAP:
            train_med = train_medians.get(feat, float("nan"))
            zero_vs_median[feat] = {
                "zero_fill_value": 0.0,
                "uci_train_median": round(train_med, 4),
                "deviation_from_median": round(abs(train_med), 4),
            }

    return {
        "diagnosis": (
            "CRITICAL: Zero-fill imputation causes every UAE patient to be "
            "predicted as CKD. Root cause: 19 features are zero-filled. "
            "Zero is not a neutral value — for clinical lab measurements "
            "(e.g. hemoglobin=0, sodium=0, blood_pressure=0), zero lies "
            "far in the CKD tail of the UCI training distribution. "
            "The model maps these extreme values to CKD correctly given "
            "its training distribution. The zero-fill result should NOT "
            "be reported as an external validation result."
        ),
        "zero_fill_result": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": round(accuracy_score(y_uae, y_zero_pred), 6),
            "specificity": round(float(tn / max(tn + fp, 1)), 6),
            "sensitivity": round(float(recall_score(y_uae, y_zero_pred, zero_division=0)), 6),
        },
        "n_features_zeroed": len([f for f in union_features
                                   if f not in uae_df.columns
                                   and f not in UCI_TO_UAE_SEMANTIC_MAP]),
        "zero_vs_uci_train_median_for_zeroed_features": zero_vs_median,
        "recommendation": (
            "Use Track A (reduced-feature model) as the primary external "
            "validation result in the paper."
        ),
    }


# =============================================================================
# Main orchestration function
# =============================================================================

def run_uae_external_validation(
    best_model_name: str,
    best_model_params: Dict[str, Any],
    full_trained_model: Any,
    union_features: List[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv_fold_indices: List[Dict[str, Any]],
    uae_df: pd.DataFrame,
    target_col: str,
    random_seed: int,
    artifacts_dir: Path,
) -> UAEValidationReport:
    """
    Main entry point. Runs both validation tracks and returns a complete
    UAEValidationReport.

    Parameters
    ----------
    best_model_name:
        Model class to use (e.g. "CatBoost"). The reduced model uses the
        same architecture as the best UCI full model.
    best_model_params:
        Hyperparameter dict for the best model.
    full_trained_model:
        The already-trained full UCI model (for Track B).
    union_features:
        Features used by the full UCI model.
    X_train, y_train:
        UCI training data (full feature set).
    X_test, y_test:
        UCI test data (for Track A internal performance).
    cv_fold_indices:
        Pre-generated UCI CV fold indices.
    uae_df:
        UAE external validation DataFrame (features only, no target).
    target_col:
        Name of the target column in uae_df.
    random_seed:
        Global random seed.
    artifacts_dir:
        Where to write all validation artifacts.
    """
    start = time.time()
    report = UAEValidationReport()

    if target_col not in uae_df.columns:
        logger.error("[UAE Validation] Target column '%s' not in UAE — abort.", target_col)
        return report

    y_uae = uae_df[target_col].values.astype(int)
    report.uae_n_rows = len(uae_df)
    report.uae_target_distribution = {
        str(k): int(v)
        for k, v in pd.Series(y_uae).value_counts().items()
    }
    report.caveats = {
        k: v for k, v in FEATURE_MAPPING_CAVEATS.items()
        if k in REDUCED_FEATURE_SET
    }
    report.feature_alignment_map = UCI_TO_UAE_SEMANTIC_MAP

    logger.info(
        "[UAE Validation] UAE: %d rows | Target distribution: %s",
        report.uae_n_rows, report.uae_target_distribution,
    )

    # ── Diagnosis of zero-fill failure ───────────────────────────────────
    report.original_zero_fill_diagnosis = diagnose_zero_fill_failure(
        full_trained_model, union_features, uae_df, y_uae, X_train
    )

    # ── Track A: reduced-feature model ───────────────────────────────────
    try:
        cv_metrics_a, test_metrics_a, uae_metrics_a = run_track_a(
            best_model_name=best_model_name,
            model_cfg_params=best_model_params,
            task_type="binary",
            X_train_full=X_train,
            y_train=y_train,
            X_test_full=X_test,
            y_test=y_test,
            cv_fold_indices=cv_fold_indices,
            uae_df=uae_df,
            y_uae=y_uae,
            random_seed=random_seed,
            artifacts_dir=artifacts_dir / "track_a",
        )
        report.track_a_valid = True
        report.track_a_model_name = f"{best_model_name}_reduced"
        report.track_a_features_used = uae_metrics_a.get("uae_features_used", [])
        report.track_a_n_features = uae_metrics_a.get("uae_n_features", 0)
        report.track_a_uci_cv_metrics = cv_metrics_a
        report.track_a_uci_test_metrics = test_metrics_a
        report.track_a_uae_metrics = uae_metrics_a
    except Exception as exc:
        logger.error("[UAE Validation] Track A failed: %s", exc, exc_info=True)
        report.track_a_valid = False

    # ── Track B: full model with training-median imputation ───────────────
    try:
        uae_metrics_b = run_track_b(
            full_model=full_trained_model,
            union_features=union_features,
            X_train_full=X_train,
            uae_df=uae_df,
            y_uae=y_uae,
            artifacts_dir=artifacts_dir / "track_b",
        )
        report.track_b_valid = True
        report.track_b_features_used = union_features
        report.track_b_features_imputed_with_train_median = (
            uae_metrics_b.get("features_imputed_from_uci_train_median", [])
        )
        report.track_b_uae_metrics = uae_metrics_b
    except Exception as exc:
        logger.error("[UAE Validation] Track B failed: %s", exc, exc_info=True)
        report.track_b_valid = False

    # ── Save complete report ──────────────────────────────────────────────
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "uae_validation_full_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2, default=str)

    elapsed = round(time.time() - start, 2)
    logger.info(
        "[UAE Validation] Complete in %.1fs. Report: %s", elapsed, report_path.resolve()
    )
    if report.track_a_valid:
        a = report.track_a_uae_metrics
        logger.info(
            "[UAE Validation] ═══ TRACK A (PRIMARY RESULT) ═══ "
            "Acc: %.4f | ROC-AUC: %.4f | Sens: %.4f | Spec: %.4f | "
            "F1: %.4f | MCC: %.4f | Features: %d",
            a.get("uae_accuracy", 0),    a.get("uae_roc_auc", 0),
            a.get("uae_sensitivity", 0), a.get("uae_specificity", 0),
            a.get("uae_f1", 0),          a.get("uae_mcc", 0),
            a.get("uae_n_features", 0),
        )

    return report