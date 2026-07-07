"""
model_utils.py
==============

Pure, stateless utility functions for the CKD model training pipeline.

Everything here is side-effect-free and independently unit-testable.
File I/O and sklearn object management live in model_training.py, not here.

Pipeline position: imported exclusively by model_training.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ckd_model_trainer")


# =============================================================================
# Exceptions
# =============================================================================


class ModelTrainingError(Exception):
    """Raised for irrecoverable training-pipeline errors."""


class LeakageViolation(ModelTrainingError):
    """
    Raised when a data-leakage constraint is violated at training time.
    Training must halt — a contaminated model is worse than no model.
    """


# =============================================================================
# Leakage guards
# =============================================================================


def assert_no_target_in_features(
    feature_cols: List[str],
    target_col: str,
    context: str = "",
) -> None:
    """
    Raise LeakageViolation if the target column appears in the feature list.
    This guards against the classic mistake of forgetting to drop the target
    before calling model.fit(X, y).
    """
    if target_col in feature_cols:
        raise LeakageViolation(
            f"TARGET LEAKAGE [{context}]: target column '{target_col}' "
            f"is present in the feature column list. "
            f"Drop it before passing features to the model."
        )


def assert_no_uae_in_training(
    train_df: pd.DataFrame,
    uae_df: pd.DataFrame,
    context: str = "",
) -> None:
    """
    Raise LeakageViolation if the UAE DataFrame is the same Python object
    as the training DataFrame. This is the structural isolation check —
    if the same object appears in both roles, UAE rows are in training.
    """
    if train_df is uae_df:
        raise LeakageViolation(
            f"UAE CONTAMINATION [{context}]: train_df and uae_df are the "
            f"SAME Python object. UAE data is in the training set. "
            f"This is a structural bug — halt and investigate."
        )


def assert_val_not_in_train_indices(
    train_indices: List[int],
    val_indices: List[int],
    fold_num: int,
    dataset_name: str,
) -> None:
    """
    Raise LeakageViolation if any val index appears in train indices
    within a CV fold (fold-level train/val overlap check).
    """
    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise LeakageViolation(
            f"[{dataset_name}] CV FOLD {fold_num}: {len(overlap)} indices "
            f"appear in both train_indices and val_indices. "
            f"A row cannot be used for both training and validation in the same fold."
        )


def assert_feature_selector_not_fitted_on_val(
    selector_fit_size: int,
    train_fold_size: int,
    fold_num: int,
    dataset_name: str,
) -> None:
    """
    Guard: the feature selector must have been fitted on training fold data,
    not on validation or test data. Checks that the selector's input size
    matches the training fold size (not larger).
    If selector_fit_size > train_fold_size, it was likely fitted on the
    wrong DataFrame.
    """
    if selector_fit_size > train_fold_size:
        raise LeakageViolation(
            f"[{dataset_name}] FEATURE SELECTION LEAKAGE — Fold {fold_num}: "
            f"Feature selector was fitted on {selector_fit_size} rows, "
            f"but the training fold only has {train_fold_size} rows. "
            f"The selector was fitted on validation or test data. "
            f"Halt — results are invalid."
        )


# =============================================================================
# Metric computation
# =============================================================================


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute the full suite of binary classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_pred:
        Predicted labels (0/1).
    y_proba:
        Predicted probabilities for class 1 (shape: [n_samples]).
        If None, AUC-based metrics are set to None.
    prefix:
        String prefix for all metric keys (e.g. "cv_fold_3_").

    Returns
    -------
    dict of metric_name → float, JSON-serialisable.
    """
    p = prefix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: Dict[str, Any] = {
        f"{p}accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"{p}balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        f"{p}precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}specificity": round(float(tn / max(tn + fp, 1)), 6),
        f"{p}f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        f"{p}tp": int(tp),
        f"{p}tn": int(tn),
        f"{p}fp": int(fp),
        f"{p}fn": int(fn),
    }

    if y_proba is not None:
        try:
            metrics[f"{p}roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        except ValueError:
            metrics[f"{p}roc_auc"] = None
        try:
            metrics[f"{p}pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 6)
        except ValueError:
            metrics[f"{p}pr_auc"] = None
    else:
        metrics[f"{p}roc_auc"] = None
        metrics[f"{p}pr_auc"] = None

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Compute the full suite of multi-class classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 … n_classes-1).
    y_pred:
        Predicted labels.
    y_proba:
        Predicted probabilities (shape: [n_samples, n_classes]).
        If None, AUC is set to None.
    n_classes:
        Number of classes.
    prefix:
        String prefix for all metric keys.

    Returns
    -------
    dict of metric_name → value, JSON-serialisable.
    """
    p = prefix
    labels = list(range(n_classes))

    metrics: Dict[str, Any] = {
        f"{p}accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"{p}balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        f"{p}macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 6
        ),
        f"{p}macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 6
        ),
        f"{p}macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 6
        ),
        f"{p}weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)), 6
        ),
        f"{p}cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 6),
        f"{p}confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    # Per-class F1 and recall
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    for i, cls in enumerate(labels):
        metrics[f"{p}class{cls}_f1"] = round(float(per_class_f1[i]), 6)
        metrics[f"{p}class{cls}_recall"] = round(float(per_class_recall[i]), 6)
        metrics[f"{p}class{cls}_precision"] = round(float(per_class_precision[i]), 6)

    if y_proba is not None and y_proba.shape[1] == n_classes:
        try:
            metrics[f"{p}macro_roc_auc"] = round(
                float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")), 6
            )
        except ValueError as exc:
            metrics[f"{p}macro_roc_auc"] = None
            logger.debug("ROC-AUC computation skipped: %s", exc)
    else:
        metrics[f"{p}macro_roc_auc"] = None

    return metrics


def aggregate_cv_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold metric dicts into mean ± std summary.

    Parameters
    ----------
    fold_metrics:
        List of metric dicts, one per CV fold.

    Returns
    -------
    dict mapping metric_name → {"mean": ..., "std": ..., "min": ..., "max": ...}
    Only numeric (float/int) metrics are aggregated; non-numeric (e.g. confusion
    matrices) are excluded from the summary but preserved in the per-fold records.
    """
    if not fold_metrics:
        return {}

    all_keys = set()
    for fm in fold_metrics:
        all_keys.update(fm.keys())

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [fm[key] for fm in fold_metrics if key in fm and isinstance(fm[key], (int, float)) and fm[key] is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "n_folds": len(values),
        }
    return summary


# =============================================================================
# Feature selection (inside CV folds only)
# =============================================================================


class FoldFeatureSelector:
    """
    Mutual-information-based feature selector designed to be used inside
    a single CV fold's training data.

    Rules:
      * Fitted ONLY on fold training data (never val, never test, never UAE).
      * Applied (transform-only) to val fold and test set.
      * The selected feature list is recorded per fold.

    Parameters
    ----------
    k:
        Number of top features to select.
    random_state:
        Seed for mutual_info_classif (which uses random shuffling internally).
    """

    def __init__(self, k: int = 20, random_state: int = 42) -> None:
        self.k = k
        self.random_state = random_state
        self._selector: Optional[SelectKBest] = None
        self.selected_feature_names_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def fit(
        self,
        X_train_fold: pd.DataFrame,
        y_train_fold: np.ndarray,
        fold_num: int,
        dataset_name: str,
    ) -> "FoldFeatureSelector":
        """
        Fit the selector on the fold's training data.
        Validates that the fit size matches the fold size (leakage guard).
        """
        assert_feature_selector_not_fitted_on_val(
            selector_fit_size=len(X_train_fold),
            train_fold_size=len(X_train_fold),
            fold_num=fold_num,
            dataset_name=dataset_name,
        )

        k_actual = min(self.k, X_train_fold.shape[1])
        self._selector = SelectKBest(
            score_func=mutual_info_classif,
            k=k_actual,
        )
        self._selector.fit(X_train_fold.values, y_train_fold)

        support_mask = self._selector.get_support()
        all_cols = X_train_fold.columns.tolist()
        self.selected_feature_names_ = [c for c, s in zip(all_cols, support_mask) if s]

        scores = self._selector.scores_
        self.feature_scores_ = {
            col: round(float(scores[i]), 6) for i, col in enumerate(all_cols)
        }

        logger.debug(
            "[%s] Fold %d: selected %d / %d features via mutual info.",
            dataset_name, fold_num, len(self.selected_feature_names_), len(all_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the fitted feature columns from X (transform-only)."""
        if self._selector is None:
            raise ModelTrainingError("FoldFeatureSelector: call fit() before transform().")
        return X[self.selected_feature_names_]

    def fit_transform(
        self,
        X_train_fold: pd.DataFrame,
        y_train_fold: np.ndarray,
        fold_num: int,
        dataset_name: str,
    ) -> pd.DataFrame:
        self.fit(X_train_fold, y_train_fold, fold_num, dataset_name)
        return self.transform(X_train_fold)


def compute_union_features(
    per_fold_feature_lists: List[List[str]],
    all_available_features: List[str],
    dataset_name: str,
) -> List[str]:
    """
    Compute the union of features selected across all CV folds.
    Preserves the original column order from all_available_features.

    This union is used for the final model refit on the full training set,
    ensuring that features useful in ANY fold are retained.

    Parameters
    ----------
    per_fold_feature_lists:
        List of feature lists, one per CV fold.
    all_available_features:
        Full ordered feature list (determines output order).
    dataset_name:
        Used in log messages.

    Returns
    -------
    Ordered list of features in the union.
    """
    union = set()
    for fold_features in per_fold_feature_lists:
        union.update(fold_features)

    ordered = [f for f in all_available_features if f in union]
    logger.info(
        "[%s] Feature union across CV folds: %d features selected "
        "(from max %d available).",
        dataset_name, len(ordered), len(all_available_features),
    )
    return ordered


# =============================================================================
# Scaling (inside CV folds, only for models that require it)
# =============================================================================


class FoldScaler:
    """
    StandardScaler wrapper designed to be used inside a single CV fold.
    Fitted ONLY on fold training data; applied transform-only to val/test.
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None

    def fit_transform(self, X_train_fold: pd.DataFrame) -> pd.DataFrame:
        self._scaler = StandardScaler()
        return pd.DataFrame(
            self._scaler.fit_transform(X_train_fold),
            columns=X_train_fold.columns,
            index=X_train_fold.index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            raise ModelTrainingError("FoldScaler: call fit_transform() before transform().")
        return pd.DataFrame(
            self._scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )


# =============================================================================
# Model factory
# =============================================================================


def build_model(
    model_name: str,
    params: Dict[str, Any],
    task_type: str,
    y_train: Optional[np.ndarray] = None,
) -> Any:
    """
    Instantiate a model from its name and parameter dict.
    Handles XGBoost's scale_pos_weight and CatBoost's auto_class_weights
    dynamically from the training label distribution.

    Parameters
    ----------
    model_name:
        One of "LogisticRegression", "RandomForest", "XGBoost",
        "LightGBM", "CatBoost".
    params:
        Hyperparameter dict (from model_config.yaml).
    task_type:
        "binary" or "multiclass".
    y_train:
        Training labels. Required for XGBoost scale_pos_weight and
        CatBoost auto_class_weights computation.

    Returns
    -------
    Unfitted sklearn-compatible estimator.
    """
    p = dict(params)  # copy so we don't mutate the config

    if model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**p)

    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**p)

    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        # Remove our config-only keys that XGBoost doesn't accept
        p.pop("use_label_encoder", None)
        if task_type == "binary" and y_train is not None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            if n_pos > 0:
                p["scale_pos_weight"] = round(n_neg / n_pos, 4)
                logger.debug(
                    "[XGBoost] scale_pos_weight set to %.4f (n_neg=%d / n_pos=%d)",
                    p["scale_pos_weight"], n_neg, n_pos,
                )
        elif task_type == "multiclass":
            n_classes = len(np.unique(y_train)) if y_train is not None else 5
            p["num_class"] = n_classes
        return XGBClassifier(**p)

    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        if task_type == "multiclass":
            p.setdefault("objective", "multiclass")
            if y_train is not None:
                p["num_class"] = int(len(np.unique(y_train)))
        return LGBMClassifier(**p)

    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        # CatBoost uses auto_class_weights for imbalance
        if task_type == "binary":
            p["auto_class_weights"] = "Balanced"
        elif task_type == "multiclass":
            p["auto_class_weights"] = "Balanced"
            p.setdefault("loss_function", "MultiClass")
        return CatBoostClassifier(**p)

    else:
        raise ModelTrainingError(
            f"Unknown model_name '{model_name}'. "
            f"Valid: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost."
        )


# =============================================================================
# Feature importance extraction
# =============================================================================


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
) -> Dict[str, float]:
    """
    Extract model-native feature importance as a JSON-serialisable dict.

    Handles the different importance attribute names across sklearn,
    XGBoost, LightGBM, and CatBoost.

    Parameters
    ----------
    model:
        Fitted estimator.
    feature_names:
        List of feature names (in the same order the model was trained on).
    model_name:
        Model identifier for logging.

    Returns
    -------
    dict mapping feature_name → importance_score, sorted descending.
    Returns empty dict if importance is unavailable.
    """
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif model_name == "LogisticRegression" and hasattr(model, "coef_"):
        # Logistic Regression: use absolute coefficient values as proxy importance.
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        logger.warning(
            "[%s] No feature importance attribute found — returning empty dict.",
            model_name,
        )
        return {}

    if len(importances) != len(feature_names):
        logger.warning(
            "[%s] Importance array length (%d) != feature_names length (%d). "
            "Cannot map importances to names.",
            model_name, len(importances), len(feature_names),
        )
        return {}

    importance_dict = {
        name: round(float(imp), 8)
        for name, imp in zip(feature_names, importances)
    }
    return dict(sorted(importance_dict.items(), key=lambda x: -x[1]))


# =============================================================================
# Calibration
# =============================================================================
def calibrate_model(
    model,
    X_calib,
    y_calib,
    method="sigmoid",
 ):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    import numpy as np

    _, counts = np.unique(y_calib, return_counts=True)

    cv_folds = min(2, counts.min())

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
        cv=cv_folds
    )

    calibrated.fit(X_calib, y_calib)

    return calibrated


# =============================================================================
# Artifact I/O helpers
# =============================================================================


def save_json_artifact(data: Any, path: Path) -> None:
    """Save a JSON-serialisable object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_safe)


def save_joblib_artifact(obj: Any, path: Path) -> None:
    """Serialise any Python object to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def save_csv_artifact(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.Series):
        return _json_safe(obj.tolist())
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# =============================================================================
# Class weight computation
# =============================================================================


def compute_class_weights_dict(y: np.ndarray) -> Dict[int, float]:
    """
    Compute sklearn-style balanced class weights.
    Returns {class_label: weight} where weight = n_samples / (n_classes * n_class_i).
    Useful for models that accept class_weight as a dict (e.g. XGBoost with
    sample_weight argument).
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}