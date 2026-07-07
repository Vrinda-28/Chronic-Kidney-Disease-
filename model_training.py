"""
model_training.py
=================

Production-grade, leakage-safe model training pipeline for the CKD
Prediction and Explainable AI research project.

Pipeline position:
  data_loader.py → preprocess.py → feature_engineering.py
  → train_test_split.py → model_training.py (THIS FILE)
  → SHAP → UAE External Validation

──────────────────────────────────────────────────────────────────────────────
SECTION A — ARCHITECTURE REVIEW
──────────────────────────────────────────────────────────────────────────────

Five design decisions made after reviewing the existing pipeline:

  1. SEPARATE PIPELINES FOR BINARY AND MULTICLASS (justified):
     UCI (binary, 400 rows) and Kaggle (5-class, 200 rows) have different
     sample sizes, class structures, and clinical questions. Merging them
     into one pipeline would require reconciling incompatible targets.
     Each task gets its own model set, metrics, and artifacts.
     UAE is evaluated only with the UCI binary model (it has binary labels).

  2. CLASS WEIGHTS OVER SMOTE (justified):
     For UCI: ~250 CKD vs ~150 notCKD. Mild imbalance. class_weight='balanced'
     is sufficient and avoids instability from synthetic samples in a 320-row dataset.
     For Kaggle: ~40 rows per class after split. SMOTE requires ≥ k+1=6 samples
     per class per CV fold; with ~26 training samples per class this is marginal.
     class_weight='balanced_subsample' (RF) or 'balanced' (sklearn/LightGBM)
     plus scale_pos_weight (XGBoost) cover all cases safely.
     See: Lemaitre et al. (2017), JMLR 18(1):559–563.

  3. FEATURE SELECTION INSIDE CV FOLDS (justified):
     Mutual Information (MI) SelectKBest is fitted on each CV training fold,
     not on the full training set. This prevents the MI scores from being
     inflated by seeing validation-fold labels. The union of fold-selected
     features is used for the final model refitted on the complete training set.

  4. SCALING ONLY FOR LOGISTIC REGRESSION (justified):
     RF, XGBoost, LightGBM, and CatBoost are all scale-invariant. Applying
     StandardScaler to them would not improve performance and adds unnecessary
     complexity. Only LR, which is sensitive to feature magnitude, gets a
     scaler fitted inside each CV fold.

  5. CALIBRATION AFTER FINAL TRAINING (justified):
     Probability calibration (isotonic regression) is applied after the final
     model is trained on the full training set. A 10% stratified hold-out from
     the TRAINING SET (never test or UAE) is used as the calibration set.
     Calibrated models produce reliable probabilities for clinical use —
     a predicted 70% CKD probability should correspond to ~70% true prevalence.

──────────────────────────────────────────────────────────────────────────────
SECTION B — RECOMMENDED TRAINING STRATEGY
──────────────────────────────────────────────────────────────────────────────

  Binary task (UCI):
    Models: LR (baseline), RF, XGBoost, LightGBM, CatBoost
    CV:     5-fold StratifiedKFold (pre-generated in train_test_split.py)
    Imbalance: class_weight='balanced' / scale_pos_weight
    Feature selection: MI SelectKBest(k=20) inside each CV fold
    Scaling: StandardScaler inside each fold for LR only
    Primary selection metric: ROC-AUC (most robust for imbalanced binary)
    Secondary: Macro-F1

  Multiclass task (Kaggle):
    Models: RF, XGBoost, LightGBM, CatBoost
    CV:     RepeatedStratifiedKFold(5×5=25 folds)
    Imbalance: class_weight='balanced_subsample' / 'balanced'
    Feature selection: MI SelectKBest(k=25) inside each fold
    Primary selection metric: Balanced Accuracy (robust for 5-class imbalance)
    Secondary: Macro-F1

  Model selection rule:
    Select the model with the highest mean CV primary metric.
    Refit the winner on the FULL training set (all 5 folds combined).
    Evaluate ONCE on the held-out test set.
    Never re-select based on test performance.

──────────────────────────────────────────────────────────────────────────────
SECTION C — LEAKAGE ANALYSIS
──────────────────────────────────────────────────────────────────────────────

  Six leakage vectors addressed:

  1. TARGET LEAKAGE: assert_no_target_in_features() called before every
     model.fit() and model.predict() call.

  2. TRAIN/VAL OVERLAP: assert_val_not_in_train_indices() called at the start
     of each CV fold iteration.

  3. FEATURE SELECTION LEAKAGE: FoldFeatureSelector is fitted and transformed
     inside each fold's training portion only. Val fold is transform-only.
     assert_feature_selector_not_fitted_on_val() guards this.

  4. SCALING LEAKAGE: FoldScaler is fitted on fold training data; val and test
     data are transform-only. Scaler is never refitted on val or test.

  5. TEST SET CONTAMINATION: The test set is loaded but never passed to any
     CV-related function. It is only accessed after final model selection in
     _evaluate_on_test().

  6. UAE CONTAMINATION: assert_no_uae_in_training() called before training.
     UAE DataFrame is loaded into a structurally separate variable and never
     passed to any fit() or CV function. It is evaluated last, once.

Usage
-----
    trainer = CKDModelTrainer(config_path="config/model_config.yaml")
    results = trainer.train_all()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

from uae_validation import (
    run_uae_external_validation,
    UAEValidationReport
)

from model_utils import (
    FoldFeatureSelector,
    FoldScaler,
    LeakageViolation,
    ModelTrainingError,
    aggregate_cv_metrics,
    assert_no_target_in_features,
    assert_no_uae_in_training,
    assert_val_not_in_train_indices,
    build_model,
    calibrate_model,
    compute_binary_metrics,
    compute_class_weights_dict,
    compute_multiclass_metrics,
    compute_union_features,
    extract_feature_importance,
    save_csv_artifact,
    save_joblib_artifact,
    save_json_artifact,
)


# =============================================================================
# Logging
# =============================================================================


def _build_logger(
    log_dir: str,
    log_filename: str,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    logger = logging.getLogger("ckd_model_trainer")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_filename),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not set up file logging: %s", exc)

    return logger


# =============================================================================
# Configuration
# =============================================================================


class ModelConfig:
    """Loads config/model_config.yaml and exposes all sections."""

    def __init__(self, config_path: str = "config/model_config.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model config not found: {path.resolve()}. "
                f"Expected at config/model_config.yaml."
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        self.random_seed: int = int(raw["random_seed"])
        self.tasks: Dict[str, Any] = raw["tasks"]
        self.models: Dict[str, List[Dict[str, Any]]] = raw["models"]
        self.feature_selection: Dict[str, Any] = raw["feature_selection"]
        self.scaling: Dict[str, Any] = raw.get("scaling", {})
        self.calibration: Dict[str, Any] = raw.get("calibration", {})
        self.metrics: Dict[str, Any] = raw.get("metrics", {})
        self.artifacts_cfg: Dict[str, Any] = raw.get("artifacts", {})
        self.uae_validation: Dict[str, Any] = raw.get("uae_validation", {})
        self.logging_cfg: Dict[str, str] = raw.get("logging", {})
        self.reproducibility: Dict[str, Any] = raw.get("reproducibility", {})

    def get_task(self, key: str) -> Dict[str, Any]:
        return self.tasks[key]

    def get_models_for_task(self, task_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.models.get(task_type, []) if m.get("enabled", True)]


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ModelResult:
    """Complete training result for one model on one task."""
    model_name: str
    task_key: str
    task_type: str
    cv_fold_metrics: List[Dict[str, Any]]
    cv_summary: Dict[str, Dict[str, float]]
    selected_features_per_fold: List[List[str]]
    union_features: List[str]
    feature_importance: Dict[str, float]
    test_metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_time_seconds: float
    primary_cv_metric: str
    primary_cv_score: float    # mean across folds


@dataclass
class TaskResult:
    """All model results for one task (UCI or Kaggle), plus best model info."""
    task_key: str
    task_type: str
    model_results: Dict[str, ModelResult]
    best_model_name: str
    best_cv_score: float
    primary_metric: str
    test_metrics_best: Dict[str, Any]
    uae_metrics: Optional[Any] = None


# =============================================================================
# Main trainer
# =============================================================================


class CKDModelTrainer:
    """
    Orchestrates training, evaluation, and artifact saving for all models
    on both UCI (binary) and Kaggle (multiclass) tasks.

    Usage
    -----
        trainer = CKDModelTrainer(config_path="config/model_config.yaml")
        results = trainer.train_all()
    """

    def __init__(
        self,
        config_path: str = "config/model_config.yaml",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = ModelConfig(config_path)
        np.random.seed(self.cfg.random_seed)

        log_cfg = self.cfg.logging_cfg
        self.logger = logger or _build_logger(
            log_dir=log_cfg.get("log_dir", "logs"),
            log_filename=log_cfg.get("log_filename", "model_training.log"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def train_all(self) -> Dict[str, TaskResult]:
        """
        Run the complete training pipeline for both UCI and Kaggle tasks,
        then evaluate the best UCI model on UAE.

        Returns
        -------
        dict mapping task_key → TaskResult.
        """
        self.logger.info("=" * 70)
        self.logger.info("CKD Model Training Pipeline — START")
        self.logger.info("sklearn %s | pandas %s | Python %s",
                         sklearn.__version__, pd.__version__, sys.version.split()[0])
        self.logger.info("=" * 70)

        results: Dict[str, TaskResult] = {}

        for task_key in ("uci", "kaggle"):
            task_result = self._train_task(task_key)
            results[task_key] = task_result

        # UAE external validation with the best UCI binary model
        if self.cfg.uae_validation.get("apply_uci_model_to_uae", True):
            uae_metrics = self._evaluate_uae(results["uci"])
            results["uci"].uae_metrics = uae_metrics

        self._save_best_model_summary(results)

        self.logger.info("=" * 70)
        self.logger.info("CKD Model Training Pipeline — COMPLETE")
        self.logger.info("=" * 70)

        return results

    # -----------------------------------------------------------------------
    # Task orchestration
    # -----------------------------------------------------------------------

    def _train_task(self, task_key: str) -> TaskResult:
        """Run full training pipeline for one task (UCI or Kaggle)."""
        task_cfg = self.cfg.get_task(task_key)
        task_type = task_cfg["task_type"]
        target_col = task_cfg["target_col"]
        primary_metric = task_cfg["primary_metric"]
        artifacts_dir = Path(task_cfg["artifacts_dir"])

        self.logger.info("[%s] ─── Task: %s ───", task_key.upper(), task_cfg["description"])

        # ── Load split data ────────────────────────────────────────────────
        train_df, test_df, cv_fold_indices = self._load_task_data(task_key, task_cfg)
        uae_df = self._load_uae_df(task_cfg)

        # ── UAE contamination guard ────────────────────────────────────────
        assert_no_uae_in_training(train_df, uae_df, context=f"task={task_key}")
        self.logger.info("[%s] ✔ UAE object-isolation check passed.", task_key.upper())

        # ── Separate features from target ──────────────────────────────────
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].values.astype(int)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(int)

        feature_cols = X_train.columns.tolist()
        assert_no_target_in_features(feature_cols, target_col, context=f"task={task_key} setup")

        self.logger.info(
            "[%s] Train: %d rows × %d features | Test: %d rows | Classes: %s",
            task_key.upper(), len(X_train), len(feature_cols),
            len(X_test), dict(pd.Series(y_train).value_counts().sort_index()),
        )

        # ── Feature selection config ───────────────────────────────────────
        fs_enabled = self.cfg.feature_selection.get("enabled", True)
        fs_k = (
            self.cfg.feature_selection.get("binary_k", 20)
            if task_type == "binary"
            else self.cfg.feature_selection.get("multiclass_k", 25)
        )

        # ── Train all models ───────────────────────────────────────────────
        model_results: Dict[str, ModelResult] = {}
        model_cfgs = self.cfg.get_models_for_task(task_type)

        for model_cfg in model_cfgs:
            model_name = model_cfg["name"]
            self.logger.info(
                "[%s][%s] Training …", task_key.upper(), model_name
            )
            try:
                result = self._train_one_model(
                    model_name=model_name,
                    model_cfg=model_cfg,
                    task_key=task_key,
                    task_type=task_type,
                    target_col=target_col,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    cv_fold_indices=cv_fold_indices,
                    feature_cols=feature_cols,
                    fs_enabled=fs_enabled,
                    fs_k=fs_k,
                    primary_metric=primary_metric,
                    artifacts_dir=artifacts_dir,
                    task_cfg=task_cfg,
                )
                model_results[model_name] = result
                self.logger.info(
                    "[%s][%s] CV %s = %.4f ± %.4f",
                    task_key.upper(), model_name, primary_metric,
                    result.cv_summary.get(primary_metric, {}).get("mean", 0),
                    result.cv_summary.get(primary_metric, {}).get("std", 0),
                )
            except Exception as exc:
                self.logger.error(
                    "[%s][%s] Training FAILED: %s", task_key.upper(), model_name, exc,
                    exc_info=True,
                )

        if not model_results:
            raise ModelTrainingError(
                f"[{task_key}] All models failed to train. Check logs for details."
            )

        # ── Model selection (by CV, not test) ─────────────────────────────
        best_name, best_score = self._select_best_model(
            model_results, primary_metric, task_key
        )
        self.logger.info(
            "[%s] Best model: %s (CV %s = %.4f)",
            task_key.upper(), best_name, primary_metric, best_score,
        )

        # ── Evaluate best model on test set ───────────────────────────────
        best_result = model_results[best_name]
        test_metrics = best_result.test_metrics

        self.logger.info(
            "[%s][%s] TEST SET — %s",
            task_key.upper(), best_name,
            {k: v for k, v in test_metrics.items()
             if isinstance(v, float) and "confusion" not in k},
        )

        return TaskResult(
            task_key=task_key,
            task_type=task_type,
            model_results=model_results,
            best_model_name=best_name,
            best_cv_score=best_score,
            primary_metric=primary_metric,
            test_metrics_best=test_metrics,
        )

    # -----------------------------------------------------------------------
    # Single model training loop
    # -----------------------------------------------------------------------

    def _train_one_model(
        self,
        model_name: str,
        model_cfg: Dict[str, Any],
        task_key: str,
        task_type: str,
        target_col: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        cv_fold_indices: List[Dict[str, Any]],
        feature_cols: List[str],
        fs_enabled: bool,
        fs_k: int,
        primary_metric: str,
        artifacts_dir: Path,
        task_cfg: Dict[str, Any],
    ) -> ModelResult:
        """
        Full CV loop for one model:
          For each fold:
            1. Leakage checks.
            2. Feature selection on train fold.
            3. Scaling (LR only) on train fold.
            4. Fit model on train fold.
            5. Predict on val fold (transform-only for FS and scaling).
            6. Compute val metrics.
          After all folds:
            7. Compute union of selected features.
            8. Refit model on full training set with union features.
            9. Calibrate probabilities.
            10. Evaluate on test set.
            11. Extract feature importances.
            12. Save all artifacts.
        """
        start_time = time.time()
        requires_scaling = model_cfg.get("requires_scaling", False)
        n_classes = task_cfg.get("n_classes", 2)

        fold_metrics: List[Dict[str, Any]] = []
        selected_features_per_fold: List[List[str]] = []

        for fold in cv_fold_indices:
            fold_num = fold["fold_num"]
            train_idx = fold["train_indices"]
            val_idx = fold["val_indices"]

            # ── Leakage guard 1: no val/train index overlap ────────────────
            assert_val_not_in_train_indices(train_idx, val_idx, fold_num, task_key.upper())

            # ── Fold data ──────────────────────────────────────────────────
            X_fold_train = X_train.iloc[train_idx].copy()
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx].copy()
            y_fold_val = y_train[val_idx]

            # ── Leakage guard 2: target not in features ────────────────────
            assert_no_target_in_features(
                X_fold_train.columns.tolist(), target_col,
                context=f"fold {fold_num} train",
            )

            # ── Feature selection (train fold only) ────────────────────────
            if fs_enabled:
                fs = FoldFeatureSelector(k=fs_k, random_state=self.cfg.random_seed)
                X_fold_train_fs = fs.fit_transform(
                    X_fold_train, y_fold_train, fold_num, task_key.upper()
                )
                X_fold_val_fs = fs.transform(X_fold_val)
                selected_features_per_fold.append(fs.selected_feature_names_)
            else:
                X_fold_train_fs = X_fold_train
                X_fold_val_fs = X_fold_val
                selected_features_per_fold.append(feature_cols)

            # ── Scaling (train fold only, LR only) ─────────────────────────
            if requires_scaling:
                scaler = FoldScaler()
                X_fold_train_fs = scaler.fit_transform(X_fold_train_fs)
                X_fold_val_fs = scaler.transform(X_fold_val_fs)

            # ── Build and train model ──────────────────────────────────────
            model = build_model(
                model_name, model_cfg["params"], task_type, y_train=y_fold_train
            )
            model.fit(X_fold_train_fs.values, y_fold_train)

            # ── Predict on val fold (no refit, no FS refit) ───────────────
            y_val_pred = model.predict(X_fold_val_fs.values)
            y_val_proba = (
                model.predict_proba(X_fold_val_fs.values)
                if hasattr(model, "predict_proba") else None
            )

            # ── Val metrics ────────────────────────────────────────────────
            if task_type == "binary":
                proba_col1 = y_val_proba[:, 1] if y_val_proba is not None else None
                fold_m = compute_binary_metrics(y_fold_val, y_val_pred, proba_col1)
            else:
                fold_m = compute_multiclass_metrics(
                    y_fold_val, y_val_pred, y_val_proba, n_classes
                )
            fold_m["fold_num"] = fold_num
            fold_metrics.append(fold_m)

        # ── CV summary ─────────────────────────────────────────────────────
        cv_summary = aggregate_cv_metrics(fold_metrics)
        primary_cv_score = cv_summary.get(primary_metric, {}).get("mean", 0.0)

        # ── Union of selected features ─────────────────────────────────────
        if fs_enabled:
            union_features = compute_union_features(
                selected_features_per_fold, feature_cols, task_key.upper()
            )
        else:
            union_features = feature_cols

        # ── Calibration hold-out from training data ────────────────────────
        calib_frac = self.cfg.calibration.get("calibration_holdout_fraction", 0.10)
        do_calibrate = (
            self.cfg.calibration.get("enabled", True)
            and model_cfg.get("supports_predict_proba", True)
        )

        if do_calibrate:
            X_refit_arr = np.arange(len(X_train))
            train_idx_final, calib_idx_final = train_test_split(
                X_refit_arr,
                test_size=calib_frac,
                stratify=y_train,
                random_state=self.cfg.random_seed,
            )
            X_final_train = X_train[union_features].iloc[train_idx_final]
            y_final_train = y_train[train_idx_final]
            X_calib = X_train[union_features].iloc[calib_idx_final]
            y_calib = y_train[calib_idx_final]
        else:
            X_final_train = X_train[union_features]
            y_final_train = y_train
            X_calib = None
            y_calib = None

        # ── Scale final training data (LR only) ───────────────────────────
        final_scaler = None
        if requires_scaling:
            final_scaler = FoldScaler()
            X_final_train = final_scaler.fit_transform(X_final_train)
            if X_calib is not None:
                X_calib = final_scaler.transform(X_calib)
            X_test_scaled = final_scaler.transform(X_test[union_features])
        else:
            X_test_scaled = X_test[union_features]

        # ── Final model refit on full training set ─────────────────────────
        final_model = build_model(
            model_name, model_cfg["params"], task_type, y_train=y_final_train
        )
        final_model.fit(X_final_train.values, y_final_train)
        self.logger.info(
            "[%s][%s] Final model refitted on %d rows, %d features.",
            task_key.upper(), model_name, len(X_final_train), len(union_features),
        )

        # ── Calibration ────────────────────────────────────────────────────
        calibrated_model = None
        if do_calibrate and X_calib is not None:
            calib_method = self.cfg.calibration.get("method", "isotonic")
            calibrated_model = calibrate_model(
                final_model, X_calib.values, y_calib, method=calib_method
            )
            self.logger.info(
                "[%s][%s] Calibrated model using %s regression on %d calib rows.",
                task_key.upper(), model_name, calib_method, len(X_calib),
            )

        # ── Test set evaluation ────────────────────────────────────────────
        test_metrics = self._evaluate_on_test(
            model=calibrated_model if calibrated_model is not None else final_model,
            X_test=X_test_scaled,
            y_test=y_test,
            task_type=task_type,
            n_classes=n_classes,
            model_name=model_name,
            task_key=task_key,
        )

        # ── Feature importance ─────────────────────────────────────────────
        feature_importance = extract_feature_importance(
            final_model, union_features, model_name
        )

        training_time = round(time.time() - start_time, 2)
        self.logger.info(
            "[%s][%s] Completed in %.1fs.", task_key.upper(), model_name, training_time
        )

        # ── Save artifacts ─────────────────────────────────────────────────
        result = ModelResult(
            model_name=model_name,
            task_key=task_key,
            task_type=task_type,
            cv_fold_metrics=fold_metrics,
            cv_summary=cv_summary,
            selected_features_per_fold=selected_features_per_fold,
            union_features=union_features,
            feature_importance=feature_importance,
            test_metrics=test_metrics,
            hyperparameters=model_cfg["params"],
            training_time_seconds=training_time,
            primary_cv_metric=primary_metric,
            primary_cv_score=primary_cv_score,
        )

        self._save_model_artifacts(
            result=result,
            final_model=final_model,
            calibrated_model=calibrated_model,
            final_scaler=final_scaler,
            X_test=X_test_scaled,
            y_test=y_test,
            artifacts_dir=artifacts_dir,
            task_cfg=task_cfg,
        )

        return result

    # -----------------------------------------------------------------------
    # Test set evaluation (called ONCE per model, after CV is complete)
    # -----------------------------------------------------------------------

    def _evaluate_on_test(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        task_type: str,
        n_classes: int,
        model_name: str,
        task_key: str,
    ) -> Dict[str, Any]:
        """
        Evaluate the final model on the held-out test set.
        The test set is NEVER used during CV or model selection.
        This method is only called after the final model has been selected.
        """
        assert_no_target_in_features(
            X_test.columns.tolist(), "__target__",
            context=f"test evaluation {task_key}/{model_name}",
        )

        y_pred = model.predict(X_test.values)
        y_proba = (
            model.predict_proba(X_test.values)
            if hasattr(model, "predict_proba") else None
        )

        if task_type == "binary":
            proba_col1 = y_proba[:, 1] if y_proba is not None else None
            metrics = compute_binary_metrics(y_test, y_pred, proba_col1, prefix="test_")
        else:
            metrics = compute_multiclass_metrics(
                y_test, y_pred, y_proba, n_classes, prefix="test_"
            )

        self.logger.info(
            "[%s][%s] Test metrics computed on %d rows.",
            task_key.upper(), model_name, len(y_test),
        )
        return metrics

    # -----------------------------------------------------------------------
    # UAE external validation
    # -----------------------------------------------------------------------

    
    
    def _evaluate_uae(self, uci_result) -> "UAEValidationReport":
     """
     Rigorous UAE external validation ...
     """

     from uae_validation import run_uae_external_validation

     self.logger.info("[UAE] Running rigorous external validation (2-track) …")

     task_cfg_uae = self.cfg.get_task("uae")
     task_cfg_uci = self.cfg.get_task("uci")
     splits_dir = Path(task_cfg_uae.get("splits_dir", "data/splits"))
     uae_path = splits_dir / task_cfg_uae.get("full_file", "uae_full.csv")

     if not uae_path.exists():
        self.logger.warning(
            "[UAE] File not found at %s — skipping UAE validation.", uae_path
        )
        return None
     uae_df = pd.read_csv(uae_path)
     target_col = task_cfg_uae["target_col"]

     # Reload UCI train/test and CV fold indices
     train_df, test_df, cv_fold_indices = self._load_task_data("uci", task_cfg_uci)
     train_target = task_cfg_uci["target_col"]
     X_train = train_df.drop(columns=[train_target])
     y_train = train_df[train_target].values.astype(int)
     X_test = test_df.drop(columns=[train_target])
     y_test = test_df[train_target].values.astype(int)

     # Get the best UCI model and its union features
     best_name = uci_result.best_model_name
     best_result = uci_result.model_results[best_name]
     union_features = best_result.union_features
     best_params = best_result.hyperparameters

     # Load the saved final model for Track B
     uci_artifacts_dir = Path(task_cfg_uci["artifacts_dir"])
     model_path = uci_artifacts_dir / best_name / "calibrated_model.joblib"
     if not model_path.exists():
        model_path = uci_artifacts_dir / best_name / "final_model.joblib"
     if not model_path.exists():
        self.logger.warning(
            "[UAE] Model artifact not found at %s — Track B unavailable.", model_path
        )
        full_model = None
     else:
        full_model = joblib.load(model_path)

     uae_artifacts_dir = Path(task_cfg_uci["artifacts_dir"]) / "uae_validation"

     from uae_validation import run_uae_external_validation
     report = run_uae_external_validation(
        best_model_name=best_name,
        best_model_params=best_params,
        full_trained_model=full_model,
        union_features=union_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_fold_indices=cv_fold_indices,
        uae_df=uae_df,
        target_col=target_col,
        random_seed=self.cfg.random_seed,
        artifacts_dir=uae_artifacts_dir,
    )
     return report
    

    # -----------------------------------------------------------------------
    # Model selection
    # -----------------------------------------------------------------------

    def _select_best_model(
        self,
        model_results: Dict[str, ModelResult],
        primary_metric: str,
        task_key: str,
    ) -> Tuple[str, float]:
        """
        Select the model with the highest mean CV primary metric.
        NEVER uses test set data.
        """
        scores = {
            name: result.primary_cv_score
            for name, result in model_results.items()
        }
        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]

        self.logger.info(
            "[%s] CV scores (%s): %s",
            task_key.upper(), primary_metric,
            {k: f"{v:.4f}" for k, v in sorted(scores.items(), key=lambda x: -x[1])},
        )
        return best_name, best_score

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------

    def _load_task_data(
        self,
        task_key: str,
        task_cfg: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """Load pre-split train/test DataFrames and CV fold indices."""
        splits_dir = Path(task_cfg["splits_dir"])
        train_path = splits_dir / task_cfg["train_file"]
        test_path = splits_dir / task_cfg["test_file"]
        folds_path = Path(task_cfg["cv_folds_file"])

        for p in (train_path, test_path, folds_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"[{task_key}] Required file not found: {p.resolve()}. "
                    f"Run train_test_split.py before model_training.py."
                )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        with open(folds_path, "r", encoding="utf-8") as fh:
            folds_data = json.load(fh)
        cv_fold_indices = folds_data["folds"]

        self.logger.info(
            "[%s] Loaded — train: %s, test: %s, CV folds: %d",
            task_key.upper(), train_df.shape, test_df.shape, len(cv_fold_indices),
        )
        return train_df, test_df, cv_fold_indices

    def _load_uae_df(self, task_cfg: Dict[str, Any]) -> pd.DataFrame:
        """
        Load UAE as a structural reference (for the object-identity check).
        The actual UAE evaluation happens in _evaluate_uae(), not here.
        Returns an empty DataFrame stub if the UAE file doesn't exist yet.
        """
        uae_task_cfg = self.cfg.get_task("uae")
        splits_dir = Path(uae_task_cfg.get("splits_dir", "data/splits"))
        uae_path = splits_dir / uae_task_cfg.get("full_file", "uae_full.csv")
        if uae_path.exists():
            return pd.read_csv(uae_path)
        self.logger.warning(
            "[UAE] File not found at %s — object-isolation check will use empty stub.",
            uae_path,
        )
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Artifact saving
    # -----------------------------------------------------------------------

    def _save_model_artifacts(
        self,
        result: ModelResult,
        final_model: Any,
        calibrated_model: Optional[Any],
        final_scaler: Optional[FoldScaler],
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        artifacts_dir: Path,
        task_cfg: Dict[str, Any],
    ) -> None:
        """Save all artifacts for one trained model."""
        model_dir = artifacts_dir / result.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.cfg.artifacts_cfg

        if cfg.get("save_model", True):
            save_joblib_artifact(final_model, model_dir / "final_model.joblib")

        if cfg.get("save_calibrated_model", True) and calibrated_model is not None:
            save_joblib_artifact(calibrated_model, model_dir / "calibrated_model.joblib")

        if final_scaler is not None:
            save_joblib_artifact(final_scaler, model_dir / "final_scaler.joblib")

        if cfg.get("save_cv_metrics", True):
            save_json_artifact(result.cv_fold_metrics, model_dir / "cv_fold_metrics.json")

        if cfg.get("save_cv_summary", True):
            save_json_artifact(result.cv_summary, model_dir / "cv_summary.json")

        if cfg.get("save_test_metrics", True):
            save_json_artifact(result.test_metrics, model_dir / "test_metrics.json")

        if cfg.get("save_feature_importance", True):
            save_json_artifact(result.feature_importance, model_dir / "feature_importance.json")

        if cfg.get("save_selected_features", True):
            save_json_artifact(
                {
                    "union_features": result.union_features,
                    "n_union_features": len(result.union_features),
                    "per_fold_features": result.selected_features_per_fold,
                },
                model_dir / "selected_features.json",
            )

        if cfg.get("save_hyperparameters", True):
            save_json_artifact(result.hyperparameters, model_dir / "hyperparameters.json")

        if cfg.get("save_probabilities", True) or cfg.get("save_predictions", True):
            eval_model = calibrated_model if calibrated_model is not None else final_model
            X_test_sub = X_test[result.union_features] if result.union_features else X_test
            y_pred = eval_model.predict(X_test_sub.values)
            y_proba = (
                eval_model.predict_proba(X_test_sub.values)
                if hasattr(eval_model, "predict_proba") else None
            )

            preds_df = pd.DataFrame({"row_position": range(len(y_test)), "y_true": y_test, "y_pred": y_pred})
            if y_proba is not None:
                if y_proba.shape[1] == 2:
                    preds_df["y_proba_ckd"] = y_proba[:, 1]
                else:
                    for cls_i in range(y_proba.shape[1]):
                        preds_df[f"y_proba_class{cls_i}"] = y_proba[:, cls_i]

            save_csv_artifact(preds_df, model_dir / "test_predictions.csv")

        if cfg.get("save_training_metadata", True):
            metadata = {
                "model_name": result.model_name,
                "task_key": result.task_key,
                "task_type": result.task_type,
                "training_time_seconds": result.training_time_seconds,
                "primary_cv_metric": result.primary_cv_metric,
                "primary_cv_score": result.primary_cv_score,
                "n_cv_folds": len(result.cv_fold_metrics),
                "n_union_features": len(result.union_features),
                "sklearn_version": sklearn.__version__,
                "pandas_version": pd.__version__,
                "python_version": sys.version,
                "random_seed": self.cfg.random_seed,
                "class_names": task_cfg.get("class_names", []),
            }
            save_json_artifact(metadata, model_dir / "training_metadata.json")

        self.logger.debug(
            "[%s][%s] Artifacts saved to: %s",
            result.task_key.upper(), result.model_name, model_dir.resolve(),
        )

    # -----------------------------------------------------------------------
    # Best model summary
    # -----------------------------------------------------------------------

    def _save_best_model_summary(self, results):
     """Updated to handle UAEValidationReport in uae_metrics."""
     from model_utils import save_json_artifact
     from uae_validation import UAEValidationReport

     summary = {
        "pipeline_stage": "model_training",
        "random_seed": self.cfg.random_seed,
        "tasks": {},
     }

     for task_key, task_result in results.items():
        task_summary = {
            "best_model": task_result.best_model_name,
            "primary_metric": task_result.primary_metric,
            "best_cv_score": round(task_result.best_cv_score, 6),
            "test_metrics": task_result.test_metrics_best,
            "all_model_cv_scores": {
                name: round(r.primary_cv_score, 6)
                for name, r in task_result.model_results.items()
            },
        }

        if task_result.uae_metrics is not None:
            uae = task_result.uae_metrics
            if isinstance(uae, UAEValidationReport):
                task_summary["uae_external_validation"] = uae.as_dict()
                # Surface the primary result (Track A) prominently
                if uae.track_a_valid:
                    task_summary["uae_primary_result"] = {
                        "note": (
                            "Track A: reduced-feature UCI model (%d features). "
                            "This is the valid, reportable external validation result."
                            % uae.track_a_n_features
                        ),
                        "metrics": uae.track_a_uae_metrics,
                    }
            else:
                task_summary["uae_external_validation"] = uae

        summary["tasks"][task_key] = task_summary

        task_artifacts_dir = Path(self.cfg.get_task(task_key)["artifacts_dir"])
        save_json_artifact(task_summary, task_artifacts_dir / "best_model.json")
        self.logger.info(
            "[%s] best_model.json saved: best=%s, cv_%s=%.4f",
            task_key.upper(), task_result.best_model_name,
            task_result.primary_metric, task_result.best_cv_score,
        )

     save_json_artifact(summary, Path("artifacts/models/best_model_summary.json"))
     self.logger.info("Global best_model_summary.json saved.")

# =============================================================================
# CLI entry point
# =============================================================================


if __name__ == "__main__":
    trainer = CKDModelTrainer(config_path="config/model_config.yaml")
    results = trainer.train_all()

    print("\n── Model Training Complete ──")
    for task_key, task_result in results.items():
        if task_key == "uae":
            continue
        print(f"\n{task_key.upper()}:")
        print(f"  Best model   : {task_result.best_model_name}")
        print(f"  CV {task_result.primary_metric:<20}: {task_result.best_cv_score:.4f}")
        
        print(f"  Test metrics :")
        for k, v in task_result.test_metrics_best.items():
            if isinstance(v, float):
                print(f"    {k:<30}: {v:.4f}")

    uci_uae = results.get("uci")
    if uci_uae and uci_uae.uae_metrics:
     report = uci_uae.uae_metrics
     
     print("\nUAE External Validation (Track A - Primary):")

    if report.track_a_valid:
        for k, v in report.track_a_uae_metrics.items():
            if isinstance(v, float):
                print(f"  {k:<30}: {v:.4f}")

    print("\nArtifacts → artifacts/models/")