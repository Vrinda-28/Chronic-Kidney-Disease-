"""
evaluate.py
===========

Final evaluation stage for the CKD Machine Learning Pipeline.

Pipeline position:
  data_loader.py → preprocess.py → feature_engineering.py
  → train_test_split.py → model_training.py
  → evaluate.py (THIS FILE)       ← you are here
  → external_validation.py

══════════════════════════════════════════════════════════════════════════
SECTION A — ARCHITECTURE REVIEW
══════════════════════════════════════════════════════════════════════════

What this script does NOT do
─────────────────────────────
  ✗ No model training — models are loaded from saved .joblib artifacts
  ✗ No feature selection — uses the union_features saved per model
  ✗ No calibration fitting — loads calibrated_model.joblib
  ✗ No hyperparameter tuning
  ✗ No modifications to saved model artifacts

What this script DOES
──────────────────────
  ✓ Loads saved calibrated model (or final_model as fallback)
  ✓ Loads saved test set (never seen during training)
  ✓ Generates publication-quality metrics with 95% bootstrap CIs
  ✓ Generates: ROC curve, PR curve, confusion matrix, calibration curve,
    threshold sweep, SHAP summary, SHAP bar, feature importance
  ✓ Writes all outputs to artifacts/evaluation/{uci,kaggle}/

══════════════════════════════════════════════════════════════════════════
SECTION B — LEAKAGE REVIEW
══════════════════════════════════════════════════════════════════════════

  1. TRAIN/TEST ISOLATION: This script only reads test_file from the
     saved splits directory. It never reads the training set.
     (Training set is used only to recover feature names for SHAP context.)

  2. MODEL LOADING: We load `calibrated_model.joblib` which was fitted
     on a held-out calibration set from the training data only.
     Evaluating it on the test set is the correct, leakage-free procedure.

  3. SHAP: SHAP is a post-hoc explanation method. Computing SHAP values on
     the test set does NOT influence model parameters. No leakage.

  4. UAE ISOLATION: This script never loads or references UAE data.
     UAE evaluation is entirely handled by external_validation.py.

══════════════════════════════════════════════════════════════════════════

Usage
-----
    python evaluate.py
    python evaluate.py --config config/evaluation_config.yaml
    python evaluate.py --tasks uci           # evaluate only UCI
    python evaluate.py --tasks kaggle        # evaluate only Kaggle
    python evaluate.py --tasks uci kaggle    # both (default)
    python evaluate.py --no-shap             # skip SHAP (faster)
    python evaluate.py --no-ci               # skip bootstrap CI (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from evaluation_utils import (
    compute_binary_metrics_with_ci,
    compute_ece,
    compute_f1_threshold,
    compute_mcc_threshold,
    compute_multiclass_metrics_full,
    compute_youden_threshold,
    generate_summary_report,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_confusion_matrix_multiclass,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_roc_ovr,
    plot_shap_bar,
    plot_shap_summary,
    plot_threshold_sweep,
    save_json,
    threshold_sweep,
    compute_shap_values,
)
from sklearn.metrics import brier_score_loss, classification_report


# =============================================================================
# Logging
# =============================================================================

def _build_logger(log_dir: str, log_filename: str,
                  console_level: str = "INFO", file_level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger("ckd_evaluator")
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
            os.path.join(log_dir, log_filename), maxBytes=5 * 1024 * 1024, backupCount=3
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("File logging unavailable: %s", e)

    return logger


# =============================================================================
# Configuration loader
# =============================================================================

class EvalConfig:
    """Loads config/evaluation_config.yaml."""

    def __init__(self, config_path: str = "config/evaluation_config.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation config not found: {path.resolve()}. "
                f"Expected at config/evaluation_config.yaml."
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        self.random_seed: int = int(raw.get("random_seed", 42))
        self.tasks: Dict[str, Any] = raw["tasks"]
        self.bootstrap: Dict[str, Any] = raw.get("bootstrap", {})
        self.threshold_cfg: Dict[str, Any] = raw.get("threshold_analysis", {})
        self.calibration_cfg: Dict[str, Any] = raw.get("calibration", {})
        self.shap_cfg: Dict[str, Any] = raw.get("shap", {})
        self.fi_cfg: Dict[str, Any] = raw.get("feature_importance", {})
        self.plot_cfg: Dict[str, Any] = raw.get("plots", {})
        self.log_cfg: Dict[str, str] = raw.get("logging", {})
        self.artifact_cfg: Dict[str, Any] = raw.get("artifacts", {})

    def get_task(self, key: str) -> Dict[str, Any]:
        return self.tasks[key]


# =============================================================================
# Model loader
# =============================================================================

def load_model(model_dir: Path, model_name: str, logger: logging.Logger) -> Any:
    """
    Load calibrated model if available, otherwise fall back to final model.
    Never loads from UAE or test directories.
    """
    calib_path = model_dir / model_name / "calibrated_model.joblib"
    final_path  = model_dir / model_name / "final_model.joblib"

    if calib_path.exists():
        logger.info("[Load] Loaded calibrated model from %s", calib_path)
        return joblib.load(calib_path)
    elif final_path.exists():
        logger.warning(
            "[Load] calibrated_model.joblib not found — using final_model.joblib from %s",
            final_path,
        )
        return joblib.load(final_path)
    else:
        raise FileNotFoundError(
            f"No model artifact found in {model_dir / model_name}. "
            f"Run model_training.py first."
        )


def load_selected_features(model_dir: Path, model_name: str) -> List[str]:
    """Load the union_features list used when training this model."""
    features_path = model_dir / model_name / "selected_features.json"
    if features_path.exists():
        with open(features_path, "r") as fh:
            data = json.load(fh)
        return data.get("union_features", [])
    return []


def load_feature_importance(model_dir: Path, model_name: str) -> Dict[str, float]:
    """Load model-native feature importance from training artifacts."""
    fi_path = model_dir / model_name / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path, "r") as fh:
            return json.load(fh)
    return {}


def load_cv_score(model_dir: Path, model_name: str) -> Optional[float]:
    """Load the primary CV score from training artifacts."""
    try:
        meta_path = model_dir / model_name / "training_metadata.json"
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        return meta.get("primary_cv_score")
    except Exception:
        return None


# =============================================================================
# Binary evaluator (UCI)
# =============================================================================

class BinaryEvaluator:
    """Evaluates the best UCI binary model on its held-out test set."""

    def __init__(self, cfg: EvalConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def evaluate(self, run_shap: bool = True, run_ci: bool = True) -> Dict[str, Any]:
        task_cfg = self.cfg.get_task("uci")
        task_name = task_cfg["description"]
        model_name = task_cfg["best_model"]
        model_dir = Path(task_cfg["model_artifacts_dir"])
        splits_dir = Path(task_cfg["splits_dir"])
        out_dir = Path(task_cfg["eval_output_dir"])

        self.logger.info("=" * 60)
        self.logger.info("[UCI] Starting evaluation — model: %s", model_name)

        # ── Load data ───────────────────────────────────────────────────────
        test_df = pd.read_csv(splits_dir / task_cfg["test_file"])
        target_col = task_cfg["target_col"]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(int)

        # ── Load model + features ────────────────────────────────────────────
        model = load_model(model_dir, model_name, self.logger)
        union_features = load_selected_features(model_dir, model_name)
        feature_importance = load_feature_importance(model_dir, model_name)
        cv_score = load_cv_score(model_dir, model_name)

        if union_features:
            # Keep only features that exist in the test set
            available = [f for f in union_features if f in X_test.columns]
            X_test_model = X_test[available]
        else:
            X_test_model = X_test
            available = X_test.columns.tolist()

        self.logger.info("[UCI] Test set: %d rows × %d features", len(y_test), len(available))

        # ── Predict ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test_model.values)
        y_proba = (
            model.predict_proba(X_test_model.values)[:, 1]
            if hasattr(model, "predict_proba") else None
        )

        if y_proba is None:
            self.logger.error("[UCI] Model has no predict_proba — skipping probability-based metrics.")
            y_proba = np.zeros(len(y_test))

        class_names = task_cfg.get("class_names", ["notckd", "ckd"])

        # ── Metrics with CI ──────────────────────────────────────────────────
        bs_cfg = self.cfg.bootstrap
        n_boot = int(bs_cfg.get("n_iterations", 1000)) if run_ci else 0
        ci_level = float(bs_cfg.get("confidence_level", 0.95))

        self.logger.info("[UCI] Computing metrics%s …",
                         f" + {n_boot}-iteration bootstrap CI" if n_boot > 0 else "")

        test_metrics = compute_binary_metrics_with_ci(
            y_test, y_proba,
            threshold=0.5,
            n_bootstrap=n_boot,
            confidence_level=ci_level,
            random_seed=self.cfg.random_seed,
        )

        # ── Calibration ──────────────────────────────────────────────────────
        n_bins = int(self.cfg.calibration_cfg.get("n_bins", 10))
        ece = compute_ece(y_test, y_proba, n_bins=n_bins)
        brier = float(brier_score_loss(y_test, y_proba))
        calib_metrics = {"ece": round(ece, 6), "brier_score": round(brier, 6), "n_bins": n_bins}

        # ── Threshold analysis ───────────────────────────────────────────────
        youden_thr, youden_sens, youden_spec, youden_j = compute_youden_threshold(y_test, y_proba)
        f1_thr, f1_best = compute_f1_threshold(y_test, y_proba)
        mcc_thr, mcc_best = compute_mcc_threshold(y_test, y_proba)
        sweep_df = threshold_sweep(y_test, y_proba, n_steps=99)

        threshold_report = {
            "youden": {"threshold": round(youden_thr, 4),
                       "sensitivity": round(youden_sens, 4),
                       "specificity": round(youden_spec, 4),
                       "youden_j": round(youden_j, 4)},
            "f1_optimal": {"threshold": round(f1_thr, 4), "f1": round(f1_best, 4)},
            "mcc_optimal": {"threshold": round(mcc_thr, 4), "mcc": round(mcc_best, 4)},
        }

        # ── Classification report ────────────────────────────────────────────
        clf_report = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
        )

        # ── Save JSON artifacts ──────────────────────────────────────────────
        summary = {
            "model_name": model_name,
            "task": "uci_binary_ckd",
            "cv_roc_auc": cv_score,
            "n_test_samples": int(len(y_test)),
            "n_features": len(available),
            "features_used": available,
            "test_metrics": test_metrics,
            "calibration_metrics": calib_metrics,
            "threshold_analysis": threshold_report,
        }
        save_json(summary, out_dir / "evaluation_metrics.json")
        save_json(clf_report, out_dir / "classification_report.json")
        save_json(calib_metrics, out_dir / "calibration" / "calibration_metrics.json")
        save_json(threshold_report, out_dir / "threshold_analysis" / "threshold_report.json")
        sweep_df.to_csv(out_dir / "threshold_analysis" / "threshold_sweep.csv", index=False)
        self.logger.info("[UCI] JSON artifacts saved to %s", out_dir)

        # ── Plots ────────────────────────────────────────────────────────────
        pcfg = self.cfg.plot_cfg

        plot_roc_curve(
            y_test, y_proba, model_name, task_name,
            out_dir / "roc_curves" / "roc_curve.png",
            roc_auc=test_metrics.get("roc_auc", {}).get("point"),
            roc_auc_ci=(
                test_metrics.get("roc_auc", {}).get("ci_lower"),
                test_metrics.get("roc_auc", {}).get("ci_upper"),
            ) if run_ci else None,
            plot_cfg=pcfg,
        )

        plot_pr_curve(
            y_test, y_proba, model_name, task_name,
            out_dir / "pr_curves" / "pr_curve.png",
            pr_auc=test_metrics.get("pr_auc", {}).get("point"),
            plot_cfg=pcfg,
        )

        plot_confusion_matrix(
            y_test, y_pred, class_names, model_name, task_name,
            out_dir / "confusion_matrix" / "confusion_matrix.png",
            threshold=0.5, plot_cfg=pcfg,
        )

        plot_calibration_curve(
            y_test, y_proba, model_name, task_name,
            out_dir / "calibration" / "calibration_curve.png",
            n_bins=n_bins, ece=ece, brier=brier, plot_cfg=pcfg,
        )

        plot_threshold_sweep(
            sweep_df, model_name, task_name,
            out_dir / "threshold_analysis" / "threshold_sweep.png",
            youden_threshold=youden_thr,
            f1_threshold=f1_thr,
            fixed_threshold=0.5,
            plot_cfg=pcfg,
        )

        # ── Feature importance ───────────────────────────────────────────────
        if feature_importance and self.cfg.fi_cfg.get("enabled", True):
            plot_feature_importance(
                feature_importance, model_name, task_name,
                out_dir / "feature_importance" / "feature_importance.png",
                top_n=int(self.cfg.fi_cfg.get("top_n", 20)),
                plot_cfg=pcfg,
            )

        # ── SHAP ─────────────────────────────────────────────────────────────
        if run_shap and self.cfg.shap_cfg.get("enabled", True):
            self.logger.info("[UCI] Computing SHAP values …")
            try:
                max_s = self.cfg.shap_cfg.get("max_samples")
                X_shap = X_test_model.iloc[:max_s] if max_s else X_test_model

                _, shap_vals = compute_shap_values(
                    model, X_shap, model_name, "binary",
                    background_samples=int(self.cfg.shap_cfg.get("background_samples", 100)),
                    random_seed=self.cfg.random_seed,
                )
                top_n = int(self.cfg.shap_cfg.get("top_n_features", 20))

                plot_shap_summary(
                    shap_vals, X_shap, model_name, task_name,
                    out_dir / "shap" / "shap_summary.png",
                    max_display=top_n, plot_cfg=pcfg,
                )
                plot_shap_bar(
                    shap_vals, X_shap, model_name, task_name,
                    out_dir / "shap" / "shap_bar.png",
                    max_display=top_n, plot_cfg=pcfg,
                )

                # Save raw SHAP values
                if self.cfg.artifact_cfg.get("save_shap_values_npy", True):
                    from evaluation_utils import _resolve_shap_array
                    shap_arr = _resolve_shap_array(shap_vals, "binary")
                    np.save(out_dir / "shap" / "shap_values.npy", shap_arr)
                    self.logger.info("[UCI] SHAP values saved.")

            except Exception as e:
                self.logger.error("[UCI] SHAP computation failed: %s", e, exc_info=True)

        self.logger.info("[UCI] Evaluation complete. Outputs: %s", out_dir.resolve())
        return summary


# =============================================================================
# Multiclass evaluator (Kaggle)
# =============================================================================

class MulticlassEvaluator:
    """Evaluates the best Kaggle 5-class model on its held-out test set."""

    def __init__(self, cfg: EvalConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def evaluate(self, run_shap: bool = True) -> Dict[str, Any]:
        task_cfg = self.cfg.get_task("kaggle")
        task_name = task_cfg["description"]
        model_name = task_cfg["best_model"]
        model_dir = Path(task_cfg["model_artifacts_dir"])
        splits_dir = Path(task_cfg["splits_dir"])
        out_dir = Path(task_cfg["eval_output_dir"])
        n_classes = int(task_cfg.get("n_classes", 5))
        class_names = task_cfg.get("class_names", [str(i) for i in range(n_classes)])

        self.logger.info("=" * 60)
        self.logger.info("[KAGGLE] Starting evaluation — model: %s", model_name)

        # ── Load data ───────────────────────────────────────────────────────
        test_df = pd.read_csv(splits_dir / task_cfg["test_file"])
        target_col = task_cfg["target_col"]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values.astype(int)

        # ── Load model + features ────────────────────────────────────────────
        model = load_model(model_dir, model_name, self.logger)
        union_features = load_selected_features(model_dir, model_name)
        feature_importance = load_feature_importance(model_dir, model_name)
        cv_score = load_cv_score(model_dir, model_name)

        if union_features:
            available = [f for f in union_features if f in X_test.columns]
            X_test_model = X_test[available]
        else:
            X_test_model = X_test
            available = X_test.columns.tolist()

        self.logger.info("[KAGGLE] Test set: %d rows × %d features",
                         len(y_test), len(available))

        # ── Predict ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test_model.values)
        y_proba = (
            model.predict_proba(X_test_model.values)
            if hasattr(model, "predict_proba") else None
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        self.logger.info("[KAGGLE] Computing multiclass metrics …")
        test_metrics = compute_multiclass_metrics_full(
            y_test, y_pred, y_proba,
            n_classes=n_classes, class_names=class_names,
        )

        # ── Save JSON artifacts ──────────────────────────────────────────────
        summary = {
            "model_name": model_name,
            "task": "kaggle_multiclass_ckd_staging",
            "cv_balanced_accuracy": cv_score,
            "n_test_samples": int(len(y_test)),
            "n_classes": n_classes,
            "class_names": class_names,
            "n_features": len(available),
            "features_used": available,
            "test_metrics": test_metrics,
        }
        save_json(summary, out_dir / "evaluation_metrics.json")
        clf_report = test_metrics.pop("test_classification_report", None) or test_metrics.pop("classification_report", None)
        if clf_report:
            save_json(clf_report, out_dir / "classification_report.json")
        self.logger.info("[KAGGLE] JSON artifacts saved to %s", out_dir)

        # ── Plots ────────────────────────────────────────────────────────────
        pcfg = self.cfg.plot_cfg

        plot_confusion_matrix_multiclass(
            y_test, y_pred, class_names, model_name, task_name,
            out_dir / "confusion_matrix" / "confusion_matrix.png",
            plot_cfg=pcfg,
        )

        if y_proba is not None:
            plot_roc_ovr(
                y_test, y_proba, class_names, model_name, task_name,
                out_dir / "roc_curves" / "roc_ovr.png",
                plot_cfg=pcfg,
            )

        if feature_importance and self.cfg.fi_cfg.get("enabled", True):
            plot_feature_importance(
                feature_importance, model_name, task_name,
                out_dir / "feature_importance" / "feature_importance.png",
                top_n=int(self.cfg.fi_cfg.get("top_n", 20)),
                plot_cfg=pcfg,
            )

        # ── SHAP ─────────────────────────────────────────────────────────────
        if run_shap and self.cfg.shap_cfg.get("enabled", True):
            self.logger.info("[KAGGLE] Computing SHAP values …")
            try:
                max_s = self.cfg.shap_cfg.get("max_samples")
                X_shap = X_test_model.iloc[:max_s] if max_s else X_test_model
                top_n = int(self.cfg.shap_cfg.get("top_n_features", 20))

                _, shap_vals = compute_shap_values(
                    model, X_shap, model_name, "multiclass",
                    background_samples=int(self.cfg.shap_cfg.get("background_samples", 100)),
                    random_seed=self.cfg.random_seed,
                )

                plot_shap_summary(
                    shap_vals, X_shap, model_name, task_name,
                    out_dir / "shap" / "shap_summary.png",
                    max_display=top_n, plot_cfg=pcfg,
                )
                plot_shap_bar(
                    shap_vals, X_shap, model_name, task_name,
                    out_dir / "shap" / "shap_bar.png",
                    max_display=top_n, plot_cfg=pcfg,
                )
            except Exception as e:
                self.logger.error("[KAGGLE] SHAP failed: %s", e, exc_info=True)

        self.logger.info("[KAGGLE] Evaluation complete. Outputs: %s", out_dir.resolve())
        return summary


# =============================================================================
# Main orchestrator
# =============================================================================

class CKDEvaluator:
    """
    Top-level evaluation orchestrator.
    Runs UCI and/or Kaggle evaluation, then generates the summary report.
    """

    def __init__(self, config_path: str = "config/evaluation_config.yaml") -> None:
        self.cfg = EvalConfig(config_path)
        log_cfg = self.cfg.log_cfg
        self.logger = _build_logger(
            log_dir=log_cfg.get("log_dir", "logs"),
            log_filename=log_cfg.get("log_filename", "evaluation.log"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )

    def run(
        self,
        tasks: Optional[List[str]] = None,
        run_shap: bool = True,
        run_ci: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation for specified tasks.

        Parameters
        ----------
        tasks : list of task keys to evaluate, e.g. ["uci", "kaggle"].
                If None, runs all tasks.
        run_shap : if False, skips SHAP computation (faster).
        run_ci : if False, skips bootstrap CI (faster).
        """
        if tasks is None:
            tasks = ["uci", "kaggle"]

        self.logger.info("=" * 70)
        self.logger.info("CKD Evaluation Pipeline — START")
        self.logger.info("Tasks: %s | SHAP: %s | Bootstrap CI: %s",
                         tasks, run_shap, run_ci)
        self.logger.info("=" * 70)

        results: Dict[str, Any] = {}

        if "uci" in tasks:
            evaluator = BinaryEvaluator(self.cfg, self.logger)
            results["uci"] = evaluator.evaluate(run_shap=run_shap, run_ci=run_ci)

        if "kaggle" in tasks:
            evaluator = MulticlassEvaluator(self.cfg, self.logger)
            results["kaggle"] = evaluator.evaluate(run_shap=run_shap)

        # ── Summary report ───────────────────────────────────────────────────
        report_path = Path(
            self.cfg.artifact_cfg.get("summary_report_path", "artifacts/evaluation/summary_report.md")
        )
        try:
            generate_summary_report(
                uci_metrics=results.get("uci"),
                kaggle_metrics=results.get("kaggle"),
                uae_report=None,     # UAE report is added by external_validation.py
                output_path=report_path,
                pipeline_metadata={"random_seed": self.cfg.random_seed,
                                   "sklearn_version": _get_sklearn_version()},
            )
        except Exception as e:
            self.logger.warning("Summary report generation failed: %s", e)

        self.logger.info("=" * 70)
        self.logger.info("CKD Evaluation Pipeline — COMPLETE")
        self.logger.info("All outputs in: artifacts/evaluation/")
        self.logger.info("=" * 70)

        return results


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CKD ML Pipeline — Evaluation Stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py
  python evaluate.py --tasks uci
  python evaluate.py --tasks uci kaggle --no-shap
  python evaluate.py --no-ci --config config/evaluation_config.yaml
        """,
    )
    parser.add_argument(
        "--config", default="config/evaluation_config.yaml",
        help="Path to evaluation_config.yaml (default: config/evaluation_config.yaml)",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=["uci", "kaggle"],
        default=["uci", "kaggle"],
        help="Which tasks to evaluate (default: uci kaggle)",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP computation (significantly faster)",
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Skip bootstrap confidence intervals (faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluator = CKDEvaluator(config_path=args.config)
    results = evaluator.run(
        tasks=args.tasks,
        run_shap=not args.no_shap,
        run_ci=not args.no_ci,
    )

    print("\n── Evaluation Complete ──\n")
    for task, result in results.items():
        print(f"{task.upper()}:")
        print(f"  Model : {result.get('model_name', 'N/A')}")
        test_m = result.get("test_metrics", {})
        for key in ("roc_auc", "f1", "balanced_accuracy", "mcc"):
            v = test_m.get(key) or test_m.get(f"test_{key}")
            if isinstance(v, dict):
                pt = v.get("point", "N/A")
                lo = v.get("ci_lower", "")
                hi = v.get("ci_upper", "")
                ci = f" (95% CI: {lo:.4f}–{hi:.4f})" if lo else ""
                print(f"  {key:<25}: {pt:.4f}{ci}")
            elif isinstance(v, (int, float)):
                print(f"  {key:<25}: {v:.4f}")
        print()

    print("Artifacts → artifacts/evaluation/")
    print("Summary   → artifacts/evaluation/summary_report.md")
