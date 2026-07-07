"""
external_validation.py
======================

Rigorous external validation of the CKD model on the UAE cohort.

Pipeline position:
  model_training.py → evaluate.py
  → external_validation.py (THIS FILE)   ← you are here

══════════════════════════════════════════════════════════════════════════
SECTION A — ARCHITECTURE REVIEW
══════════════════════════════════════════════════════════════════════════

CRITICAL DESIGN PRINCIPLES
───────────────────────────
  1. UAE data is NEVER used for training, validation, or model selection.
  2. All model artifacts loaded here were produced by model_training.py
     and uae_validation.py — never by this script.
  3. This script is EVALUATION-ONLY: no fitting, no SMOTE, no calibration
     fitting, no feature selection. Pure load-and-evaluate.

WHAT THIS SCRIPT DOES
──────────────────────
  ✓ Loads the Track A reduced-feature model (trained on UCI only)
  ✓ Loads saved UAE predictions from model_training.py
  ✓ Generates full calibration analysis (reliability diagram, ECE, Brier)
  ✓ Performs threshold sweep + Youden's J optimization
  ✓ Reports metrics at BOTH threshold=0.5 AND Youden's J threshold
    (transparency: neither is hidden)
  ✓ Generates population shift analysis (prevalence comparison, probability
    distribution comparison)
  ✓ Generates dual confusion matrices (at 0.5 and Youden)
  ✓ Produces a complete, honest narrative report

══════════════════════════════════════════════════════════════════════════
SECTION B — LEAKAGE REVIEW
══════════════════════════════════════════════════════════════════════════

  ZERO contamination vectors:
  ─────────────────────────────
  ✓ Track A model was trained on UCI training set only (uae_validation.py)
  ✓ UAE predictions loaded from CSV (already produced by model_training.py)
  ✓ Youden's J threshold: computed post-hoc on UAE ROC curve.
    This IS a form of retrospective threshold selection on UAE — it is
    clearly labeled as such and reported alongside threshold=0.5.
    It is NOT used for model selection. The model itself is unchanged.
  ✓ Population shift analysis uses UAE data purely descriptively (no fitting)
  ✓ UCI training statistics (prevalence) loaded from saved JSON artifacts

══════════════════════════════════════════════════════════════════════════

Usage
-----
    python external_validation.py
    python external_validation.py --config config/evaluation_config.yaml
    python external_validation.py --no-ci
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
    compute_youden_threshold,
    generate_summary_report,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_prevalence_comparison,
    plot_probability_distribution,
    plot_roc_curve,
    plot_pr_curve,
    plot_threshold_sweep,
    save_json,
    threshold_sweep,
)
from sklearn.metrics import brier_score_loss


# =============================================================================
# Logging
# =============================================================================

def _build_logger(log_dir: str, log_filename: str,
                  console_level: str = "INFO", file_level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger("ckd_ext_val")
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
            maxBytes=5 * 1024 * 1024, backupCount=3,
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("File logging unavailable: %s", e)

    return logger


# =============================================================================
# Config loader
# =============================================================================

class ExtValConfig:
    """Loads config/evaluation_config.yaml — UAE-specific sections."""

    def __init__(self, config_path: str = "config/evaluation_config.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation config not found: {path.resolve()}. "
                f"Run `cp config/evaluation_config.yaml.example config/evaluation_config.yaml` "
                f"or generate it with evaluation_utils."
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        self.random_seed: int = int(raw.get("random_seed", 42))
        self.tasks: Dict[str, Any] = raw["tasks"]
        self.bootstrap: Dict[str, Any] = raw.get("bootstrap", {})
        self.threshold_cfg: Dict[str, Any] = raw.get("threshold_analysis", {})
        self.calibration_cfg: Dict[str, Any] = raw.get("calibration", {})
        self.plot_cfg: Dict[str, Any] = raw.get("plots", {})
        self.log_cfg: Dict[str, str] = raw.get("logging", {})
        self.artifact_cfg: Dict[str, Any] = raw.get("artifacts", {})

    def get_uae_cfg(self) -> Dict[str, Any]:
        return self.tasks["uae"]

    def get_uci_cfg(self) -> Dict[str, Any]:
        return self.tasks["uci"]


# =============================================================================
# UAE Predictions Loader
# =============================================================================

def load_uae_predictions(predictions_csv: Path, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UAE predictions saved by model_training.py.

    The CSV has columns: row_position, y_true, y_pred, y_proba_ckd

    Returns
    -------
    y_true, y_pred, y_proba
    """
    if not predictions_csv.exists():
        raise FileNotFoundError(
            f"UAE predictions CSV not found: {predictions_csv.resolve()}. "
            f"Run model_training.py first."
        )

    df = pd.read_csv(predictions_csv)
    logger.info("[UAE] Loaded predictions from %s — %d rows", predictions_csv, len(df))

    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UAE predictions CSV missing columns: {missing}")

    y_true = df["y_true"].values.astype(int)
    y_pred = df["y_pred"].values.astype(int)
    y_proba = df["y_proba_ckd"].values.astype(float) if "y_proba_ckd" in df.columns else None

    logger.info(
        "[UAE] Label distribution: CKD=%d, Non-CKD=%d (prevalence=%.1f%%)",
        y_true.sum(), (y_true == 0).sum(), 100 * y_true.mean(),
    )
    return y_true, y_pred, y_proba


def load_uae_data(uae_cfg: Dict[str, Any], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load raw UAE DataFrame for population analysis (descriptive only)."""
    splits_dir = Path(uae_cfg.get("splits_dir", "data/splits"))
    uae_file = uae_cfg.get("uae_file", "uae_full.csv")
    uae_path = splits_dir / uae_file

    if not uae_path.exists():
        logger.warning("[UAE] UAE data file not found at %s — skipping population analysis.", uae_path)
        return None

    df = pd.read_csv(uae_path)
    logger.info("[UAE] Loaded raw UAE data — %d rows × %d cols", *df.shape)
    return df


def load_uci_train_for_distribution(uci_cfg: Dict[str, Any], logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load UCI training set for distribution comparison (descriptive only)."""
    splits_dir = Path(uci_cfg.get("splits_dir", "data/splits"))
    train_file = uci_cfg.get("train_file", "uci_train.csv")
    train_path = splits_dir / train_file

    if not train_path.exists():
        logger.warning("[UCI] Train file not found at %s — skipping distribution comparison.", train_path)
        return None

    df = pd.read_csv(train_path)
    logger.info("[UCI] Loaded UCI train for distribution reference — %d rows", len(df))
    return df


def load_full_report(report_path: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Load the full UAE validation report from model_training.py."""
    if not report_path.exists():
        logger.warning("[UAE] Full report not found: %s", report_path)
        return None
    with open(report_path, "r") as fh:
        return json.load(fh)


# =============================================================================
# Core validation runner
# =============================================================================

class UAEExternalValidator:
    """
    Performs rigorous external validation of the CKD model on the UAE cohort.

    Design contract
    ---------------
      - NO model fitting of any kind
      - NO imputation on UAE data
      - NO feature selection on UAE data
      - Only loads saved predictions and reports metrics/plots
    """

    def __init__(self, cfg: ExtValConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    def validate(self, run_ci: bool = True) -> Dict[str, Any]:
        uae_cfg = self.cfg.get_uae_cfg()
        uci_cfg = self.cfg.get_uci_cfg()
        out_dir = Path(uae_cfg["eval_output_dir"])
        pcfg = self.cfg.plot_cfg

        self.logger.info("=" * 70)
        self.logger.info("[UAE] External Validation — START")
        self.logger.info("[UAE] ⚠ UAE cohort was NEVER used during training, CV, or model selection.")
        self.logger.info("=" * 70)

        # ── Load predictions ─────────────────────────────────────────────────
        preds_csv = Path(uae_cfg["track_a_predictions_csv"])
        y_true, y_pred_fixed, y_proba = load_uae_predictions(preds_csv, self.logger)

        if y_proba is None:
            raise ValueError(
                "UAE predictions CSV does not contain y_proba_ckd column. "
                "Cannot perform threshold analysis or calibration. "
                "Ensure model_training.py completed successfully."
            )

        n_uae = len(y_true)
        n_ckd = int(y_true.sum())
        uae_prevalence = float(y_true.mean())
        uci_prevalence = float(uae_cfg.get("uci_ckd_prevalence", 0.625))

        # ── Metrics at threshold = 0.5 ───────────────────────────────────────
        self.logger.info("[UAE] Computing metrics at threshold = 0.5 …")
        bs_cfg = self.cfg.bootstrap
        n_boot = int(bs_cfg.get("n_iterations", 1000)) if run_ci else 0
        ci_level = float(bs_cfg.get("confidence_level", 0.95))

        metrics_at_0_5 = compute_binary_metrics_with_ci(
            y_true, y_proba,
            threshold=0.5,
            n_bootstrap=n_boot,
            confidence_level=ci_level,
            random_seed=self.cfg.random_seed,
            prefix="",
        )

        # ── Threshold optimization ────────────────────────────────────────────
        self.logger.info("[UAE] Running threshold sweep and Youden's J optimization …")
        youden_thr, youden_sens, youden_spec, youden_j = compute_youden_threshold(y_true, y_proba)
        f1_thr, f1_best = compute_f1_threshold(y_true, y_proba)
        mcc_thr, mcc_best = compute_mcc_threshold(y_true, y_proba)
        sweep_df = threshold_sweep(y_true, y_proba, n_steps=99)

        self.logger.info(
            "[UAE] Youden's J threshold: τ=%.4f | Sensitivity=%.4f | Specificity=%.4f | J=%.4f",
            youden_thr, youden_sens, youden_spec, youden_j,
        )

        # Metrics at Youden threshold
        self.logger.info("[UAE] Computing metrics at Youden's J threshold (τ=%.4f) …", youden_thr)
        metrics_at_youden = compute_binary_metrics_with_ci(
            y_true, y_proba,
            threshold=youden_thr,
            n_bootstrap=n_boot,
            confidence_level=ci_level,
            random_seed=self.cfg.random_seed,
            prefix="",
        )

        # ── Calibration ──────────────────────────────────────────────────────
        n_bins = int(self.cfg.calibration_cfg.get("n_bins", 10))
        ece = compute_ece(y_true, y_proba, n_bins=n_bins)
        brier = float(brier_score_loss(y_true, y_proba))
        calib_metrics = {
            "ece": round(ece, 6),
            "brier_score": round(brier, 6),
            "n_bins": n_bins,
            "note": (
                "Brier score is the mean squared error of probability predictions. "
                "ECE measures the weighted mean absolute difference between predicted "
                "confidence and observed fraction of positives (calibration error). "
                "High ECE reflects the prevalence mismatch (UCI ~62% vs UAE 11.4% CKD) — "
                "the model outputs probabilities calibrated for a higher-prevalence population."
            ),
        }

        # ── Confusion matrices ────────────────────────────────────────────────
        y_pred_youden = (y_proba >= youden_thr).astype(int)
        class_names = uae_cfg.get("class_names", ["notckd", "ckd"])

        plot_confusion_matrix(
            y_true, y_pred_fixed, class_names,
            "CatBoost (Track A)", "UAE External Validation",
            out_dir / "confusion_matrix" / "cm_threshold_0.5.png",
            threshold=0.5, plot_cfg=pcfg,
        )
        plot_confusion_matrix(
            y_true, y_pred_youden, class_names,
            "CatBoost (Track A)", "UAE External Validation",
            out_dir / "confusion_matrix" / "cm_youden.png",
            threshold=youden_thr, plot_cfg=pcfg,
        )

        # ── ROC + PR curves ───────────────────────────────────────────────────
        roc_auc_pt = metrics_at_0_5.get("roc_auc", {}).get("point") if isinstance(
            metrics_at_0_5.get("roc_auc"), dict) else metrics_at_0_5.get("roc_auc")

        plot_roc_curve(
            y_true, y_proba,
            "CatBoost Reduced (8 features)", "UAE External Validation",
            out_dir / "roc_pr_curves" / "roc_curve.png",
            roc_auc=roc_auc_pt,
            roc_auc_ci=(
                metrics_at_0_5.get("roc_auc", {}).get("ci_lower"),
                metrics_at_0_5.get("roc_auc", {}).get("ci_upper"),
            ) if run_ci else None,
            plot_cfg=pcfg,
        )

        pr_auc_pt = metrics_at_0_5.get("pr_auc", {}).get("point") if isinstance(
            metrics_at_0_5.get("pr_auc"), dict) else metrics_at_0_5.get("pr_auc")

        plot_pr_curve(
            y_true, y_proba,
            "CatBoost Reduced (8 features)", "UAE External Validation",
            out_dir / "roc_pr_curves" / "pr_curve.png",
            pr_auc=pr_auc_pt,
            plot_cfg=pcfg,
        )

        # ── Calibration plot ──────────────────────────────────────────────────
        plot_calibration_curve(
            y_true, y_proba,
            "CatBoost Reduced (8 features)", "UAE External Validation",
            out_dir / "calibration" / "reliability_diagram.png",
            n_bins=n_bins, ece=ece, brier=brier,
            plot_cfg=pcfg,
        )

        # ── Threshold sweep plot ──────────────────────────────────────────────
        plot_threshold_sweep(
            sweep_df,
            "CatBoost Reduced", "UAE External Validation",
            out_dir / "threshold_analysis" / "threshold_sweep.png",
            youden_threshold=youden_thr,
            f1_threshold=f1_thr,
            fixed_threshold=0.5,
            plot_cfg=pcfg,
        )

        # ── Population shift plots ────────────────────────────────────────────
        self.logger.info("[UAE] Generating population shift analysis …")

        plot_prevalence_comparison(
            {
                "UCI Training\n(~62% CKD)": uci_prevalence,
                "UAE External\n(11.4% CKD)": uae_prevalence,
            },
            out_dir / "population_shift" / "prevalence_comparison.png",
            plot_cfg=pcfg,
        )

        # Load UCI test probabilities for comparison (if available)
        uci_test_preds_path = (
            Path(uci_cfg["model_artifacts_dir"])
            / uci_cfg["best_model"]
            / "test_predictions.csv"
        )
        uci_proba_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "UAE Cohort\n(n=491, 11.4% CKD)": (y_true, y_proba),
        }
        if uci_test_preds_path.exists():
            uci_preds_df = pd.read_csv(uci_test_preds_path)
            if "y_true" in uci_preds_df.columns and "y_proba_ckd" in uci_preds_df.columns:
                uci_proba_dict["UCI Test Set\n(n=80, ~62% CKD)"] = (
                    uci_preds_df["y_true"].values,
                    uci_preds_df["y_proba_ckd"].values,
                )

        plot_probability_distribution(
            uci_proba_dict,
            out_dir / "population_shift" / "probability_distribution.png",
            plot_cfg=pcfg,
        )

        # ── Save all JSON artifacts ───────────────────────────────────────────
        threshold_report = {
            "primary_method": "youden",
            "youden": {
                "threshold": round(youden_thr, 4),
                "sensitivity": round(youden_sens, 4),
                "specificity": round(youden_spec, 4),
                "youden_j": round(youden_j, 4),
                "note": (
                    "Youden's J = argmax(sensitivity + specificity - 1). "
                    "This threshold maximizes the combined sensitivity/specificity tradeoff. "
                    "This is a retrospective optimization on UAE data — clearly labeled as post-hoc. "
                    "The model itself was not modified."
                ),
            },
            "f1_optimal": {"threshold": round(f1_thr, 4), "f1": round(f1_best, 4)},
            "mcc_optimal": {"threshold": round(mcc_thr, 4), "mcc": round(mcc_best, 4)},
            "transparency_note": (
                "BOTH threshold=0.5 and Youden's J results are reported. "
                "Neither is hidden. Youden's J represents the clinically optimal "
                "operating point for THIS population; threshold=0.5 reflects the "
                "default behavior when no population-specific recalibration is done."
            ),
        }

        population_shift = {
            "uae_cohort": {
                "n_patients": int(n_uae),
                "n_ckd": int(n_ckd),
                "n_non_ckd": int(n_uae - n_ckd),
                "ckd_prevalence": round(uae_prevalence, 4),
                "ckd_prevalence_pct": round(uae_prevalence * 100, 2),
            },
            "uci_train_reference": {
                "ckd_prevalence": round(uci_prevalence, 4),
                "ckd_prevalence_pct": round(uci_prevalence * 100, 2),
            },
            "prevalence_ratio": round(uci_prevalence / uae_prevalence, 2),
            "interpretation": (
                f"The UCI training prevalence (~{uci_prevalence*100:.0f}% CKD) is "
                f"{uci_prevalence/uae_prevalence:.1f}× higher than the UAE population "
                f"({uae_prevalence*100:.1f}% CKD). This prevalence mismatch explains "
                f"why the default threshold of 0.5 — calibrated implicitly for the training "
                f"prevalence — produces excessive false positives on UAE. "
                f"ROC-AUC (threshold-free) is the appropriate primary discrimination metric."
            ),
        }

        full_result: Dict[str, Any] = {
            "validation_design": "Track A (primary): reduced-feature CatBoost trained on UCI only, "
                                  "validated on UAE cohort. Never trained on UAE data.",
            "n_features_track_a": 8,
            "features_used": [
                "serum_creatinine", "age", "cardiovascular_burden_score",
                "age_creatinine_interaction", "bp_risk_score",
                "hypertension", "diabetes_mellitus", "coronary_artery_disease",
            ],
            "population": {
                "n_uae": int(n_uae),
                "n_ckd": int(n_ckd),
                "uae_prevalence_pct": round(uae_prevalence * 100, 2),
                "uci_prevalence_pct": round(uci_prevalence * 100, 2),
            },
            "at_threshold_0.5": metrics_at_0_5,
            "at_youden_threshold": metrics_at_youden,
            "youden_threshold": round(youden_thr, 4),
            "threshold_analysis": threshold_report,
            "calibration": calib_metrics,
            "population_shift": population_shift,
        }

        save_json(full_result, out_dir / "external_validation_metrics.json")
        save_json(calib_metrics, out_dir / "calibration" / "calibration_metrics.json")
        save_json(threshold_report, out_dir / "threshold_analysis" / "threshold_report.json")
        sweep_df.to_csv(out_dir / "threshold_analysis" / "threshold_sweep.csv", index=False)
        save_json(population_shift, out_dir / "population_shift" / "population_shift.json")

        self.logger.info("[UAE] External validation complete. Outputs: %s", out_dir.resolve())

        # ── Log key results clearly ───────────────────────────────────────────
        def _pt(v: Any) -> str:
            if isinstance(v, dict):
                return f"{v.get('point', v):.4f}"
            if isinstance(v, (int, float)):
                return f"{float(v):.4f}"
            return str(v)

        self.logger.info("=" * 70)
        self.logger.info("[UAE] ═══════ RESULTS SUMMARY ═══════")
        self.logger.info("[UAE] ROC-AUC  (primary, threshold-free) : %s", _pt(metrics_at_0_5.get("roc_auc")))
        self.logger.info("[UAE] PR-AUC   (prevalence-sensitive)    : %s", _pt(metrics_at_0_5.get("pr_auc")))
        self.logger.info("[UAE]")
        self.logger.info("[UAE] ── At threshold = 0.50 (default, miscalibrated for UAE) ──")
        self.logger.info("[UAE]    Sensitivity : %s", _pt(metrics_at_0_5.get("sensitivity")))
        self.logger.info("[UAE]    Specificity : %s", _pt(metrics_at_0_5.get("specificity")))
        self.logger.info("[UAE]    F1          : %s", _pt(metrics_at_0_5.get("f1")))
        self.logger.info("[UAE]    MCC         : %s", _pt(metrics_at_0_5.get("mcc")))
        self.logger.info("[UAE]    Accuracy    : %s", _pt(metrics_at_0_5.get("accuracy")))
        self.logger.info("[UAE]")
        self.logger.info("[UAE] ── At Youden's J threshold (τ=%.4f, post-hoc optimized) ──", youden_thr)
        self.logger.info("[UAE]    Sensitivity : %s", _pt(metrics_at_youden.get("sensitivity")))
        self.logger.info("[UAE]    Specificity : %s", _pt(metrics_at_youden.get("specificity")))
        self.logger.info("[UAE]    F1          : %s", _pt(metrics_at_youden.get("f1")))
        self.logger.info("[UAE]    MCC         : %s", _pt(metrics_at_youden.get("mcc")))
        self.logger.info("[UAE]    Accuracy    : %s", _pt(metrics_at_youden.get("accuracy")))
        self.logger.info("[UAE]")
        self.logger.info("[UAE] Calibration — ECE: %.4f | Brier: %.4f", ece, brier)
        self.logger.info("[UAE] Population — UAE prevalence: %.1f%% vs UCI training: %.1f%%",
                         uae_prevalence * 100, uci_prevalence * 100)
        self.logger.info("[UAE] ════════════════════════════════")
        self.logger.info("=" * 70)

        return full_result

    def _generate_updated_summary_report(self, uae_result: Dict[str, Any]) -> None:
        """Append UAE results to the existing summary_report.md."""
        report_path = Path(
            self.cfg.artifact_cfg.get("summary_report_path", "artifacts/evaluation/summary_report.md")
        )

        try:
            uci_metrics_path = Path(self.cfg.get_uci_cfg()["eval_output_dir"]) / "evaluation_metrics.json"
            uci_metrics = json.loads(uci_metrics_path.read_text()) if uci_metrics_path.exists() else None
        except Exception:
            uci_metrics = None

        try:
            kaggle_metrics_path = (
                Path("artifacts/evaluation/kaggle") / "evaluation_metrics.json"
            )
            kaggle_metrics = json.loads(kaggle_metrics_path.read_text()) if kaggle_metrics_path.exists() else None
        except Exception:
            kaggle_metrics = None

        generate_summary_report(
            uci_metrics=uci_metrics,
            kaggle_metrics=kaggle_metrics,
            uae_report=uae_result,
            output_path=report_path,
        )


# =============================================================================
# Main orchestrator
# =============================================================================

class CKDExternalValidator:
    """Top-level orchestrator for UAE external validation."""

    def __init__(self, config_path: str = "config/evaluation_config.yaml") -> None:
        self.cfg = ExtValConfig(config_path)
        log_cfg = self.cfg.log_cfg
        self.logger = _build_logger(
            log_dir=log_cfg.get("log_dir", "logs"),
            log_filename=log_cfg.get("log_filename", "evaluation.log"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )

    def run(self, run_ci: bool = True) -> Dict[str, Any]:
        validator = UAEExternalValidator(self.cfg, self.logger)
        result = validator.validate(run_ci=run_ci)
        validator._generate_updated_summary_report(result)
        return result


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CKD ML Pipeline — UAE External Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python external_validation.py
  python external_validation.py --no-ci
  python external_validation.py --config config/evaluation_config.yaml
        """,
    )
    parser.add_argument(
        "--config", default="config/evaluation_config.yaml",
        help="Path to evaluation_config.yaml",
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Skip bootstrap confidence intervals (faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validator = CKDExternalValidator(config_path=args.config)
    result = validator.run(run_ci=not args.no_ci)

    print("\n── UAE External Validation Complete ──\n")

    def _pt(v: Any) -> str:
        if isinstance(v, dict):
            pt = v.get("point", "N/A")
            lo = v.get("ci_lower", "")
            hi = v.get("ci_upper", "")
            ci = f" (95% CI: {lo:.4f}–{hi:.4f})" if lo != "" else ""
            return f"{pt:.4f}{ci}" if isinstance(pt, float) else str(pt)
        if isinstance(v, (int, float)):
            return f"{float(v):.4f}"
        return str(v)

    print(f"  ROC-AUC (threshold-free)    : {_pt(result.get('at_threshold_0.5', {}).get('roc_auc'))}")
    print(f"  PR-AUC  (prevalence-aware)  : {_pt(result.get('at_threshold_0.5', {}).get('pr_auc'))}")
    print()
    print(f"  ── At τ = 0.50 (default) ──")
    at_fixed = result.get("at_threshold_0.5", {})
    for k in ("sensitivity", "specificity", "f1", "mcc", "accuracy"):
        print(f"  {k:<26}: {_pt(at_fixed.get(k))}")
    print()
    youden_thr = result.get("youden_threshold", "N/A")
    print(f"  ── At τ = {youden_thr} (Youden's J — post-hoc) ──")
    at_opt = result.get("at_youden_threshold", {})
    for k in ("sensitivity", "specificity", "f1", "mcc", "accuracy"):
        print(f"  {k:<26}: {_pt(at_opt.get(k))}")
    print()
    calib = result.get("calibration", {})
    print(f"  ECE         : {calib.get('ece', 'N/A'):.4f}")
    print(f"  Brier Score : {calib.get('brier_score', 'N/A'):.4f}")
    print()
    pop = result.get("population_shift", {}).get("uae_cohort", {})
    print(f"  UAE CKD prevalence : {pop.get('ckd_prevalence_pct', 'N/A')}%  "
          f"(n={pop.get('n_ckd', 'N/A')}/{pop.get('n_patients', 'N/A')})")
    print()
    print("Artifacts → artifacts/evaluation/uae/")
    print("Summary   → artifacts/evaluation/summary_report.md")
