"""
ablation_study.py
=================

Feature-subset ablation study for the CKD ML Pipeline.

Pipeline position:
  preprocess.py → feature_engineering.py → train_test_split.py
  → model_training.py → evaluate.py → external_validation.py
  → explainability.py → ablation_study.py  (THIS FILE)  ← you are here

All paths are resolved through pipeline_paths.PipelinePaths which reads
config/evaluation_config.yaml — the same config used by evaluate.py.
Nothing is hardcoded.

══════════════════════════════════════════════════════════════════════════
PURPOSE
══════════════════════════════════════════════════════════════════════════

  "Do we really need all these features, or can a simpler model achieve
   nearly the same performance?"

  Evaluates the saved (frozen) model on progressively smaller feature
  subsets ranked by global SHAP importance. No retraining of the full
  pipeline is performed. Non-selected features are replaced with their
  TRAINING-SET MEAN (mask-based ablation).

══════════════════════════════════════════════════════════════════════════
APPROACH — MASK-BASED ABLATION (LEAKAGE-SAFE)
══════════════════════════════════════════════════════════════════════════

  • The model is FROZEN — not refitted.
  • For each feature subset, non-selected features are set to their
    training-set mean (computed from data/splits/uci_train.csv).
  • The imputation mean is from the training set only — no leakage.

══════════════════════════════════════════════════════════════════════════
FEATURE RANKING
══════════════════════════════════════════════════════════════════════════

  Priority order for SHAP ranking source:
    1. Pre-computed SHAP CSV from explainability.py
       (artifacts/explainability/uci/{model}/shap_values.csv)
    2. Global importance CSV from explainability.py
       (artifacts/explainability/uci/global_shap_importance.csv)
    3. Compute SHAP on the fly (if shap is installed)
    4. Fallback: model-native feature_importances_ / coef_

══════════════════════════════════════════════════════════════════════════
OUTPUTS  →  artifacts/ablation/
══════════════════════════════════════════════════════════════════════════

  ablation_results.csv               Full metric table
  ablation_report.md                 Markdown summary
  ablation_heatmap.png               Metric × subset heatmap (best model)
  ablation_roc_auc.png               ROC-AUC line plot
  ablation_f1.png                    F1 line plot
  ablation_roc_auc_bar.png           Bar chart (best model)
  {model}/shap_feature_ranking.csv   SHAP-ranked feature list used

Usage
-----
    python ablation_study.py
    python ablation_study.py --model CatBoost
    python ablation_study.py --subsets 3 5 10 20
    python ablation_study.py --output-dir artifacts/ablation
    python ablation_study.py --config config/evaluation_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Logging
# =============================================================================

def _build_logger(log_dir: str = "logs", log_file: str = "ablation_study.log") -> logging.Logger:
    logger = logging.getLogger("ckd_ablation")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_file), maxBytes=5 * 1024 * 1024, backupCount=2
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("File logging unavailable: %s", e)
    return logger


logger = _build_logger()

# =============================================================================
# Constants
# =============================================================================

UCI_MODELS    = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
PRIORITY_ORDER = ["CatBoost", "XGBoost", "RandomForest", "LightGBM", "LogisticRegression"]
TASK_KEY      = "uci"

# Clinical candidate features — used as a fixed interpretable reference subset
CLINICAL_CANDIDATES = [
    "serum_creatinine", "hemoglobin", "albumin", "specific_gravity",
    "packed_cell_volume", "blood_urea", "blood_pressure", "age",
    # short-name aliases also used in some encodings
    "sc", "hemo", "al", "sg", "pcv", "bu", "bp",
]

# =============================================================================
# Optional dependency guards
# =============================================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.error("matplotlib not installed. Run: pip install matplotlib")

try:
    import shap as _shap_mod
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

# =============================================================================
# Path resolver
# =============================================================================

def _get_paths(config_path: str = "config/evaluation_config.yaml"):
    from pipeline_paths import PipelinePaths
    return PipelinePaths(config_path)

# =============================================================================
# Helpers
# =============================================================================

def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _safe(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, dict):            return {str(k): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):   return [_safe(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_safe(data), fh, indent=2)


def _savefig(fig: "plt.Figure", path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("[Plot] Saved → %s", path)

# =============================================================================
# Model loading
# =============================================================================

def _load_model(pp, model_name: str) -> Any:
    calib = pp.calibrated_model(TASK_KEY, model_name)
    final = pp.final_model(TASK_KEY, model_name)
    if calib.exists():
        return joblib.load(calib)
    if final.exists():
        logger.warning("[%s] Using final_model.joblib (calibrated not found).", model_name)
        return joblib.load(final)
    raise FileNotFoundError(
        f"No model artifact found for '{model_name}'.\n"
        f"  Checked: {calib}\n"
        f"  Checked: {final}\n"
        f"  Run model_training.py first."
    )

# =============================================================================
# Data loading  (THE FIX — reads paths from evaluation_config.yaml)
# =============================================================================

def _load_data(pp, model_name: str) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, List[str]]:
    """
    Returns (X_test, y_test, X_train, all_features).

    Paths are resolved from config/evaluation_config.yaml:
      splits_dir / test_file  →  e.g. data/splits/uci_test.csv
      splits_dir / train_file →  e.g. data/splits/uci_train.csv

    X_train is only loaded for computing imputation means — never for model
    fitting in this module.
    """
    target_col = pp.target_col(TASK_KEY)

    # ── Test set ──
    test_path = pp.test_csv(TASK_KEY)   # raises clearly if missing
    df_test   = pd.read_csv(test_path)
    logger.info("[%s] Test CSV: %s  (%d rows)", model_name, test_path, len(df_test))

    if target_col not in df_test.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {test_path}.\n"
            f"  Columns: {df_test.columns.tolist()}"
        )
    y_test = df_test[target_col].values.astype(int)

    # ── Train set (for imputation means only) ──
    train_path = pp.train_csv_optional(TASK_KEY)
    if train_path is not None:
        df_train = pd.read_csv(train_path)
        logger.debug("[%s] Train CSV: %s  (%d rows)", model_name, train_path, len(df_train))
    else:
        logger.warning(
            "[%s] Train CSV not available — using test-set means for imputation "
            "(conservative; run train_test_split.py to restore train CSV).", model_name
        )
        df_train = df_test.copy()

    # ── Feature list ──
    feat_json = pp.selected_features_json(TASK_KEY, model_name)
    feat_data = _load_json(feat_json)
    if isinstance(feat_data, dict):
        features = feat_data.get("union_features", feat_data.get("features", []))
    elif isinstance(feat_data, list):
        features = feat_data
    else:
        features = []

    if not features:
        logger.warning(
            "[%s] selected_features.json missing — falling back to all numeric non-target columns.",
            model_name,
        )
        features = [
            c for c in df_test.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df_test[c])
        ]

    # Ensure all features exist in both DataFrames
    for f in features:
        if f not in df_test.columns:
            df_test[f]  = 0.0
        if f not in df_train.columns:
            df_train[f] = 0.0

    X_test  = df_test[features].fillna(0.0)
    X_train = df_train[features].fillna(0.0)
    logger.info("[%s] Feature matrix: %d rows × %d features", model_name, *X_test.shape)
    return X_test, y_test, X_train, features

# =============================================================================
# SHAP Feature Ranking
# =============================================================================

def _get_shap_ranking(
    pp,
    model_name: str,
    X_test: pd.DataFrame,
    model: Any,
    all_features: List[str],
) -> List[Tuple[str, float]]:
    """
    Return list of (feature_name, mean_abs_shap) sorted descending.

    Tries (in order):
      1. Pre-computed shap_values.csv from explainability.py
      2. Global SHAP importance CSV from explainability.py
      3. Compute SHAP on the fly (requires shap package)
      4. Model-native feature importance / coef_ (fallback)
    """
    explainability_dir = pp.explainability_dir(TASK_KEY)

    # 1. Per-model SHAP CSV
    shap_csv = explainability_dir / model_name / "shap_values.csv"
    if shap_csv.exists():
        logger.info("[%s] SHAP ranking: loading pre-computed values from %s", model_name, shap_csv)
        sv_df    = pd.read_csv(shap_csv)
        mean_abs = sv_df.abs().mean(axis=0)
        ranking  = sorted(zip(mean_abs.index.tolist(), mean_abs.tolist()), key=lambda x: -x[1])
        return [(f, v) for f, v in ranking if f in all_features]

    # 2. Global importance CSV
    global_csv = explainability_dir / "global_shap_importance.csv"
    if global_csv.exists():
        logger.info("[%s] SHAP ranking: loading global importance from %s", model_name, global_csv)
        df_imp = pd.read_csv(global_csv, index_col=0)
        df_imp = df_imp[df_imp.index.isin(all_features)]
        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False)
        return list(zip(df_imp.index.tolist(), df_imp["mean_abs_shap"].tolist()))

    # 3. Compute SHAP on the fly
    if _SHAP_OK:
        logger.info("[%s] SHAP ranking: computing on the fly...", model_name)
        try:
            from sklearn.calibration import CalibratedClassifierCV
            base = model
            if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
                base = model.calibrated_classifiers_[0].estimator

            if model_name in ("RandomForest", "XGBoost", "LightGBM", "CatBoost"):
                expl = _shap_mod.TreeExplainer(base, feature_perturbation="tree_path_dependent")
                sv   = expl.shap_values(X_test.values)
                if isinstance(sv, list) and len(sv) == 2:
                    sv = sv[1]
                elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                    sv = sv[:, :, 1]
            else:
                expl = _shap_mod.LinearExplainer(base, X_test.values)
                sv   = expl.shap_values(X_test.values)
                if isinstance(sv, list):
                    sv = sv[0]

            mean_abs = np.abs(sv).mean(axis=0)
            return sorted(zip(all_features, mean_abs.tolist()), key=lambda x: -x[1])
        except Exception as e:
            logger.warning("[%s] On-the-fly SHAP failed: %s — falling back to model importance.", model_name, e)

    # 4. Model-native importance
    return _fallback_importance(model, all_features)


def _fallback_importance(model: Any, all_features: List[str]) -> List[Tuple[str, float]]:
    """Use model-native feature importance as a last-resort ranking."""
    base = model
    try:
        from sklearn.calibration import CalibratedClassifierCV
        if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
    except Exception:
        pass

    if hasattr(base, "feature_importances_"):
        imp = base.feature_importances_
        n   = min(len(imp), len(all_features))
        return sorted(zip(all_features[:n], imp[:n].tolist()), key=lambda x: -x[1])
    if hasattr(base, "coef_"):
        imp = np.abs(base.coef_).flatten()
        n   = min(len(imp), len(all_features))
        return sorted(zip(all_features[:n], imp[:n].tolist()), key=lambda x: -x[1])

    logger.warning("[Fallback] No feature importance available — using equal weights.")
    return [(f, 1.0) for f in all_features]

# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        matthews_corrcoef, precision_score, recall_score,
        roc_auc_score, average_precision_score, brier_score_loss,
        confusion_matrix,
    )
    roc_auc = pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        pass
    try:
        pr_auc  = float(average_precision_score(y_true, y_proba))
    except Exception:
        pass

    tn = fp = fn = tp = 0
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        pass

    return {
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc,
        "f1":           float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "sensitivity":  float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":  float(tn / max(tn + fp, 1)),
        "precision":    float(precision_score(y_true, y_pred, zero_division=0)),
        "brier_score":  float(brier_score_loss(y_true, y_proba)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

# =============================================================================
# Ablation core
# =============================================================================

def _mask_features(
    X_test: pd.DataFrame,
    active: List[str],
    all_features: List[str],
    train_means: pd.Series,
) -> pd.DataFrame:
    """
    Return a copy of X_test where features NOT in `active` are replaced
    by their training-set mean. Column order is preserved.
    """
    X_masked = X_test.copy()
    for f in all_features:
        if f not in active:
            X_masked[f] = float(train_means.get(f, 0.0))
    return X_masked


def run_ablation_for_model(
    pp,
    model_name: str,
    output_dir: Path,
    subset_sizes: List[int],
    include_clinical_baseline: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float]]]:
    """
    Run ablation for one model.
    Returns (results_list, shap_ranking).
    """
    logger.info("[%s] Loading model and data...", model_name)

    try:
        model = _load_model(pp, model_name)
    except FileNotFoundError as e:
        logger.warning("[%s] Skipping — model not found: %s", model_name, e)
        return [], []

    try:
        X_test, y_test, X_train, all_features = _load_data(pp, model_name)
    except (FileNotFoundError, ValueError) as e:
        logger.warning("[%s] Skipping (data): %s", model_name, e)
        return [], []

    train_means = X_train.mean(axis=0)

    # ── SHAP ranking ──
    shap_ranking    = _get_shap_ranking(pp, model_name, X_test, model, all_features)
    ranked_features = [f for f, _ in shap_ranking]

    # Save ranking CSV
    rank_dir = output_dir / model_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shap_ranking, columns=["feature", "mean_abs_shap"]).assign(
        rank=range(1, len(shap_ranking) + 1)
    ).to_csv(rank_dir / "shap_feature_ranking.csv", index=False)

    # ── Define feature subsets ──
    subsets: List[Tuple[str, List[str]]] = [("All Features", all_features)]

    for k in sorted(subset_sizes, reverse=True):
        top_k = ranked_features[:k]
        if 0 < len(top_k) < len(all_features):
            subsets.append((f"Top {k} SHAP", top_k))

    if include_clinical_baseline:
        clinical = [f for f in CLINICAL_CANDIDATES if f in all_features]
        if clinical and set(clinical) != set(all_features):
            subsets.append(("Clinical Baseline", clinical))

    logger.info("[%s] Evaluating %d feature subsets...", model_name, len(subsets))
    results: List[Dict[str, Any]] = []

    for subset_label, active_feats in subsets:
        n = len(active_feats)
        logger.info("  [%s] %-25s  n_features=%d", model_name, subset_label, n)

        X_masked = _mask_features(X_test, active_feats, all_features, train_means)
        t0 = time.perf_counter()
        try:
            y_proba = model.predict_proba(X_masked.values)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)
        except Exception as e:
            logger.warning(
                "  [%s] predict_proba failed for subset '%s': %s",
                model_name, subset_label, e,
            )
            continue
        inference_s = time.perf_counter() - t0

        metrics = _compute_metrics(y_test, y_pred, y_proba)
        results.append({
            "model":           model_name,
            "subset":          subset_label,
            "n_features":      n,
            "active_features": active_feats,
            "inference_s":     round(inference_s, 4),
            **metrics,
        })
        logger.info(
            "    AUC=%.4f  F1=%.4f  MCC=%.4f  BalAcc=%.4f",
            metrics["roc_auc"], metrics["f1"], metrics["mcc"], metrics["balanced_acc"],
        )

    return results, shap_ranking

# =============================================================================
# Plotting
# =============================================================================

def plot_ablation_results(df: pd.DataFrame, output_dir: Path) -> None:
    if not _MPL_OK or df.empty:
        return

    models  = df["model"].unique().tolist()
    metrics = [
        ("roc_auc",      "ROC-AUC"),
        ("f1",           "F1 Score"),
        ("mcc",          "MCC"),
        ("balanced_acc", "Balanced Accuracy"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # ── Line plots per metric ──
    for metric_key, metric_label in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, mn in enumerate(models):
            df_m = df[df["model"] == mn].sort_values("n_features")
            ax.plot(
                df_m["n_features"], df_m[metric_key],
                marker="o", label=mn, color=colors[i], linewidth=2, markersize=6,
            )
            for _, row in df_m.iterrows():
                if row["subset"] != "All Features":
                    ax.annotate(
                        row["subset"], (row["n_features"], row[metric_key]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=colors[i], alpha=0.8,
                    )
        ax.set_xlabel("Number of Features", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(
            f"Ablation Study — {metric_label} vs. Feature Count\n"
            "(UCI CKD, mask-based ablation)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")
        lo = df[metric_key].dropna().min()
        hi = df[metric_key].dropna().max()
        ax.set_ylim(max(0.0, lo - 0.05), min(1.05, hi + 0.03))
        fig.tight_layout()
        _savefig(fig, output_dir / f"ablation_{metric_key}.png")

    # ── Heatmap for best model ──
    best_mn = (
        df.groupby("model")["roc_auc"].mean().idxmax()
        if len(models) > 1 else models[0]
    )
    df_best = df[df["model"] == best_mn].sort_values("n_features", ascending=False)
    hcols   = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_acc", "sensitivity", "specificity"]
    hdata   = df_best.set_index("subset")[hcols].astype(float)

    fig, ax = plt.subplots(figsize=(12, max(4, len(df_best) * 0.6 + 2)))
    im = ax.imshow(hdata.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(hcols)))
    ax.set_xticklabels(
        ["ROC-AUC", "PR-AUC", "F1", "MCC", "Bal.Acc", "Sensitivity", "Specificity"],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_yticks(range(len(hdata)))
    ax.set_yticklabels(hdata.index.tolist(), fontsize=9)
    ax.set_title(
        f"Ablation Heatmap — {best_mn}\n(mask-based, SHAP-ranked subsets)",
        fontsize=12, fontweight="bold",
    )
    for i in range(len(hdata)):
        for j in range(len(hcols)):
            val = hdata.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.85 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Metric Value")
    fig.tight_layout()
    _savefig(fig, output_dir / "ablation_heatmap.png")

    # ── Bar chart: ROC-AUC per subset (best model) ──
    df_bar = df_best.sort_values("n_features", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_bar)))
    bars = ax.bar(df_bar["subset"], df_bar["roc_auc"],
                  color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title(
        f"Ablation Study — ROC-AUC by Feature Subset\n({best_mn}, UCI CKD)",
        fontsize=12, fontweight="bold",
    )
    lo_bar = df_bar["roc_auc"].min()
    ax.set_ylim(max(0.0, lo_bar - 0.05), 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, df_bar["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    _savefig(fig, output_dir / "ablation_roc_auc_bar.png")

# =============================================================================
# Markdown Report
# =============================================================================

def generate_ablation_report(
    df: pd.DataFrame,
    output_dir: Path,
    shap_ranking: Optional[List[Tuple[str, float]]] = None,
) -> None:
    if df.empty:
        logger.warning("[Report] No results — skipping ablation report.")
        return

    out = output_dir / "ablation_report.md"
    lines: List[str] = [
        "# Ablation Study Report",
        "",
        "**Dataset:** UCI CKD  ",
        "**Approach:** Mask-based ablation — non-selected features replaced with training-set mean  ",
        "**Feature ranking:** SHAP importance (positive class)  ",
        "",
        "> This study evaluates whether a simpler model with fewer features",
        "> can achieve performance comparable to the full-feature model.",
        "",
        "---",
        "",
        "## Results by Model",
        "",
    ]

    for mn in df["model"].unique():
        df_m = df[df["model"] == mn].sort_values("n_features", ascending=False)
        lines += [
            f"### {mn}",
            "",
            "| Feature Subset | # Features | ROC-AUC | F1 | MCC | Bal. Acc | Sensitivity | Specificity |",
            "|----------------|-----------|---------|-----|-----|----------|-------------|-------------|",
        ]
        for _, row in df_m.iterrows():
            lines.append(
                f"| **{row['subset']}** | {row['n_features']} "
                f"| {row['roc_auc']:.4f} | {row['f1']:.4f} "
                f"| {row['mcc']:.4f} | {row['balanced_acc']:.4f} "
                f"| {row['sensitivity']:.4f} | {row['specificity']:.4f} |"
            )

        full = df_m[df_m["subset"] == "All Features"]
        if not full.empty:
            full_auc = float(full.iloc[0]["roc_auc"])
            lines += ["", "**Performance retention vs. All Features:**", ""]
            for _, row in df_m.iterrows():
                if row["subset"] == "All Features":
                    continue
                drop = full_auc - row["roc_auc"]
                ret  = row["roc_auc"] / full_auc * 100 if full_auc > 0 else 0
                sym  = "✅" if drop < 0.005 else ("⚠️" if drop < 0.02 else "❌")
                lines.append(
                    f"- **{row['subset']}**: retention={ret:.1f}% "
                    f"(Δ={drop:+.4f}) {sym}"
                )
        lines += ["", "---", ""]

    lines += [
        "## Interpretation",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| ✅ | ROC-AUC drop < 0.005 — effectively equivalent |",
        "| ⚠️ | ROC-AUC drop 0.005–0.020 — slight trade-off |",
        "| ❌ | ROC-AUC drop > 0.020 — meaningful performance loss |",
        "",
    ]

    if shap_ranking:
        lines += [
            "## Top 15 SHAP-Ranked Features",
            "",
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|----------|",
        ]
        for rank, (feat, imp) in enumerate(shap_ranking[:15], 1):
            lines.append(f"| {rank} | `{feat}` | {imp:.4f} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Methodology Note",
        "",
        "Non-selected features are replaced with their **training-set mean** before",
        "prediction. The model is not retrained. This is leakage-safe because the",
        "imputation values come from the training set only.",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("[Report] Ablation report → %s", out)

# =============================================================================
# Main orchestrator
# =============================================================================

def run_ablation(
    models_to_run:             List[str],
    output_dir:                Path,
    subset_sizes:              List[int],
    include_clinical_baseline: bool = True,
    config_path:               str  = "config/evaluation_config.yaml",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CKD ABLATION STUDY")
    logger.info("Models  : %s", models_to_run)
    logger.info("Subsets : Top %s SHAP features", subset_sizes)
    logger.info("Output  : %s", output_dir)
    logger.info("=" * 70)

    try:
        pp = _get_paths(config_path)
    except (FileNotFoundError, KeyError) as e:
        logger.error("Cannot load pipeline paths: %s", e)
        sys.exit(1)

    all_results:    List[Dict[str, Any]]        = []
    best_ranking:   Optional[List[Tuple[str, float]]] = None

    for model_name in models_to_run:
        logger.info("-" * 50)
        results, ranking = run_ablation_for_model(
            pp, model_name, output_dir, subset_sizes, include_clinical_baseline,
        )
        all_results.extend(results)
        if best_ranking is None and ranking:
            best_ranking = ranking

    if not all_results:
        logger.error(
            "No ablation results produced.\n"
            "  Verify that model artifacts exist in: %s\n"
            "  Verify that test CSV exists at: data/splits/uci_test.csv\n"
            "  Run model_training.py then train_test_split.py if missing.",
            pp.model_dir(TASK_KEY),
        )
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    save_cols = [c for c in df.columns if c != "active_features"]
    df[save_cols].to_csv(output_dir / "ablation_results.csv", index=False)
    logger.info("[Ablation] Results CSV → %s", output_dir / "ablation_results.csv")

    _save_json(all_results, output_dir / "ablation_results_detailed.json")

    logger.info("[Ablation] Generating plots...")
    plot_ablation_results(df, output_dir)

    logger.info("[Ablation] Generating Markdown report...")
    generate_ablation_report(df, output_dir, shap_ranking=best_ranking)

    # ── Console summary ──
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    for mn in df["model"].unique():
        df_m = df[df["model"] == mn].sort_values("n_features", ascending=False)
        full = df_m[df_m["subset"] == "All Features"]["roc_auc"].values
        if len(full):
            logger.info("\n  Model: %s  (Full AUC = %.4f)", mn, full[0])
            for _, row in df_m.iterrows():
                drop = full[0] - row["roc_auc"]
                logger.info(
                    "    %-25s  n=%3d  AUC=%.4f  F1=%.4f  Δ=%+.4f",
                    row["subset"], row["n_features"],
                    row["roc_auc"], row["f1"], drop,
                )

    logger.info("=" * 70)
    logger.info("ABLATION COMPLETE → %s", output_dir.resolve())
    return df

# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CKD Ablation Study — feature subset evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", default=PRIORITY_ORDER,
        help=f"Models to ablate (default: all). Choices: {UCI_MODELS}",
    )
    p.add_argument(
        "--subsets", nargs="+", type=int, default=[3, 5, 10, 15, 20],
        help="Top-N SHAP feature subset sizes (default: 3 5 10 15 20)",
    )
    p.add_argument(
        "--output-dir", default="artifacts/ablation",
        help="Output directory (default: artifacts/ablation)",
    )
    p.add_argument(
        "--no-clinical-baseline", action="store_true",
        help="Skip the clinical baseline feature subset",
    )
    p.add_argument(
        "--config", default="config/evaluation_config.yaml",
        help="Path to evaluation_config.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_ablation(
        models_to_run             = args.model,
        output_dir                = Path(args.output_dir),
        subset_sizes              = args.subsets,
        include_clinical_baseline = not args.no_clinical_baseline,
        config_path               = args.config,
    )


if __name__ == "__main__":
    main()
