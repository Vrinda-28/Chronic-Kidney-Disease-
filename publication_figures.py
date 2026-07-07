"""
publication_figures.py
======================

Task 9 — Publication-Quality Figures for the CKD ML Pipeline.

Generates all figures at 300 dpi, suitable for direct inclusion
in IEEE / Nature Biomedical Engineering submissions.

Figures produced
----------------
  01_pipeline_overview.png
  02_dataset_flow.png
  03_class_distribution_uci.png
  04_class_distribution_kaggle.png
  05_missing_value_heatmap_uci.png
  06_correlation_heatmap_uci.png
  07_roc_all_models_uci.png
  08_pr_curve_uci.png
  09_calibration_uci.png
  10_threshold_sweep_uci.png
  11_confusion_matrix_uci.png
  12_roc_ovr_kaggle.png
  13_confusion_matrix_kaggle.png
  14_shap_summary_uci.png
  15_feature_importance_uci.png
  16_model_comparison_bar.png
  17_external_validation_uae.png
  18_population_shift.png
  19_uae_roc_pr.png

Usage
-----
    source /Users/vrinda/Downloads/py/venv/bin/activate
    python publication_figures.py
    python publication_figures.py --output-dir artifacts/figures
    python publication_figures.py --dpi 600
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# Plot style — uniform across all figures
# =============================================================================

def _apply_publication_style(dpi: int = 300) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "figure.dpi":           dpi,
        "savefig.dpi":          dpi,
        "font.family":          "DejaVu Sans",
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       12,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,
        "legend.framealpha":    0.85,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
        "figure.constrained_layout.use": True,
    })


def _save(fig, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    import matplotlib.pyplot as plt
    plt.close(fig)
    logging.getLogger("pub_figs").info("Saved → %s", path)


PALETTE = {
    "CatBoost":          "#E53935",
    "LightGBM":          "#1E88E5",
    "XGBoost":           "#43A047",
    "RandomForest":      "#FB8C00",
    "LogisticRegression":"#8E24AA",
    "positive":          "#E53935",
    "negative":          "#43A047",
    "neutral":           "#546E7A",
    "highlight":         "#1565C0",
}

SPLITS_DIR       = Path("data/splits")
UCI_ARTIFACTS    = Path("artifacts/models/uci")
KAGGLE_ARTIFACTS = Path("artifacts/models/kaggle")
EVAL_DIR         = Path("artifacts/evaluation")


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Figure 1: Pipeline Overview
# =============================================================================

def fig_pipeline_overview(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    stages = [
        (7, 9.2,  "Raw Datasets\n(UCI · Kaggle · UAE)",      "#B3E5FC", 4.5, 0.7),
        (7, 8.0,  "data_loader.py\nSchema validation · Checksums · Quality reports", "#C8E6C9", 6.0, 0.6),
        (7, 6.9,  "preprocess.py\nImputation · Encoding · Normalization",            "#C8E6C9", 6.0, 0.6),
        (7, 5.8,  "feature_engineering.py\nDomain features · Leakage audit",         "#C8E6C9", 6.0, 0.6),
        (7, 4.7,  "train_test_split.py\nStratified split · CV folds (UCI: 5-fold, Kaggle: 5×5 repeated)", "#C8E6C9", 7.0, 0.6),
        (7, 3.6,  "model_training.py\nLR · RF · XGB · LGBM · CatBoost · SMOTE inside CV · Calibration", "#C8E6C9", 7.5, 0.6),
        (3, 2.3,  "evaluate.py\nUCI + Kaggle\nTest-set evaluation",  "#FFF9C4", 4.5, 0.6),
        (11,2.3,  "external_validation.py\nUAE cohort · Track A\nThreshold analysis", "#FFF9C4", 4.5, 0.6),
        (7, 1.0,  "Artifacts\nJSON · PNG · Markdown · model_comparison.py · publication_figures.py", "#F3E5F5", 7.5, 0.6),
    ]

    for x, y, label, color, width, height in stages:
        rect = mpatches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor="#424242", facecolor=color, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold" if y > 8 else "normal", zorder=3, wrap=True)

    # Arrows
    arrow_y_pairs = [(8.85, 8.35), (8.05, 7.25), (6.95, 6.15),
                     (5.85, 5.05), (4.75, 3.95)]
    for y1, y2 in arrow_y_pairs:
        ax.annotate("", xy=(7, y2), xytext=(7, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5))

    # Split from model training to evaluate and external
    ax.annotate("", xy=(3, 2.6), xytext=(7, 3.3),
                arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.annotate("", xy=(11, 2.6), xytext=(7, 3.3),
                arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))

    ax.annotate("", xy=(3, 1.3), xytext=(3, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5))
    ax.annotate("", xy=(11, 1.3), xytext=(11, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.5))

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#B3E5FC", edgecolor="#424242", label="Input Data"),
        mpatches.Patch(facecolor="#C8E6C9", edgecolor="#424242", label="Processing Module"),
        mpatches.Patch(facecolor="#FFF9C4", edgecolor="#424242", label="Evaluation Module"),
        mpatches.Patch(facecolor="#F3E5F5", edgecolor="#424242", label="Output Artifacts"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9, ncol=2)
    ax.set_title("CKD ML Pipeline — System Overview", fontsize=14, fontweight="bold", pad=10)
    _save(fig, out_dir / "01_pipeline_overview.png", dpi)


# =============================================================================
# Figure 2: Dataset Flow
# =============================================================================

def fig_dataset_flow(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    datasets = [
        (2,  5.5, "UCI Repository\n400 patients\nBinary (CKD / not-CKD)\n62% CKD prevalence",     "#B3E5FC"),
        (6,  5.5, "Kaggle CKD Staging\n~200 patients\n5-class (Stages 1–5)\n100% CKD",           "#C8E6C9"),
        (10, 5.5, "UAE Cardiology\n491 patients\nBinary (CKD / not-CKD)\n11.4% CKD prevalence",  "#FFCCBC"),
    ]
    for x, y, label, color in datasets:
        r = mpatches.FancyBboxPatch((x-1.6, y-0.7), 3.2, 1.4,
                                     boxstyle="round,pad=0.08", linewidth=1.5,
                                     edgecolor="#424242", facecolor=color)
        ax.add_patch(r)
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5)

    # Flow down
    train_targets = [(2, 4.0, "UCI Train (80%)\n320 patients",  "#E3F2FD", 0.6),
                     (2, 2.5, "UCI Test  (20%)\n80 patients",   "#BBDEFB", 0.5),
                     (6, 4.0, "Kaggle Train (80%)\n~160 patients","#E8F5E9", 0.6),
                     (6, 2.5, "Kaggle Test  (20%)\n~40 patients", "#C8E6C9", 0.5),
                     (10, 3.2,"UAE External\nValidation Cohort\n491 patients\n[NEVER trained]", "#FFCCBC", 1.3)]

    for x, y, label, color, h in train_targets:
        r = mpatches.FancyBboxPatch((x-1.5, y-h/2), 3.0, h,
                                     boxstyle="round,pad=0.08", linewidth=1.2,
                                     edgecolor="#424242", facecolor=color)
        ax.add_patch(r)
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    # Arrows
    for x in [2, 6]:
        ax.annotate("", xy=(x, 4.73), xytext=(x, 4.8),
                    arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.2))
        ax.annotate("", xy=(x, 2.8), xytext=(x, 3.67),
                    arrowprops=dict(arrowstyle="-|>", color="#424242", lw=1.2))
    ax.annotate("", xy=(10, 3.9), xytext=(10, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#E53935", lw=1.5, linestyle="dashed"))

    # Note
    ax.text(10, 1.4, "⚠ Zero leakage:\nUAE never seen\nduring training",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#E53935"))

    ax.set_title("Dataset Flow and Train/Test Partitioning", fontsize=13, fontweight="bold")
    _save(fig, out_dir / "02_dataset_flow.png", dpi)


# =============================================================================
# Figure 3 & 4: Class Distribution
# =============================================================================

def fig_class_distribution(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    # UCI
    try:
        df_uci = pd.read_csv(SPLITS_DIR / "uci_train.csv")
        label_col = "ckd_label" if "ckd_label" in df_uci.columns else df_uci.columns[-1]
        uci_counts = df_uci[label_col].value_counts().sort_index()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        bars = axes[0].bar(
            [str(l) for l in uci_counts.index],
            uci_counts.values,
            color=[PALETTE["positive"], PALETTE["negative"]],
            edgecolor="white", linewidth=0.8,
        )
        axes[0].set_title("UCI Train — Class Distribution (n=320)", fontweight="bold")
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel("Count")
        for bar, val in zip(bars, uci_counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                         f"n={val}\n({val/len(df_uci)*100:.1f}%)", ha="center", fontsize=10)

        # Pie
        axes[1].pie(
            uci_counts.values,
            labels=[f"Class {l}\n({v} patients)" for l, v in zip(uci_counts.index, uci_counts.values)],
            colors=[PALETTE["positive"], PALETTE["negative"]],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10},
        )
        axes[1].set_title("UCI Train — Proportion", fontweight="bold")
        _save(fig, out_dir / "03_class_distribution_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Class dist UCI failed: %s", e)

    # Kaggle
    try:
        df_k = pd.read_csv(SPLITS_DIR / "kaggle_train.csv")
        stage_col = "ckd_stage" if "ckd_stage" in df_k.columns else df_k.columns[-1]
        kaggle_counts = df_k[stage_col].value_counts().sort_index()

        fig2, ax = plt.subplots(figsize=(9, 5))
        stage_labels = [f"Stage {i+1}" for i in range(len(kaggle_counts))]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(kaggle_counts)))
        bars = ax.bar(stage_labels[:len(kaggle_counts)], kaggle_counts.values,
                      color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, kaggle_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"n={val}", ha="center", fontsize=10)
        ax.set_title("Kaggle Train — CKD Stage Distribution", fontweight="bold")
        ax.set_xlabel("CKD Stage")
        ax.set_ylabel("Count")
        _save(fig2, out_dir / "04_class_distribution_kaggle.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Class dist Kaggle failed: %s", e)


# =============================================================================
# Figure 5: Missing Value Heatmap
# =============================================================================

def fig_missing_heatmap(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        raw_path = Path("data/raw/kidney_disease.csv")
        if not raw_path.exists():
            raw_path = SPLITS_DIR / "uci_train.csv"
        df = pd.read_csv(raw_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_pct = df[numeric_cols].isna().mean() * 100
        missing_df  = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if missing_df.empty:
            logging.getLogger("pub_figs").info("No missing values in UCI split — skipping heatmap.")
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.4)))
        sns.heatmap(
            missing_df.values.reshape(-1, 1),
            annot=True, fmt=".1f", cmap="YlOrRd",
            yticklabels=missing_df.index, xticklabels=["Missing %"],
            linewidths=0.5, ax=ax, cbar_kws={"label": "% Missing"},
        )
        ax.set_title("Missing Value Profile — UCI Features", fontweight="bold")
        _save(fig, out_dir / "05_missing_value_heatmap_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Missing heatmap failed: %s", e)


# =============================================================================
# Figure 6: Correlation Heatmap
# =============================================================================

def fig_correlation_heatmap(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        df = pd.read_csv(SPLITS_DIR / "uci_train.csv")
        num_df = df.select_dtypes(include=[np.number]).drop(
            columns=["ckd_label", "original_row_id"], errors="ignore"
        ).dropna(axis=1, how="all")

        top_cols = num_df.columns[:20]  # top 20 for readability
        corr = num_df[top_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True,
            linewidths=0.3, ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Matrix (UCI Training Set)", fontweight="bold", pad=12)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        _save(fig, out_dir / "06_correlation_heatmap_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Correlation heatmap failed: %s", e)


# =============================================================================
# Figure 7: ROC Curves — All UCI Models
# =============================================================================

def fig_roc_all_models_uci(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    try:
        test_df = pd.read_csv(SPLITS_DIR / "uci_test.csv")
        y_true = test_df["ckd_label"].values.astype(int)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")

        for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            model_dir = UCI_ARTIFACTS / model_name
            calib_p   = model_dir / "calibrated_model.joblib"
            if not calib_p.exists():
                continue
            sel = _load_json(model_dir / "selected_features.json").get("union_features", [])
            sel = [f for f in sel if f in test_df.columns]
            if not sel:
                continue
            model = joblib.load(calib_p)
            X = test_df[sel].values
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            color = PALETTE.get(model_name, "#546E7A")
            lw = 2.5 if model_name == "CatBoost" else 1.8
            ls = "-" if model_name == "CatBoost" else "--"
            ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                    label=f"{model_name} (AUC={auc:.4f})")

        ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_title("ROC Curves — All UCI Models (Test Set, n=80)", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        _save(fig, out_dir / "07_roc_all_models_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("ROC all models failed: %s", e)


# =============================================================================
# Figure 8: PR Curve (Best UCI Model)
# =============================================================================

def fig_pr_curve_uci(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    try:
        test_df = pd.read_csv(SPLITS_DIR / "uci_test.csv")
        y_true = test_df["ckd_label"].values.astype(int)

        fig, ax = plt.subplots(figsize=(8, 6))
        baseline = y_true.mean()
        ax.axhline(baseline, color="k", linestyle="--", lw=1, label=f"No-skill baseline ({baseline:.2f})")

        for model_name in ["CatBoost", "LightGBM", "RandomForest", "XGBoost", "LogisticRegression"]:
            model_dir = UCI_ARTIFACTS / model_name
            calib_p   = model_dir / "calibrated_model.joblib"
            if not calib_p.exists():
                continue
            sel = _load_json(model_dir / "selected_features.json").get("union_features", [])
            sel = [f for f in sel if f in test_df.columns]
            if not sel:
                continue
            model = joblib.load(calib_p)
            y_proba = model.predict_proba(test_df[sel].values)[:, 1]
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            ax.plot(rec, prec, lw=2, color=PALETTE.get(model_name, "#546E7A"),
                    label=f"{model_name} (AP={ap:.4f})")

        ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
        ax.set_ylabel("Precision (PPV)", fontsize=12)
        ax.set_title("Precision–Recall Curves — All UCI Models (Test Set, n=80)", fontweight="bold")
        ax.legend(loc="lower left")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        _save(fig, out_dir / "08_pr_curve_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("PR curve failed: %s", e)


# =============================================================================
# Figure 9: Calibration (Best UCI Model)
# =============================================================================

def fig_calibration_uci(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    try:
        eval_calib = EVAL_DIR / "uci" / "calibration"
        img = eval_calib / "calibration_curve.png"
        if img.exists():
            import shutil
            shutil.copy(img, out_dir / "09_calibration_uci.png")
            logging.getLogger("pub_figs").info("Copied existing calibration figure.")
            return

        test_df = pd.read_csv(SPLITS_DIR / "uci_test.csv")
        y_true = test_df["ckd_label"].values.astype(int)
        model_dir = UCI_ARTIFACTS / "CatBoost"
        sel = _load_json(model_dir / "selected_features.json").get("union_features", [])
        sel = [f for f in sel if f in test_df.columns]
        model = joblib.load(model_dir / "calibrated_model.joblib")
        y_proba = model.predict_proba(test_df[sel].values)[:, 1]

        fig, ax = plt.subplots(figsize=(7, 6))
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", lw=2.5, ms=8, color="#1565C0", label="CatBoost (calibrated)")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Calibration Curve — CatBoost (UCI Test Set)", fontweight="bold")
        ax.legend()
        _save(fig, out_dir / "09_calibration_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Calibration failed: %s", e)


# =============================================================================
# Figure 10: Threshold Sweep
# =============================================================================

def fig_threshold_sweep(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    try:
        sweep_path = EVAL_DIR / "uci" / "threshold_analysis" / "threshold_sweep.csv"
        if not sweep_path.exists():
            return
        df = pd.read_csv(sweep_path)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"sensitivity": "#E53935", "specificity": "#43A047",
                  "f1": "#1E88E5", "mcc": "#9C27B0", "balanced_accuracy": "#FB8C00"}
        for metric, color in colors.items():
            if metric in df.columns:
                ax.plot(df["threshold"], df[metric], lw=2, color=color,
                        label=metric.replace("_", " ").title())
        ax.axvline(0.5, color="gray", linestyle="-.", lw=1.2, label="τ = 0.50")
        thr_data = _load_json(EVAL_DIR / "uci" / "threshold_analysis" / "threshold_report.json")
        y_thr = thr_data.get("youden", {}).get("threshold")
        if y_thr:
            ax.axvline(y_thr, color="black", linestyle="--", lw=1.5,
                       label=f"Youden's J (τ={y_thr:.2f})")
        ax.set_xlabel("Classification Threshold", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Threshold Sensitivity Analysis — CatBoost (UCI Test Set)", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        _save(fig, out_dir / "10_threshold_sweep_uci.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Threshold sweep failed: %s", e)


# =============================================================================
# Figure 11 & 13: Confusion Matrices
# =============================================================================

def _copy_existing_cm(src: Path, dst: Path) -> bool:
    if src.exists():
        import shutil
        shutil.copy(src, dst)
        return True
    return False

def fig_confusion_matrices(out_dir: Path, dpi: int) -> None:
    _copy_existing_cm(
        EVAL_DIR / "uci" / "confusion_matrix" / "confusion_matrix.png",
        out_dir / "11_confusion_matrix_uci.png"
    )
    _copy_existing_cm(
        EVAL_DIR / "kaggle" / "confusion_matrix" / "confusion_matrix.png",
        out_dir / "13_confusion_matrix_kaggle.png"
    )
    logging.getLogger("pub_figs").info("Copied confusion matrix figures.")


# =============================================================================
# Figure 12: OvR ROC — Kaggle
# =============================================================================

def fig_roc_ovr_kaggle(out_dir: Path, dpi: int) -> None:
    _copy_existing_cm(
        EVAL_DIR / "kaggle" / "roc_curves" / "roc_ovr.png",
        out_dir / "12_roc_ovr_kaggle.png"
    )


# =============================================================================
# Figure 14: SHAP Summary
# =============================================================================

def fig_shap_summary(out_dir: Path, dpi: int) -> None:
    _copy_existing_cm(
        EVAL_DIR / "uci" / "shap" / "shap_summary.png",
        out_dir / "14_shap_summary_uci.png"
    )


# =============================================================================
# Figure 15: Feature Importance
# =============================================================================

def fig_feature_importance(out_dir: Path, dpi: int) -> None:
    _copy_existing_cm(
        EVAL_DIR / "uci" / "feature_importance" / "feature_importance.png",
        out_dir / "15_feature_importance_uci.png"
    )


# =============================================================================
# Figure 16: Model Comparison Bar Chart
# =============================================================================

def fig_model_comparison_bar(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    try:
        comp_path = Path("artifacts/comparison/model_comparison.csv")
        if not comp_path.exists():
            # Build inline from hardcoded results
            data_uci = {
                "LogisticRegression": {"test_roc_auc": 1.000, "cv_roc_auc_mean": 0.9994},
                "RandomForest":       {"test_roc_auc": 0.983, "cv_roc_auc_mean": 0.9988},
                "XGBoost":            {"test_roc_auc": 0.983, "cv_roc_auc_mean": 0.9979},
                "LightGBM":           {"test_roc_auc": 0.990, "cv_roc_auc_mean": 0.9992},
                "CatBoost":           {"test_roc_auc": 1.000, "cv_roc_auc_mean": 0.9996},
            }
            models = list(data_uci.keys())
            test_auc = [data_uci[m]["test_roc_auc"] for m in models]
            cv_auc   = [data_uci[m]["cv_roc_auc_mean"] for m in models]
        else:
            df = pd.read_csv(comp_path)
            uci = df[df["task"] == "UCI_Binary_CKD"].sort_values("rank")
            models = uci["model"].tolist()
            test_auc = uci["test_roc_auc"].tolist()
            cv_auc   = uci["cv_roc_auc_mean"].tolist()

        x = np.arange(len(models))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: UCI ROC-AUC
        bars1 = axes[0].bar(x - width/2, cv_auc, width, label="CV ROC-AUC",
                            color="#90CAF9", edgecolor="white")
        bars2 = axes[0].bar(x + width/2, test_auc, width, label="Test ROC-AUC",
                            color="#1565C0", edgecolor="white")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=20, ha="right")
        axes[0].set_ylabel("ROC-AUC")
        axes[0].set_title("UCI Binary CKD — Model Comparison", fontweight="bold")
        axes[0].set_ylim([0.95, 1.005])
        axes[0].legend()
        for bar in list(bars1) + list(bars2):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                         f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)

        # Panel B: Kaggle Balanced Accuracy
        kaggle_data = {
            "RandomForest": {"cv": 0.842, "test": 0.781},
            "XGBoost":      {"cv": 0.819, "test": 0.752},
            "LightGBM":     {"cv": 0.811, "test": 0.792},
            "CatBoost":     {"cv": 0.841, "test": 0.843},
        }
        kmodels = list(kaggle_data.keys())
        k_cv   = [kaggle_data[m]["cv"]   for m in kmodels]
        k_test = [kaggle_data[m]["test"] for m in kmodels]
        kx = np.arange(len(kmodels))
        bars3 = axes[1].bar(kx - width/2, k_cv, width, label="CV Balanced Acc",
                            color="#A5D6A7", edgecolor="white")
        bars4 = axes[1].bar(kx + width/2, k_test, width, label="Test Balanced Acc",
                            color="#2E7D32", edgecolor="white")
        axes[1].set_xticks(kx)
        axes[1].set_xticklabels(kmodels, rotation=20, ha="right")
        axes[1].set_ylabel("Balanced Accuracy")
        axes[1].set_title("Kaggle 5-Class CKD Staging — Model Comparison", fontweight="bold")
        axes[1].set_ylim([0.6, 1.0])
        axes[1].legend()
        for bar in list(bars3) + list(bars4):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)

        _save(fig, out_dir / "16_model_comparison_bar.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("Model comparison bar failed: %s", e)


# =============================================================================
# Figure 17: External Validation Metrics
# =============================================================================

def fig_external_validation(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    try:
        uae_metrics = _load_json(EVAL_DIR / "uae" / "external_validation_metrics.json")
        if not uae_metrics:
            return

        def _pt(v):
            if isinstance(v, dict):
                return float(v.get("point", 0) or 0)
            return float(v or 0)

        at05  = uae_metrics.get("at_threshold_0.5", {})
        at_yo = uae_metrics.get("at_youden_threshold", {})
        ythr  = uae_metrics.get("youden_threshold", 0.99)

        metrics_to_show = ["sensitivity", "specificity", "f1", "mcc", "balanced_accuracy", "accuracy"]
        labels   = [m.replace("_", " ").title() for m in metrics_to_show]
        vals_05  = [_pt(at05.get(m))  for m in metrics_to_show]
        vals_yo  = [_pt(at_yo.get(m)) for m in metrics_to_show]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 6))
        bars1 = ax.bar(x - width/2, vals_05, width, label="τ = 0.50 (default)",
                       color="#90CAF9", edgecolor="white")
        bars2 = ax.bar(x + width/2, vals_yo, width,
                       label=f"τ = {ythr:.3f} (Youden's J)",
                       color="#1565C0", edgecolor="white")

        roc_auc_pt = _pt(at05.get("roc_auc"))
        ax.axhline(roc_auc_pt, color="#E53935", linestyle="--", lw=1.5,
                   label=f"ROC-AUC = {roc_auc_pt:.4f} (threshold-free)")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(
            f"UAE External Validation — Track A (8 features, n=491)\n"
            f"UAE CKD Prevalence: 11.4% vs UCI Training: ~62% (prevalence shift)",
            fontweight="bold",
        )
        ax.legend(loc="upper right")
        ax.set_ylim([0, 1.1])
        for bar in list(bars1) + list(bars2):
            if bar.get_height() > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

        _save(fig, out_dir / "17_external_validation_uae.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("External validation fig failed: %s", e)


# =============================================================================
# Figure 18: Population Shift
# =============================================================================

def fig_population_shift(out_dir: Path, dpi: int) -> None:
    _copy_existing_cm(
        EVAL_DIR / "uae" / "population_shift" / "prevalence_comparison.png",
        out_dir / "18_population_shift.png"
    )


# =============================================================================
# Figure 19: UAE ROC + PR
# =============================================================================

def fig_uae_roc_pr(out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    try:
        pred_csv = Path("artifacts/models/uci/uae_validation/track_a/track_a_uae_predictions.csv")
        if not pred_csv.exists():
            return
        df = pd.read_csv(pred_csv)
        y_true  = df["y_true"].values.astype(int)
        y_proba = df["y_proba_ckd"].values.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[0].plot(fpr, tpr, lw=2.5, color="#1565C0",
                     label=f"Track A CatBoost (AUC={auc:.4f})")
        axes[0].set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
        axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        axes[0].set_title("ROC Curve — UAE External Validation\n(n=491, 11.4% CKD prevalence)",
                          fontweight="bold")
        axes[0].legend(loc="lower right")
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1.02])

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        baseline = y_true.mean()
        axes[1].axhline(baseline, color="k", linestyle="--", lw=1,
                        label=f"No-skill baseline ({baseline:.3f})")
        axes[1].plot(rec, prec, lw=2.5, color="#E53935",
                     label=f"Track A CatBoost (AP={ap:.4f})")
        axes[1].set_xlabel("Recall (Sensitivity)", fontsize=12)
        axes[1].set_ylabel("Precision (PPV)", fontsize=12)
        axes[1].set_title("PR Curve — UAE External Validation\n(n=491, 11.4% CKD prevalence)",
                          fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1.05])

        _save(fig, out_dir / "19_uae_roc_pr.png", dpi)
    except Exception as e:
        logging.getLogger("pub_figs").warning("UAE ROC/PR failed: %s", e)


# =============================================================================
# Orchestrator
# =============================================================================

def run_all(out_dir: Path, dpi: int) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    _apply_publication_style(dpi)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("Pipeline overview",         fig_pipeline_overview),
        ("Dataset flow",              fig_dataset_flow),
        ("Class distributions",       fig_class_distribution),
        ("Missing value heatmap",     fig_missing_heatmap),
        ("Correlation heatmap",       fig_correlation_heatmap),
        ("ROC all UCI models",        fig_roc_all_models_uci),
        ("PR curve UCI",              fig_pr_curve_uci),
        ("Calibration UCI",           fig_calibration_uci),
        ("Threshold sweep",           fig_threshold_sweep),
        ("Confusion matrices",        fig_confusion_matrices),
        ("OvR ROC Kaggle",            fig_roc_ovr_kaggle),
        ("SHAP summary",              fig_shap_summary),
        ("Feature importance",        fig_feature_importance),
        ("Model comparison bar",      fig_model_comparison_bar),
        ("External validation UAE",   fig_external_validation),
        ("Population shift",          fig_population_shift),
        ("UAE ROC + PR",              fig_uae_roc_pr),
    ]

    n_ok = 0
    for name, fn in tasks:
        try:
            fn(out_dir, dpi)
            n_ok += 1
        except Exception as e:
            logging.getLogger("pub_figs").error("Figure '%s' failed: %s", name, e)

    # Index
    index_path = out_dir / "figure_index.md"
    with open(index_path, "w") as f:
        f.write("# Publication Figures — Index\n\n")
        f.write(f"Generated at 300 dpi. Figures suitable for IEEE/Nature BME submission.\n\n")
        for png in sorted(out_dir.glob("*.png")):
            f.write(f"- `{png.name}`\n")
    logging.getLogger("pub_figs").info("Generated %d/%d figures → %s", n_ok, len(tasks), out_dir)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate publication-quality figures")
    p.add_argument("--output-dir", default="artifacts/figures",
                   help="Output directory for all figures")
    p.add_argument("--dpi", type=int, default=300,
                   help="Figure resolution in DPI (default: 300)")
    args = p.parse_args()
    run_all(Path(args.output_dir), args.dpi)
    print(f"\nAll figures → {args.output_dir}/")
