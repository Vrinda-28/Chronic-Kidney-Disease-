"""
explainability.py
=================

Dedicated explainability and interpretability module for the CKD ML Pipeline.

Pipeline position:
  preprocess.py → feature_engineering.py → train_test_split.py
  → model_training.py → evaluate.py → external_validation.py
  → explainability.py  (THIS FILE)   ← you are here
  → ablation_study.py

All paths are resolved through pipeline_paths.PipelinePaths which reads
config/evaluation_config.yaml — the same config used by evaluate.py.
Nothing is hardcoded.

══════════════════════════════════════════════════════════════════════════
WHAT THIS MODULE PROVIDES
══════════════════════════════════════════════════════════════════════════

  1. Global SHAP summary
       Mean |SHAP| bar chart aggregated across all UCI models.

  2. SHAP beeswarm (per model)
       Per-model beeswarm plot with feature-value color coding.

  3. SHAP dependence plots
       Top-N features: SHAP value vs. raw feature value scatter.

  4. Patient-level waterfall plots
       For each model: best correctly predicted CKD, best correctly
       predicted notCKD, worst false negative, worst false positive.

  5. Cross-model agreement analysis
       Heatmap: for each test patient, do all models agree?
       CSV + summary JSON saved.

  6. LIME explanations (optional, --no-lime to skip)
       Most confident CKD, most confident notCKD, boundary case.

  7. Clinician-friendly Markdown report
       Plain-English description of the top-3 SHAP drivers per patient.

══════════════════════════════════════════════════════════════════════════
SHAP COMPATIBILITY LAYER
══════════════════════════════════════════════════════════════════════════

  All SHAP computation flows through compute_shap() which returns a
  SHAPResult dataclass.  SHAPResult.values is ALWAYS a 2-D NumPy array
  of shape (n_samples, n_features) for the positive class.  No plotting
  function ever receives a raw SHAP output — they only consume SHAPResult.

  Supported output formats from raw SHAP (all normalised internally):

    • ndarray  (n_samples, n_features)            – standard binary
    • ndarray  (n_samples, n_features, n_classes) – KernelExplainer / old API
    • ndarray  (n_classes, n_samples, n_features) – some SHAP versions
    • list of ndarray, one per class              – old TreeExplainer API
    • shap.Explanation object                     – new SHAP API

  KernelExplainer is always called with
      lambda x: model.predict_proba(x)[:, positive_class]
  so its output is scalar-per-sample → (n_samples, n_features) directly.
  This eliminates the 3-D format entirely from the KernelExplainer path.

══════════════════════════════════════════════════════════════════════════
NO LEAKAGE
══════════════════════════════════════════════════════════════════════════

  • Loads only calibrated_model.joblib (final_model.joblib as fallback).
  • Loads only the held-out test set (never the train set for fitting).
  • Training data loaded ONLY as SHAP/LIME reference background —
    no model parameters are changed.
  • SHAP and LIME are purely post-hoc explanation methods.

Usage
-----
    python explainability.py
    python explainability.py --model CatBoost
    python explainability.py --no-lime
    python explainability.py --patients 3
    python explainability.py --output-dir artifacts/explainability/uci
    python explainability.py --config config/evaluation_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Logging
# =============================================================================

def _build_logger(log_dir: str = "logs", log_file: str = "explainability.log") -> logging.Logger:
    logger = logging.getLogger("ckd_explainability")
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

UCI_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
TASK_KEY   = "uci"

# =============================================================================
# Optional dependency guards
# =============================================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.error("matplotlib not installed. Run: pip install matplotlib")

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    logger.warning("shap not installed — SHAP plots skipped. Run: pip install shap")

# =============================================================================
# Path resolver
# =============================================================================

def _get_paths(config_path: str = "config/evaluation_config.yaml"):
    from pipeline_paths import PipelinePaths
    return PipelinePaths(config_path)

# =============================================================================
# JSON / file helpers
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
        logger.info("[Load] %s ← calibrated_model.joblib", model_name)
        return joblib.load(calib)
    if final.exists():
        logger.warning("[Load] %s ← final_model.joblib (calibrated not found)", model_name)
        return joblib.load(final)
    raise FileNotFoundError(
        f"No model artifact found for '{model_name}'.\n"
        f"  Checked: {calib}\n"
        f"  Checked: {final}\n"
        f"  Run model_training.py first."
    )

# =============================================================================
# Data loading — paths from evaluation_config.yaml
# =============================================================================

def _load_test_data(pp, model_name: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load the held-out test CSV and the union feature list for this model.

    Paths resolved from config/evaluation_config.yaml:
      splits_dir / test_file  →  e.g. data/splits/uci_test.csv
      target_col              →  e.g. ckd_label
    """
    test_path  = pp.test_csv(TASK_KEY)
    target_col = pp.target_col(TASK_KEY)

    df = pd.read_csv(test_path)
    logger.info("[Data] Test CSV loaded: %s  (%d rows, %d cols)",
                test_path, len(df), len(df.columns))

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {test_path}.\n"
            f"  Columns present: {df.columns.tolist()}"
        )
    y_test = df[target_col].values.astype(int)

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
            "[%s] selected_features.json missing or empty — "
            "falling back to all numeric non-target columns.", model_name,
        )
        features = [
            c for c in df.columns
            if c != target_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing   = [f for f in features if f not in df.columns]
    available = [f for f in features if f in df.columns]
    if missing:
        logger.warning(
            "[%s] %d feature(s) absent from test CSV — zero-filled: %s",
            model_name, len(missing), missing,
        )
        for f in missing:
            df[f] = 0.0

    feat_names = available + missing
    X_test = df[feat_names].fillna(0.0)
    logger.info("[%s] Feature matrix: %d rows × %d features", model_name, *X_test.shape)
    return X_test, y_test, feat_names


def _load_train_background(pp, model_name: str, max_samples: int = 100) -> Optional[pd.DataFrame]:
    """
    Load a sample from the training CSV for SHAP/LIME background.
    Returns None gracefully if the train file is absent.
    """
    train_path = pp.train_csv_optional(TASK_KEY)
    if train_path is None:
        logger.warning("[%s] Train CSV not found — will use test data as background.", model_name)
        return None

    target_col = pp.target_col(TASK_KEY)
    df         = pd.read_csv(train_path)

    feat_json = pp.selected_features_json(TASK_KEY, model_name)
    feat_data = _load_json(feat_json)
    if isinstance(feat_data, dict):
        features = feat_data.get("union_features", feat_data.get("features", []))
    elif isinstance(feat_data, list):
        features = feat_data
    else:
        features = [c for c in df.columns if c != target_col]

    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    bg = df[features].fillna(0.0)
    if len(bg) > max_samples:
        bg = bg.sample(max_samples, random_state=42)
    logger.debug("[%s] Background: %d rows from %s", model_name, len(bg), train_path)
    return bg


# =============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                    SHAP COMPATIBILITY LAYER                             ║
# ║                                                                         ║
# ║  Everything below this header is the redesigned SHAP layer.            ║
# ║  The public contract is simple:                                         ║
# ║                                                                         ║
# ║      result = compute_shap(model, model_name, X, background,           ║
# ║                            feature_names)                               ║
# ║                                                                         ║
# ║  result.values  is ALWAYS  ndarray (n_samples, n_features), float64.   ║
# ║  result.top_feature_indices(n)  returns  List[int]  (Python ints).     ║
# ║  No plotting function ever touches raw SHAP output.                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# =============================================================================


# ---------------------------------------------------------------------------
# SHAPResult — the ONE canonical representation
# ---------------------------------------------------------------------------

@dataclass
class SHAPResult:
    """
    Standardised SHAP output container.

    Invariants enforced in __post_init__:
      • values.ndim == 2
      • values.shape == (n_samples, n_features)
      • values.shape[1] == len(feature_names)
      • expected_value is a Python float
    """

    values:         np.ndarray   # (n_samples, n_features), float64, positive class
    expected_value: float        # scalar base value for the positive class
    feature_names:  List[str]    # length == n_features
    model_name:     str

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 2:
            raise ValueError(
                f"[SHAPResult] values must be 2-D, got shape {self.values.shape}"
            )
        if self.values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"[SHAPResult] values has {self.values.shape[1]} columns "
                f"but feature_names has {len(self.feature_names)} entries."
            )
        self.expected_value = float(self.expected_value)

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    def mean_abs(self) -> np.ndarray:
        """
        1-D array of mean |SHAP| per feature.
        shape: (n_features,)  — always 1-D regardless of input dimensions.
        """
        ma = np.abs(self.values).mean(axis=0)
        assert ma.ndim == 1, f"mean_abs produced {ma.ndim}-D array — BUG"
        return ma

    def top_feature_indices(self, n: int) -> List[int]:
        """
        Python int list of up to n feature indices sorted by mean |SHAP|,
        descending.  Returns Python ints, never numpy integers, so they are
        safe for all list/DataFrame indexing operations.
        """
        n = min(n, self.n_features)
        return [int(i) for i in np.argsort(self.mean_abs())[::-1][:n]]

    def patient_shap(self, idx: int) -> np.ndarray:
        """1-D SHAP value array for patient at row idx. Shape: (n_features,)."""
        row = self.values[idx]
        if row.ndim != 1:
            raise ValueError(f"patient_shap returned {row.ndim}-D array — BUG")
        return row


# ---------------------------------------------------------------------------
# Model unwrapping — handles CalibratedClassifierCV + FrozenEstimator
# ---------------------------------------------------------------------------

def _unwrap_model(model: Any) -> Any:
    """
    Unwrap sklearn wrappers to reach the actual base estimator.

    Handles (in order):
      1. CalibratedClassifierCV  →  estimator inside calibrated_classifiers_
      2. FrozenEstimator          →  .estimator  (sklearn internal wrapper)
      3. Any other object         →  returned as-is

    FrozenEstimator is an sklearn-internal class used inside CalibratedClassifierCV
    to prevent re-fitting.  SHAP explainers do not recognise it, so we must
    unwrap it before passing to TreeExplainer / LinearExplainer.
    """
    current = model

    # Step 1 — CalibratedClassifierCV
    if hasattr(current, "calibrated_classifiers_"):
        # sklearn ≥ 1.2 stores a list of (estimator, calibrator) pairs
        inner = current.calibrated_classifiers_[0]
        # inner may be a _CalibratedClassifier namedtuple-like with .estimator
        current = getattr(inner, "estimator", inner)

    # Legacy attribute name
    if hasattr(current, "base_estimator") and not hasattr(current, "fit"):
        current = current.base_estimator

    # Step 2 — FrozenEstimator (sklearn.frozen._frozen.FrozenEstimator)
    if type(current).__name__ == "FrozenEstimator":
        current = getattr(current, "estimator", current)

    # Step 3 — Some calibrators store the model as .clf
    if type(current).__name__ in ("_SigmoidCalibration",):
        current = getattr(current, "clf", current)

    return current


# ---------------------------------------------------------------------------
# _normalize_shap_raw — the universal format adapter
# ---------------------------------------------------------------------------

def _normalize_shap_raw(
    raw: Any,
    n_features: int,
    positive_class: int = 1,
) -> np.ndarray:
    """
    Convert ANY SHAP raw output → float64 ndarray of shape (n_samples, n_features).

    This is the only place that needs to understand SHAP's output formats.
    All callers receive a clean 2-D array.

    Handled formats
    ───────────────
    ┌──────────────────────────────────────────┬───────────────────────────────────┐
    │ Raw format                                │ Action                            │
    ├──────────────────────────────────────────┼───────────────────────────────────┤
    │ shap.Explanation                          │ extract .values, recurse          │
    │ list[ndarray], len == n_classes           │ take [positive_class]             │
    │ list[ndarray], len == 1                   │ take [0]                          │
    │ ndarray (n_samples, n_features)           │ return as-is                      │
    │ ndarray (n_samples, n_features, n_classes)│ slice [:, :, positive_class]     │
    │ ndarray (n_classes, n_samples, n_features)│ slice [positive_class]           │
    └──────────────────────────────────────────┴───────────────────────────────────┘

    Raises
    ------
    ValueError  if no interpretation yields shape (n_samples, n_features).
    """
    # ── Unwrap shap.Explanation ──────────────────────────────────────────────
    if hasattr(raw, "values"):
        return _normalize_shap_raw(raw.values, n_features, positive_class)

    # ── Unwrap list (old API: one array per class) ───────────────────────────
    if isinstance(raw, list):
        if len(raw) == 0:
            raise ValueError("[_normalize_shap_raw] Empty list returned by SHAP.")
        idx = positive_class if positive_class < len(raw) else len(raw) - 1
        candidate = np.asarray(raw[idx], dtype=np.float64)
        if candidate.ndim == 2 and candidate.shape[1] == n_features:
            return candidate
        # Unexpected shape — still recurse once for nested lists
        return _normalize_shap_raw(candidate, n_features, positive_class)

    # ── Handle NumPy arrays ──────────────────────────────────────────────────
    arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 2:
        # Standard: (n_samples, n_features)
        if arr.shape[1] == n_features:
            return arr
        raise ValueError(
            f"[_normalize_shap_raw] 2-D SHAP array has shape {arr.shape} "
            f"but n_features={n_features}. Cannot interpret."
        )

    if arr.ndim == 3:
        n0, n1, n2 = arr.shape
        # Case (a): (n_samples, n_features, n_classes)
        if n1 == n_features:
            cls_idx = min(positive_class, n2 - 1)
            return arr[:, :, cls_idx]
        # Case (b): (n_classes, n_samples, n_features)
        if n2 == n_features:
            cls_idx = min(positive_class, n0 - 1)
            return arr[cls_idx]
        raise ValueError(
            f"[_normalize_shap_raw] 3-D SHAP array has shape {arr.shape}. "
            f"Neither dim-1 ({n1}) nor dim-2 ({n2}) matches n_features={n_features}."
        )

    raise ValueError(
        f"[_normalize_shap_raw] Cannot interpret SHAP output: "
        f"ndim={arr.ndim}, shape={getattr(arr, 'shape', '?')}."
    )


# ---------------------------------------------------------------------------
# _normalize_expected_value — extract scalar base value
# ---------------------------------------------------------------------------

def _normalize_expected_value(
    explainer: Any,
    raw_shap: Any,
    positive_class: int = 1,
    fallback: float = 0.0,
) -> float:
    """
    Extract a scalar expected (base) value for the positive class.

    Tries (in order):
      1. explainer.expected_value
      2. raw_shap.base_values  (shap.Explanation attribute)
      3. fallback
    """
    ev = None

    # Source 1 — explainer object
    if hasattr(explainer, "expected_value"):
        ev = explainer.expected_value

    # Source 2 — Explanation.base_values
    if ev is None and hasattr(raw_shap, "base_values"):
        bv = raw_shap.base_values
        ev = float(bv.flat[0]) if hasattr(bv, "flat") else bv

    if ev is None:
        return fallback

    # Scalar
    if np.isscalar(ev):
        return float(ev)

    # Array-like: pick the positive-class entry
    ev_arr = np.asarray(ev, dtype=np.float64).flatten()
    if ev_arr.size == 0:
        return fallback
    if ev_arr.size > positive_class:
        return float(ev_arr[positive_class])
    return float(ev_arr[-1])


# ---------------------------------------------------------------------------
# compute_shap — the single public entry point
# ---------------------------------------------------------------------------

def compute_shap(
    model:        Any,
    model_name:   str,
    X:            pd.DataFrame,
    background:   Optional[pd.DataFrame],
    feature_names: List[str],
    positive_class: int = 1,
) -> SHAPResult:
    """
    Compute SHAP values and return a validated SHAPResult.

    Explainer selection strategy
    ────────────────────────────
    1. TreeExplainer  — RF, XGB, LGB, CatBoost.  Uses the unwrapped base
       estimator so FrozenEstimator never reaches SHAP.
    2. LinearExplainer — LogisticRegression (requires coef_ attribute).
    3. KernelExplainer — universal fallback.  Crucially, we pass
           lambda x: model.predict_proba(x)[:, positive_class]
       so SHAP receives a scalar output per sample → always produces
       (n_samples, n_features), never (n_samples, n_features, n_classes).

    After raw SHAP values are obtained, _normalize_shap_raw() converts any
    remaining format differences to a clean 2-D array, which is validated
    inside SHAPResult.__post_init__.
    """
    if not _SHAP_OK:
        raise ImportError("shap not installed. Run: pip install shap")

    n_features  = len(feature_names)
    X_arr       = X.values                          # (n_samples, n_features)
    base        = _unwrap_model(model)              # actual estimator
    raw_shap    = None
    explainer   = None

    # ── Strategy 1: TreeExplainer ──────────────────────────────────────────
    if model_name in ("RandomForest", "XGBoost", "LightGBM", "CatBoost"):
        logger.debug("[%s] Trying TreeExplainer on %s", model_name, type(base).__name__)
        try:
            explainer = shap.TreeExplainer(
                base,
                feature_perturbation="tree_path_dependent",
            )
            raw_shap = explainer.shap_values(X_arr)
            logger.debug("[%s] TreeExplainer succeeded.", model_name)
        except Exception as exc:
            logger.warning(
                "[%s] TreeExplainer failed (%s). Will try KernelExplainer.",
                model_name, exc,
            )
            explainer = None
            raw_shap  = None

    # ── Strategy 2: LinearExplainer ────────────────────────────────────────
    if raw_shap is None and model_name == "LogisticRegression":
        logger.debug("[%s] Trying LinearExplainer on %s", model_name, type(base).__name__)
        try:
            if not hasattr(base, "coef_"):
                raise AttributeError("Base estimator has no coef_ — not a fitted LR.")
            bg_arr   = background.values if background is not None else X_arr
            explainer = shap.LinearExplainer(
                base, bg_arr,
                feature_perturbation="correlation_dependent",
            )
            raw_shap = explainer.shap_values(X_arr)
            logger.debug("[%s] LinearExplainer succeeded.", model_name)
        except Exception as exc:
            logger.warning(
                "[%s] LinearExplainer failed (%s). Will use KernelExplainer.",
                model_name, exc,
            )
            explainer = None
            raw_shap  = None

    # ── Strategy 3: KernelExplainer (universal fallback) ───────────────────
    if raw_shap is None:
        logger.info(
            "[%s] Using KernelExplainer (slower). "
            "Tip: run explainability.py after model_training.py to enable TreeExplainer.",
            model_name,
        )
        bg_data = (
            background.values
            if background is not None
            else X_arr[: min(50, len(X_arr))]
        )
        # KEY: pass a scalar predict function so output is (n_samples, n_features),
        # never (n_samples, n_features, n_classes).
        def _predict_positive(x: np.ndarray) -> np.ndarray:
            return model.predict_proba(x)[:, positive_class]

        explainer = shap.KernelExplainer(_predict_positive, bg_data)
        raw_shap  = explainer.shap_values(X_arr, nsamples=100)

    # ── Normalise to 2-D ───────────────────────────────────────────────────
    values_2d = _normalize_shap_raw(raw_shap, n_features, positive_class)

    # ── Extract scalar expected value ──────────────────────────────────────
    expected   = _normalize_expected_value(
        explainer, raw_shap, positive_class, fallback=float(np.mean(values_2d))
    )

    result = SHAPResult(
        values        = values_2d,
        expected_value = expected,
        feature_names  = list(feature_names),
        model_name     = model_name,
    )
    logger.info(
        "[%s] SHAP computed — shape %s, E[f(x)] = %.4f",
        model_name, result.values.shape, result.expected_value,
    )
    return result


# =============================================================================
# ── PLOTTING ─────────────────────────────────────────────────────────────────
# All plotting functions accept SHAPResult.  They never receive raw SHAP output.
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Global SHAP Summary (cross-model consensus)
# ---------------------------------------------------------------------------

def plot_global_shap_summary(
    all_results: Dict[str, "SHAPResult"],
    output_dir:  Path,
) -> None:
    """Mean |SHAP| bar chart averaged across all successfully processed models."""
    if not _MPL_OK or not all_results:
        return

    feature_accum: Dict[str, List[float]] = {}
    for result in all_results.values():
        ma = result.mean_abs()                              # always 1-D
        for feat, val in zip(result.feature_names, ma):
            feature_accum.setdefault(feat, []).append(float(val))

    global_imp = {f: float(np.mean(v)) for f, v in feature_accum.items()}
    df_imp = (
        pd.DataFrame.from_dict(global_imp, orient="index", columns=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df_imp)))
    bars    = ax.barh(
        df_imp.index, df_imp["mean_abs_shap"],
        color=colors[::-1], edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP Value| (averaged across all models)", fontsize=12)
    ax.set_title(
        "Global Feature Importance — Consensus Across All Models\n"
        "(CKD Prediction, UCI Dataset)",
        fontsize=13, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", labelsize=9)

    n_models = len(all_results)
    coverage  = {f: len(v) for f, v in feature_accum.items() if f in df_imp.index}
    for bar, feat in zip(bars, df_imp.index):
        cov = coverage.get(feat, 0)
        if cov < n_models:
            ax.text(
                bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"({cov}/{n_models})", va="center", fontsize=7, color="gray",
            )
    fig.tight_layout()
    _savefig(fig, output_dir / "global_shap_summary.png")

    # Importance CSV
    df_full = pd.DataFrame.from_dict(global_imp, orient="index", columns=["mean_abs_shap"])
    df_full["n_models"] = [len(feature_accum[f]) for f in df_full.index]
    df_full.sort_values("mean_abs_shap", ascending=False).to_csv(
        output_dir / "global_shap_importance.csv"
    )
    logger.info("[Global SHAP] CSV → %s", output_dir / "global_shap_importance.csv")


# ---------------------------------------------------------------------------
# 2. SHAP Beeswarm (per model)
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    result:       "SHAPResult",
    X:            pd.DataFrame,
    output_dir:   Path,
    max_features: int = 20,
) -> None:
    """Per-model SHAP beeswarm / dot summary plot."""
    if not _SHAP_OK or not _MPL_OK:
        return

    out = output_dir / result.model_name / "shap_beeswarm.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # top indices are Python ints — safe for all indexing
    top_idx = result.top_feature_indices(max_features)
    sv_top  = result.values[:, top_idx]   # (n_samples, len(top_idx))
    X_top   = X.iloc[:, top_idx]

    try:
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_idx) * 0.4)))
        plt.sca(ax)
        shap.summary_plot(
            sv_top, X_top,
            plot_type="dot", max_display=max_features, show=False, color_bar=True,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"SHAP Beeswarm — {result.model_name}\n(UCI CKD Test Set)",
            fontsize=12, fontweight="bold", y=1.01,
        )
        _savefig(fig, out)
    except Exception as exc:
        logger.warning("[Beeswarm] %s: %s", result.model_name, exc)


# ---------------------------------------------------------------------------
# 3. SHAP Dependence Plots
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    result:    "SHAPResult",
    X:         pd.DataFrame,
    output_dir: Path,
    n_top:     int = 5,
) -> None:
    """
    SHAP dependence plot for top-n features.

    Uses result.values which is always 2-D (n_samples, n_features).
    All indices are Python ints — no numpy-scalar indexing issues.
    shap.dependence_plot receives:
      • ind      : Python int
      • shap_values : 2-D ndarray (n_samples, n_features)
      • features : 2-D ndarray  (n_samples, n_features)
    """
    if not _SHAP_OK or not _MPL_OK:
        return

    dep_dir = output_dir / result.model_name / "dependence_plots"
    dep_dir.mkdir(parents=True, exist_ok=True)

    top_indices = result.top_feature_indices(n_top)   # List[int]
    sv_2d       = result.values                        # (n_samples, n_features) — 2-D
    X_arr       = X.values                            # (n_samples, n_features)
    feat_names  = result.feature_names

    # Validate alignment
    if sv_2d.shape[1] != X_arr.shape[1]:
        logger.warning(
            "[Dependence] %s: SHAP columns (%d) ≠ X columns (%d). Skipping.",
            result.model_name, sv_2d.shape[1], X_arr.shape[1],
        )
        return

    for rank, feat_idx in enumerate(top_indices, 1):
        feat = feat_names[feat_idx]             # feat_idx is a Python int — safe
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                ind          = feat_idx,        # Python int ✓
                shap_values  = sv_2d,           # 2-D ndarray ✓
                features     = X_arr,           # 2-D ndarray ✓
                feature_names = feat_names,
                ax           = ax,
                show         = False,
                alpha        = 0.7,
            )
            ax.set_title(
                f"SHAP Dependence — {feat}\n({result.model_name}, rank #{rank})",
                fontsize=11, fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)
            safe_name = feat.replace("/", "_").replace(" ", "_")
            _savefig(fig, dep_dir / f"dep_{rank:02d}_{safe_name}.png")
        except Exception as exc:
            logger.warning("[Dependence] %s / %s: %s", result.model_name, feat, exc)


# ---------------------------------------------------------------------------
# 4. Patient-level Waterfall Plots
# ---------------------------------------------------------------------------

def _select_patient_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_each: int = 2,
) -> Dict[str, List[int]]:
    """Select representative patient indices for each outcome category."""
    cases: Dict[str, List[int]] = {}

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(tp):
        cases["true_positive"] = tp[np.argsort(y_proba[tp])[::-1][:n_each]].tolist()

    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    if len(tn):
        cases["true_negative"] = tn[np.argsort(1.0 - y_proba[tn])[::-1][:n_each]].tolist()

    fn = np.where((y_true == 1) & (y_pred == 0))[0]
    if len(fn):
        cases["false_negative"] = fn[np.argsort(y_proba[fn])[:n_each]].tolist()

    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    if len(fp):
        cases["false_positive"] = fp[np.argsort(y_proba[fp])[::-1][:n_each]].tolist()

    return cases


def _waterfall_bar_fallback(
    sv_1d:      np.ndarray,    # 1-D (n_features,)
    feat_names: List[str],
    case_label: str,
    model_name: str,
    proba:      float,
    out:        Path,
) -> None:
    """Plain bar chart waterfall used when shap.plots.waterfall is unavailable."""
    top_n   = min(15, len(feat_names))
    top_idx = [int(i) for i in np.argsort(np.abs(sv_1d))[::-1][:top_n]]
    feats   = [feat_names[i] for i in top_idx]
    vals    = sv_1d[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#d73027" if v > 0 else "#4575b4" for v in vals]
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("SHAP Value (impact on CKD probability)", fontsize=10)
    ax.set_title(
        f"Patient Waterfall — {case_label}\nModel: {model_name} | p(CKD)={proba:.3f}",
        fontsize=10, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(
        handles=[
            mpatches.Patch(color="#d73027", label="↑ Increases CKD risk"),
            mpatches.Patch(color="#4575b4", label="↓ Decreases CKD risk"),
        ],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()
    _savefig(fig, out)


def plot_waterfall_plots(
    result:     "SHAPResult",
    X:          pd.DataFrame,
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_proba:    np.ndarray,
    output_dir: Path,
    n_patients: int = 2,
) -> Dict[str, Any]:
    """
    Generate patient-level waterfall plots.

    result.values is always 2-D.  patient_shap(idx) returns a guaranteed
    1-D array so all indexing with Python ints is safe.
    """
    if not _SHAP_OK or not _MPL_OK:
        return {}

    wf_dir     = output_dir / result.model_name / "waterfall"
    wf_dir.mkdir(parents=True, exist_ok=True)
    feat_names = result.feature_names
    cases      = _select_patient_cases(y_true, y_pred, y_proba, n_each=n_patients)
    summaries: Dict[str, Any] = {}

    label_map = {
        "true_positive":  "✓ Correctly Predicted CKD",
        "true_negative":  "✓ Correctly Predicted notCKD",
        "false_negative": "⚠ Missed CKD (False Negative)",
        "false_positive": "⚠ Over-predicted CKD (False Positive)",
    }

    for case_type, indices in cases.items():
        for rank, patient_idx in enumerate(indices, 1):
            # patient_idx comes from .tolist() so it is already a Python int
            sv_1d      = result.patient_shap(patient_idx)   # 1-D, guaranteed
            proba_val  = float(y_proba[patient_idx])
            case_label = label_map.get(case_type, case_type)

            # top3_idx are Python ints — safe for feat_names[i]
            top3_idx   = [int(i) for i in np.argsort(np.abs(sv_1d))[::-1][:3]]

            summaries[f"{case_type}_{rank}"] = {
                "patient_index": patient_idx,
                "case_type":     case_type,
                "case_label":    case_label,
                "y_true":        int(y_true[patient_idx]),
                "y_pred":        int(y_pred[patient_idx]),
                "y_proba":       proba_val,
                "top3_features": [
                    {
                        "feature":       feat_names[i],
                        "shap_value":    float(sv_1d[i]),
                        "feature_value": float(X.iloc[patient_idx, i]),
                    }
                    for i in top3_idx
                ],
            }

            out_path = wf_dir / f"{case_type.replace('_', '-')}_{rank:02d}.png"

            # Try modern shap.plots.waterfall first, fall back to bar chart
            try:
                exp = shap.Explanation(
                    values        = sv_1d,                          # 1-D ✓
                    base_values   = result.expected_value,          # scalar ✓
                    data          = X.iloc[patient_idx].values,     # 1-D ✓
                    feature_names = feat_names,
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.sca(ax)
                shap.plots.waterfall(exp, max_display=15, show=False)
                fig = plt.gcf()
                fig.suptitle(
                    f"Patient Waterfall — {case_label}\n"
                    f"Model: {result.model_name} | p(CKD)={proba_val:.3f}",
                    fontsize=10, fontweight="bold", y=1.02,
                )
                _savefig(fig, out_path)
            except Exception as exc:
                logger.warning(
                    "[Waterfall] %s patient #%d: %s — using bar fallback.",
                    result.model_name, patient_idx, exc,
                )
                _waterfall_bar_fallback(
                    sv_1d, feat_names, case_label,
                    result.model_name, proba_val, out_path,
                )

    return summaries


# =============================================================================
# 5. Cross-Model Agreement
# =============================================================================

def analyze_cross_model_agreement(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir:  Path,
) -> pd.DataFrame:
    if not _MPL_OK:
        return pd.DataFrame()

    model_names = list(predictions.keys())
    if len(model_names) < 2:
        logger.warning("[Agreement] Need ≥ 2 models. Skipping.")
        return pd.DataFrame()

    y_true_ref   = predictions[model_names[0]][0]
    n            = len(y_true_ref)
    pred_matrix  = np.zeros((n, len(model_names)), dtype=int)
    proba_matrix = np.zeros((n, len(model_names)))

    for j, mn in enumerate(model_names):
        _, yp, yprob = predictions[mn]
        if yp    is not None: pred_matrix[:, j]  = yp
        if yprob is not None: proba_matrix[:, j] = yprob

    majority  = (pred_matrix.mean(axis=1) >= 0.5).astype(int)
    agreement = (pred_matrix == majority[:, None]).mean(axis=1)

    df_ag = pd.DataFrame(pred_matrix, columns=model_names)
    df_ag.insert(0, "y_true", y_true_ref)
    df_ag["majority_vote"]     = majority
    df_ag["agreement_score"]   = agreement
    df_ag["high_disagreement"] = agreement < 0.8
    df_ag["mean_proba"]        = proba_matrix.mean(axis=1)
    df_ag["proba_std"]         = proba_matrix.std(axis=1)

    out_dir = output_dir / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ag.to_csv(out_dir / "agreement_analysis.csv", index=True, index_label="patient_index")

    # Heatmap
    n_show     = min(80, n)
    sorted_idx = np.argsort(agreement)[:n_show]
    correct    = (pred_matrix[sorted_idx] == y_true_ref[sorted_idx, None]).astype(float)
    cmap       = LinearSegmentedColormap.from_list("ckg", ["#d73027", "#1a9850"])

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), 10))
    im = ax.imshow(correct.T, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel("Patients (sorted by disagreement, most disagreed first)", fontsize=10)
    ax.set_title(
        "Cross-Model Agreement Heatmap\nGreen = Correct, Red = Incorrect",
        fontsize=12, fontweight="bold",
    )
    ax2 = ax.twinx()
    ax2.plot(range(n_show), agreement[sorted_idx], color="navy",
             linewidth=1.5, marker=".", markersize=3)
    ax2.set_ylabel("Agreement Score", fontsize=9, color="navy")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", colors="navy")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Correct (1) / Incorrect (0)")
    fig.tight_layout()
    _savefig(fig, out_dir / "agreement_heatmap.png")

    # Distribution + scatter
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(agreement, bins=20, color="#3288bd", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.8, color="#d73027", linestyle="--", linewidth=1.5,
                    label="High disagreement threshold (0.8)")
    axes[0].set_xlabel("Agreement Score")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("Agreement Distribution", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].scatter(
        df_ag["proba_std"], df_ag["agreement_score"],
        c=df_ag["y_true"], cmap="RdYlGn", alpha=0.6, edgecolors="white", linewidth=0.3,
    )
    axes[1].set_xlabel("Std of Predicted Probabilities")
    axes[1].set_ylabel("Agreement Score")
    axes[1].set_title("Uncertainty vs. Agreement", fontsize=11, fontweight="bold")
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(
        handles=[
            mpatches.Patch(color="#1a9850", label="CKD (y=1)"),
            mpatches.Patch(color="#d73027", label="notCKD (y=0)"),
        ],
        fontsize=8,
    )
    fig2.suptitle("Cross-Model Disagreement Analysis", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    _savefig(fig2, out_dir / "disagreement_distribution.png")

    n_dis  = int((agreement < 1.0).sum())
    n_high = int((agreement < 0.8).sum())
    logger.info(
        "[Agreement] Patients: %d | Any disagreement: %d | High disagreement (<0.8): %d",
        n, n_dis, n_high,
    )
    _save_json({
        "n_patients":          n,
        "n_any_disagreement":  n_dis,
        "n_high_disagreement": n_high,
        "mean_agreement":      float(agreement.mean()),
        "models_compared":     model_names,
    }, out_dir / "agreement_summary.json")

    return df_ag


# =============================================================================
# 6. LIME (optional)
# =============================================================================

def explain_with_lime(
    model:      Any,
    X_bg:       pd.DataFrame,
    X_test:     pd.DataFrame,
    y_proba:    np.ndarray,
    y_true:     np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        logger.warning("[LIME] 'lime' not installed — skipping. pip install lime")
        return

    if not _MPL_OK:
        return

    lime_dir   = output_dir / model_name / "lime"
    lime_dir.mkdir(parents=True, exist_ok=True)
    feat_names = list(X_test.columns)
    X_bg_arr   = X_bg.values if X_bg is not None else X_test.values[:50]

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data    = X_bg_arr,
        feature_names    = feat_names,
        class_names      = ["notCKD", "CKD"],
        mode             = "classification",
        random_state     = 42,
        discretize_continuous = True,
    )

    cases = {
        "most_confident_CKD":    int(np.argmax(y_proba)),
        "most_confident_notCKD": int(np.argmin(y_proba)),
        "boundary_case":         int(np.argmin(np.abs(y_proba - 0.5))),
    }

    for case_name, idx in cases.items():
        try:
            exp = lime_exp.explain_instance(
                data_row  = X_test.values[idx],
                predict_fn = model.predict_proba,
                num_features = 15,
                num_samples  = 500,
            )
            with open(lime_dir / f"{case_name}.html", "w") as fh:
                fh.write(exp.as_html())
            fig = exp.as_pyplot_figure(label=1)
            fig.suptitle(
                f"LIME — {case_name.replace('_', ' ').title()}\n"
                f"Model: {model_name} | p(CKD)={y_proba[idx]:.3f} "
                f"| Truth: {'CKD' if y_true[idx] else 'notCKD'}",
                fontsize=10, fontweight="bold",
            )
            _savefig(fig, lime_dir / f"{case_name}.png")
            _save_json({
                "case_name":     case_name,
                "patient_index": idx,
                "y_true":        int(y_true[idx]),
                "y_proba":       float(y_proba[idx]),
                "lime_features": exp.as_list(label=1),
            }, lime_dir / f"{case_name}.json")
            logger.info("[LIME] %s / %s → %s", model_name, case_name, lime_dir)
        except Exception as exc:
            logger.warning("[LIME] %s / %s: %s", model_name, case_name, exc)


# =============================================================================
# 7. Clinician Markdown Report
# =============================================================================

def generate_clinician_report(
    model_name:          str,
    patient_summaries:   Dict[str, Any],
    global_shap_features: List[Tuple[str, float]],
    output_dir:          Path,
) -> None:
    report_dir = output_dir / model_name
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path   = report_dir / "clinician_explanation_report.md"

    def _dir(sv: float) -> str:
        return "**increased**" if sv > 0 else "**decreased**"

    def _str(sv: float) -> str:
        av = abs(sv)
        if av > 0.3:  return "strongly"
        if av > 0.1:  return "moderately"
        if av > 0.05: return "slightly"
        return "marginally"

    lines: List[str] = [
        "# Clinician Explanation Report",
        "",
        f"**Model:** {model_name}  ",
        "**Task:** CKD Prediction (UCI Dataset)  ",
        "**Scope:** Patient-level SHAP explanations for selected test cases  ",
        "",
        "> ⚠️ **Disclaimer:** For research purposes only. Not medical advice.",
        "> All predictions must be reviewed by a qualified clinician.",
        "",
        "---",
        "",
        "## Top Features (Global SHAP Importance)",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| |",
        "|------|---------|--------------| ",
    ]
    for rank, (feat, imp) in enumerate(global_shap_features[:10], 1):
        lines.append(f"| {rank} | `{feat}` | {imp:.4f} |")

    lines += ["", "---", "", "## Patient-Level Explanations", ""]

    for case_key, pdata in patient_summaries.items():
        ct    = pdata["case_type"]
        idx   = pdata["patient_index"]
        pr    = pdata["y_proba"]
        yt    = "CKD" if pdata["y_true"] == 1 else "notCKD"
        yp    = "CKD" if pdata["y_pred"] == 1 else "notCKD"
        label = pdata["case_label"]
        top3  = pdata["top3_features"]

        lines += [
            f"### {label}  (Patient #{idx})",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| True Diagnosis | {yt} |",
            f"| Model Prediction | {yp} |",
            f"| CKD Probability | {pr:.1%} |",
            "",
            "**Key drivers for this prediction:**",
            "",
        ]
        for rp, fi in enumerate(top3, 1):
            lines.append(
                f"{rp}. `{fi['feature']}` = **{fi['feature_value']:.3g}** — "
                f"this {_str(fi['shap_value'])} {_dir(fi['shap_value'])} "
                f"the CKD probability (SHAP = {fi['shap_value']:+.3f})."
            )
        if ct == "false_negative":
            lines += [
                "",
                "> ⚠️ **Clinical Note:** This CKD patient was **missed**.",
                f"> Despite having CKD, the model assigned only {pr:.1%} probability.",
            ]
        elif ct == "false_positive":
            lines += [
                "",
                "> ⚠️ **Clinical Note:** This patient was **over-flagged**.",
                f"> The model predicted CKD ({pr:.1%}) but the true diagnosis is notCKD.",
            ]
        lines += ["", "---", ""]

    lines += [
        "## Interpretation Guide",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| Positive SHAP | Pushes prediction *towards* CKD |",
        "| Negative SHAP | Pushes prediction *away from* CKD |",
        "| Large \\|SHAP\\| | Feature was influential for this patient |",
    ]

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("[Report] Clinician report → %s", out_path)


# =============================================================================
# Main orchestrator
# =============================================================================

def run_explainability(
    models_to_run:      List[str],
    output_dir:         Path,
    n_patients:         int  = 2,
    run_lime:           bool = True,
    n_dependence_plots: int  = 5,
    config_path:        str  = "config/evaluation_config.yaml",
) -> None:
    """Full explainability pipeline for UCI binary classification models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CKD EXPLAINABILITY MODULE")
    logger.info("Models : %s", models_to_run)
    logger.info("Output : %s", output_dir)
    logger.info("=" * 70)

    if not _SHAP_OK:
        logger.error("SHAP is required. Install with: pip install shap")
        sys.exit(1)

    try:
        pp = _get_paths(config_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Cannot load pipeline paths: %s", exc)
        sys.exit(1)

    # Accumulators for cross-model outputs
    all_results:     Dict[str, SHAPResult] = {}
    all_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for model_name in models_to_run:
        logger.info("-" * 50)
        logger.info("[%s] Starting...", model_name)

        # ── Load model ──────────────────────────────────────────────────────
        try:
            model = _load_model(pp, model_name)
        except FileNotFoundError as exc:
            logger.warning("[%s] Skipping — model not found:\n  %s", model_name, exc)
            continue

        # ── Load data ────────────────────────────────────────────────────────
        try:
            X_test, y_test, feat_names = _load_test_data(pp, model_name)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[%s] Skipping — data load failed:\n  %s", model_name, exc)
            continue

        # ── Predict ──────────────────────────────────────────────────────────
        try:
            y_proba = model.predict_proba(X_test.values)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)
        except Exception as exc:
            logger.warning("[%s] predict_proba failed: %s", model_name, exc)
            continue

        all_predictions[model_name] = (y_test, y_pred, y_proba)

        # ── Background (training data sample) ───────────────────────────────
        background = _load_train_background(pp, model_name, max_samples=100)

        # ── Compute SHAP — returns validated SHAPResult ──────────────────────
        try:
            result = compute_shap(model, model_name, X_test, background, feat_names)
            all_results[model_name] = result
        except Exception as exc:
            logger.warning("[%s] SHAP computation failed: %s", model_name, exc)
            continue

        # ── 2. Beeswarm ────────────────────────────────────────────────────
        logger.info("[%s] Beeswarm...", model_name)
        plot_shap_beeswarm(result, X_test, output_dir)

        # ── 3. Dependence plots ────────────────────────────────────────────
        logger.info("[%s] Dependence plots...", model_name)
        plot_shap_dependence(result, X_test, output_dir, n_top=n_dependence_plots)

        # ── 4. Waterfall plots ─────────────────────────────────────────────
        logger.info("[%s] Waterfall plots...", model_name)
        patient_summaries = plot_waterfall_plots(
            result, X_test, y_test, y_pred, y_proba,
            output_dir, n_patients=n_patients,
        )

        # ── 7. Clinician report ────────────────────────────────────────────
        ma          = result.mean_abs()        # 1-D (n_features,)
        top_feats   = sorted(zip(feat_names, ma.tolist()), key=lambda x: -x[1])
        logger.info("[%s] Clinician report...", model_name)
        generate_clinician_report(model_name, patient_summaries, top_feats, output_dir)

        # ── 6. LIME ────────────────────────────────────────────────────────
        if run_lime:
            if background is None:
                logger.warning(
                    "[%s] LIME skipped — training background not available.", model_name
                )
            else:
                logger.info("[%s] LIME explanations...", model_name)
                explain_with_lime(
                    model, background, X_test, y_proba, y_test, model_name, output_dir
                )

        # ── Save SHAP values CSV ───────────────────────────────────────────
        sv_csv = output_dir / model_name / "shap_values.csv"
        pd.DataFrame(result.values, columns=feat_names).to_csv(sv_csv, index=False)
        logger.info("[%s] SHAP values CSV → %s", model_name, sv_csv)

    # ── 1. Global SHAP summary (all models) ─────────────────────────────────
    if all_results:
        logger.info("Plotting global SHAP summary...")
        plot_global_shap_summary(all_results, output_dir)
    else:
        logger.warning("[Global SHAP] No SHAP values computed for any model.")

    # ── 5. Cross-model agreement ─────────────────────────────────────────────
    if len(all_predictions) >= 2:
        logger.info("Cross-model agreement analysis...")
        analyze_cross_model_agreement(all_predictions, output_dir)
    elif all_predictions:
        logger.info("[Agreement] Only 1 model processed — cross-model analysis skipped.")
    else:
        logger.warning("[Agreement] No models produced predictions — skipping.")

    logger.info("=" * 70)
    logger.info("EXPLAINABILITY COMPLETE → %s", output_dir.resolve())
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CKD Explainability Module — SHAP, LIME, and clinician reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", nargs="+", default=UCI_MODELS,
        help=f"Models to explain (default: all). Choices: {UCI_MODELS}",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: from config → artifacts/explainability/uci)",
    )
    p.add_argument(
        "--patients", type=int, default=2,
        help="Waterfall patients per case type (default: 2)",
    )
    p.add_argument(
        "--no-lime", action="store_true",
        help="Skip LIME explanations (faster)",
    )
    p.add_argument(
        "--dependence-plots", type=int, default=5,
        help="Top-N SHAP dependence plots (default: 5)",
    )
    p.add_argument(
        "--config", default="config/evaluation_config.yaml",
        help="Path to evaluation_config.yaml",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        try:
            pp      = _get_paths(args.config)
            out_dir = pp.explainability_dir(TASK_KEY)
        except Exception:
            out_dir = Path("artifacts/explainability/uci")

    run_explainability(
        models_to_run      = args.model,
        output_dir         = out_dir,
        n_patients         = args.patients,
        run_lime           = not args.no_lime,
        n_dependence_plots = args.dependence_plots,
        config_path        = args.config,
    )


if __name__ == "__main__":
    main()
