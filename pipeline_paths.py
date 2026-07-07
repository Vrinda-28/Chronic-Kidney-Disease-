"""
pipeline_paths.py
=================

Centralised path resolver for post-evaluation pipeline stages.

Consumed by:
  explainability.py   — SHAP, LIME, waterfall, clinician reports
  ablation_study.py   — feature-subset ablation

Design goals
------------
  • Single source of truth: reads config/evaluation_config.yaml (the same
    config already used by evaluate.py) so there is no path duplication.
  • Typed helpers: callers get Path objects, not raw strings.
  • Fail-fast: raises FileNotFoundError / KeyError with helpful messages
    rather than silently returning wrong paths.
  • Zero side-effects: this module only reads; it never writes files.

Usage
-----
    from pipeline_paths import PipelinePaths

    pp = PipelinePaths()                        # reads config/evaluation_config.yaml
    test_csv  = pp.test_csv("uci")              # → Path("data/splits/uci_test.csv")
    train_csv = pp.train_csv("uci")             # → Path("data/splits/uci_train.csv")
    model_dir = pp.model_dir("uci")             # → Path("artifacts/models/uci")
    target    = pp.target_col("uci")            # → "ckd_label"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "config/evaluation_config.yaml"


class PipelinePaths:
    """
    Reads config/evaluation_config.yaml once and exposes typed path helpers.

    All returned Path objects are relative to the project root (cwd when
    the pipeline scripts are run, which is always the project root).
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG) -> None:
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"[PipelinePaths] Config not found: {cfg_file.resolve()}\n"
                f"  Expected: config/evaluation_config.yaml\n"
                f"  Run all pipeline stages from the project root directory."
            )
        with open(cfg_file, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        self._tasks: Dict[str, Any] = raw.get("tasks", {})
        if not self._tasks:
            raise KeyError(
                "[PipelinePaths] 'tasks' key missing from evaluation_config.yaml"
            )
        logger.debug("[PipelinePaths] Loaded config from %s", cfg_file)

    # -------------------------------------------------------------------------
    # Task config accessor
    # -------------------------------------------------------------------------

    def _task(self, task_key: str) -> Dict[str, Any]:
        if task_key not in self._tasks:
            raise KeyError(
                f"[PipelinePaths] Unknown task '{task_key}'. "
                f"Available tasks: {list(self._tasks.keys())}"
            )
        return self._tasks[task_key]

    # -------------------------------------------------------------------------
    # Data split paths
    # -------------------------------------------------------------------------

    def splits_dir(self, task_key: str) -> Path:
        """Directory containing split CSVs for this task."""
        return Path(self._task(task_key)["splits_dir"])

    def test_csv(self, task_key: str) -> Path:
        """Full path to the held-out test CSV for this task."""
        t = self._task(task_key)
        p = Path(t["splits_dir"]) / t["test_file"]
        if not p.exists():
            raise FileNotFoundError(
                f"[PipelinePaths] Test split not found: {p}\n"
                f"  Expected from config: splits_dir={t['splits_dir']}, "
                f"test_file={t['test_file']}\n"
                f"  Run train_test_split.py first."
            )
        return p

    def train_csv(self, task_key: str) -> Path:
        """Full path to the training CSV for this task (used as SHAP/LIME background)."""
        t = self._task(task_key)
        key = "train_file"
        if key not in t:
            raise KeyError(
                f"[PipelinePaths] 'train_file' not defined for task '{task_key}' "
                f"in evaluation_config.yaml"
            )
        p = Path(t["splits_dir"]) / t[key]
        if not p.exists():
            raise FileNotFoundError(
                f"[PipelinePaths] Train split not found: {p}\n"
                f"  Run train_test_split.py first."
            )
        return p

    def train_csv_optional(self, task_key: str) -> "Path | None":
        """Like train_csv() but returns None instead of raising if file is absent."""
        try:
            return self.train_csv(task_key)
        except (KeyError, FileNotFoundError):
            return None

    # -------------------------------------------------------------------------
    # Model artifact paths
    # -------------------------------------------------------------------------

    def model_dir(self, task_key: str) -> Path:
        """Root directory for trained model artifacts for this task."""
        t   = self._task(task_key)
        key = "model_artifacts_dir"
        if key not in t:
            raise KeyError(
                f"[PipelinePaths] 'model_artifacts_dir' not defined for task "
                f"'{task_key}' in evaluation_config.yaml"
            )
        return Path(t[key])

    def calibrated_model(self, task_key: str, model_name: str) -> Path:
        """Path to calibrated_model.joblib for a specific model."""
        return self.model_dir(task_key) / model_name / "calibrated_model.joblib"

    def final_model(self, task_key: str, model_name: str) -> Path:
        """Path to final_model.joblib for a specific model (fallback)."""
        return self.model_dir(task_key) / model_name / "final_model.joblib"

    def selected_features_json(self, task_key: str, model_name: str) -> Path:
        """Path to selected_features.json for a specific model."""
        return self.model_dir(task_key) / model_name / "selected_features.json"

    # -------------------------------------------------------------------------
    # Schema helpers
    # -------------------------------------------------------------------------

    def target_col(self, task_key: str) -> str:
        """Target column name in the split CSVs for this task."""
        return self._task(task_key)["target_col"]

    def task_type(self, task_key: str) -> str:
        """'binary' or 'multiclass'."""
        return self._task(task_key).get("task_type", "binary")

    def class_names(self, task_key: str) -> list:
        """List of class name strings."""
        return self._task(task_key).get("class_names", [])

    def best_model(self, task_key: str) -> str:
        """Name of the best model as recorded in evaluation_config.yaml."""
        return self._task(task_key).get("best_model", "")

    # -------------------------------------------------------------------------
    # Output artifact paths
    # -------------------------------------------------------------------------

    def explainability_dir(self, task_key: str) -> Path:
        """Output directory for explainability artifacts."""
        return Path("artifacts/explainability") / task_key

    def ablation_dir(self) -> Path:
        """Output directory for ablation artifacts."""
        return Path("artifacts/ablation")
