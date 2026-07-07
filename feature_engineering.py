"""
feature_engineering.py
=======================

Production-grade, leakage-safe feature engineering stage for the CKD
Prediction and Explainable AI research project.

Position in the pipeline
-------------------------
    data_loader.py  ->  preprocess.py  ->  feature_engineering.py (THIS FILE)
    ->  train_test_split.py  ->  SMOTE (train folds only)  ->  models  ->  SHAP
    ->  UAE external validation

This module:
    * Loads the already-processed, ML-ready CSVs produced by preprocess.py
      (``data/processed/{uci,kaggle,uae}_processed.csv``). It never reads
      raw data and never re-runs ingestion/preprocessing.
    * Creates medically meaningful engineered features, **only** when their
      required source columns are present in a given dataset - the three
      datasets have different schemas (UCI: 24 clinical features, Kaggle:
      similar plus eGFR/diastolic BP, UAE: a baseline + comorbidity-history
      cohort) and this module never assumes column parity between them.
    * Every engineered feature is a deterministic, row-wise arithmetic
      transform of EXISTING columns. Nothing is fit, fitted-and-applied,
      learned, or estimated from the data - there are no statistics
      computed across rows, across datasets, or against the UAE cohort.
      This is what keeps the stage inherently free of data leakage and
      keeps UAE strictly isolated: UAE rows never influence anything
      computed for UCI/Kaggle rows, and vice versa.
    * NEVER reads ckd_label / ckd_stage_label / ckd_binary_label /
      ckd_affected when constructing a feature. This is enforced three
      ways: (1) no feature definition references them by construction,
      (2) ckd_binary_label and ckd_affected are additionally treated as
      raw target-PROXY columns (not just targets) and are physically
      removed from the Kaggle dataset before feature creation even runs -
      see LEAKAGE_PRONE_COLUMNS and the project leakage review that
      justified this (a crosstab showed 100% class purity for two CKD
      stages against ckd_binary_label), and (3) a generic, informational
      statistical screen (_audit_statistical_leakage) flags any other
      low-cardinality column with suspiciously high association against
      the target for human review, without auto-removing it.
    * Does NOT train models, split data, run SMOTE, scale, reduce
      dimensionality, or select features. It only adds columns.

Outputs
-------
    data/engineered/uci_engineered.csv
    data/engineered/kaggle_engineered.csv
    data/engineered/uae_engineered.csv
    data/engineered/<dataset>_provenance.csv      (source_dataset /
                                                     original_row_id, kept
                                                     separately - see
                                                     "Provenance handling")
    artifacts/feature_engineering/feature_summary.json
    artifacts/feature_engineering/feature_summary.joblib  (same content,
                                                            for downstream
                                                            Python loading)

Provenance handling
--------------------
Per the project's architectural rule, ``source_dataset`` and
``original_row_id`` are dropped from every engineered training dataset
(they are identifiers, not clinical features, and would otherwise leak
dataset identity into a supposedly schema-unified feature set). They are
NOT discarded, however - each dataset's provenance is saved to its own
``<dataset>_provenance.csv``, row-order-aligned 1:1 with the engineered
CSV (an explicit ``row_position`` column is included in the provenance
file specifically so this alignment is verifiable, not just assumed).

Usage
-----
    python feature_engineering.py

    # or
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    uci_eng, kaggle_eng, uae_eng = fe.run()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import normalized_mutual_info_score
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - sklearn is a project dependency,
    # but the statistical leakage audit degrades gracefully without it
    # rather than hard-failing feature engineering itself.
    _SKLEARN_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================

# Columns that are targets or target-derived. ABSOLUTE RULE: none of these
# may ever appear in a feature's `required_columns`. Enforced defensively
# in _assert_no_target_leakage(), not just by convention.
TARGET_LIKE_COLUMNS: Set[str] = {
    "ckd_label",
    "ckd_stage_label",
    "ckd_binary_label",
    "ckd_affected",
}

# Columns added by data_loader.py that are identifiers, not clinical
# features. Dropped from engineered outputs; preserved in a separate
# provenance file (see module docstring).
PROVENANCE_COLUMNS: Set[str] = {"source_dataset", "original_row_id"}

# -----------------------------------------------------------------------
# Leakage-prone RAW columns (reviewed 2026 - see project leakage review).
#
# These are columns that exist in the *processed* dataset (i.e. they
# survived data_loader.py + preprocess.py) but are themselves disease-
# status fields, not clinical measurements - they were never engineered
# by this module, they arrived as raw passthrough columns.
#
# Evidence for "kaggle.ckd_binary_label":
#   A crosstab against ckd_stage_label showed 100% class purity for two
#   of five stages (every row with stage in {s3, s5} maps to ckd=1; only
#   stage s1 leans cleanly the other way). Real independent clinical
#   biomarkers essentially never reach 100% purity without overfitting -
#   this is the signature of a column derived from (or alongside) the
#   same diagnostic process that produced the label, not an independent
#   measurement. It is therefore treated as a target proxy and excluded.
#
# Evidence for "kaggle.ckd_affected":
#   No independent measurement underlies this column's name - "affected"
#   denotes disease status itself, structurally identical in kind to
#   ckd_binary_label. Excluded on the same reasoning, with the runtime
#   statistical audit below (_audit_statistical_leakage) reporting its
#   actual association with the target on real data so this decision can
#   be confirmed (or contested) by inspecting feature_summary.json rather
#   than taken purely on naming.
#
# If you disagree with either exclusion after reviewing the audit output,
# remove the relevant entry here - nothing else in the file depends on
# this exact dictionary contents.
# -----------------------------------------------------------------------
LEAKAGE_PRONE_COLUMNS: Dict[str, Dict[str, str]] = {
    "kaggle": {
        "ckd_binary_label": (
            "Confirmed target proxy: crosstab against ckd_stage_label shows "
            "100% class purity for stages s3 and s5 (and near-total purity "
            "for s4), consistent with this column being a disease-status "
            "field derived alongside the target rather than an independent "
            "clinical measurement. Including it would artificially inflate "
            "Accuracy/F1/AUC/Balanced Accuracy."
        ),
        "ckd_affected": (
            "Likely target proxy: column name denotes disease-affectation "
            "status itself (structurally identical in kind to "
            "ckd_binary_label, which was confirmed via crosstab). Excluded "
            "by the same domain reasoning; see this dataset's "
            "leakage_audit.statistical_screen in feature_summary.json for "
            "the measured association on the actual data, which should be "
            "used to confirm or contest this decision."
        ),
    },
    "uci": {},
    "uae": {},
}

# Low-cardinality (categorical/flag-like) columns are screened for
# suspiciously strong association with the target (see
# _audit_statistical_leakage). This deliberately does NOT cover continuous
# lab values (serum_creatinine, egfr, etc.) - strong correlation there is
# expected and desirable, not a leakage signal. The threshold below flags
# for human review only; nothing is auto-removed based on this screen.
LEAKAGE_AUDIT_MAX_CARDINALITY: int = 10
LEAKAGE_AUDIT_NMI_FLAG_THRESHOLD: float = 0.5

# Reference values used by a couple of composite scores below. These are
# commonly-cited normal/threshold ranges from general clinical literature
# (NOT derived/fitted from this project's data, so using them introduces
# no leakage and is reproducible without depending on any particular
# sample's statistics). They are deliberately simple, transparent, and
# documented inline at each point of use - these composite scores are
# engineered heuristics for ML/SHAP purposes, not validated diagnostic
# indices (e.g. they are not the Framingham score, FIB-4, etc.).
CLINICAL_REFERENCE_VALUES: Dict[str, float] = {
    "hemoglobin_normal_g_dl": 15.0,       # approx. upper-normal adult reference
    "pcv_normal_pct": 45.0,               # approx. upper-normal packed cell volume
    "rbc_normal_million_per_ul": 5.0,     # approx. upper-normal RBC count
    "bmi_obesity_threshold": 30.0,        # WHO obesity threshold (kg/m^2)
    "cholesterol_borderline_high_mg_dl": 200.0,  # ATP III borderline-high total cholesterol
}


# =============================================================================
# Logging
# =============================================================================

def _build_logger(
    log_dir: str = "logs",
    log_filename: str = "feature_engineering.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """Same pattern as data_loader.py / preprocess.py, under its own
    logger name and log file so runs don't interleave with other stages."""
    logger = logging.getLogger("ckd_feature_engineering")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, log_filename), maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not set up file logging at %s: %s", log_dir, exc)

    return logger


# =============================================================================
# Exceptions
# =============================================================================

class FeatureEngineeringError(Exception):
    """Base exception for this module."""


class TargetLeakageError(FeatureEngineeringError):
    """
    Raised if a feature definition ever references a target-like column.
    This should be unreachable in normal operation (every feature
    definition below is hand-written to avoid target columns) - it exists
    as a defensive, fail-loud guard against future edits introducing
    leakage by accident.
    """


# =============================================================================
# Reporting data structures
# =============================================================================

@dataclass
class FeatureCreationRecord:
    """One row of "did this feature get created for this dataset, and why/why not"."""

    feature_name: str
    dataset: str
    created: bool
    reason: str
    source_columns_used: List[str] = field(default_factory=list)
    n_values_computed: int = 0
    n_values_null_after_compute: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "dataset": self.dataset,
            "created": self.created,
            "reason": self.reason,
            "source_columns_used": self.source_columns_used,
            "n_values_computed": self.n_values_computed,
            "n_values_null_after_compute": self.n_values_null_after_compute,
        }


@dataclass
class FeatureDefinitionMeta:
    """Static documentation for one engineered feature, independent of dataset."""

    name: str
    formula: str
    clinical_rationale: str


@dataclass
class LeakageExclusionRecord:
    """One row of 'this raw column was excluded as a target proxy, here's why'."""

    column: str
    dataset: str
    rationale: str
    n_values_excluded: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "dataset": self.dataset,
            "rationale": self.rationale,
            "n_values_excluded": self.n_values_excluded,
        }


@dataclass
class StatisticalLeakageFlag:
    """One row of 'this column is suspiciously associated with the target -
    review it' from the generic, informational-only statistical screen."""

    column: str
    dataset: str
    target_column: str
    normalized_mutual_info: float
    n_unique_values: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "dataset": self.dataset,
            "target_column": self.target_column,
            "normalized_mutual_info": round(self.normalized_mutual_info, 4),
            "n_unique_values": self.n_unique_values,
            "note": (
                "Flagged for human/domain review only - NOT auto-removed. "
                "High association can mean genuine strong clinical signal "
                "OR a target proxy; this screen cannot distinguish the two "
                "on its own."
            ),
        }


# =============================================================================
# Safe arithmetic helpers
# =============================================================================

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Element-wise division that turns zero/near-zero denominators into NaN
    instead of +/-inf. No rows are dropped - a NaN result simply means
    that particular engineered value couldn't be meaningfully computed
    for that row (e.g. a recorded age or eGFR of exactly 0).
    """
    denom = denominator.astype("float64").replace(0, np.nan)
    return numerator.astype("float64") / denom


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion; non-numeric values become NaN (never
    raises - processed data should already be numeric, but this keeps
    feature construction robust to unexpected upstream dtype drift)."""
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# Feature engineering orchestrator
# =============================================================================

class FeatureEngineer:
    """
    Loads the processed UCI / Kaggle / UAE datasets and adds medically
    meaningful, leakage-free engineered features to each, independently.

    Parameters
    ----------
    processed_dir:
        Directory containing ``uci_processed.csv``, ``kaggle_processed.csv``,
        ``uae_processed.csv`` (the output of preprocess.py).
    engineered_dir:
        Directory to write ``<dataset>_engineered.csv`` and
        ``<dataset>_provenance.csv`` to.
    artifacts_dir:
        Directory to write ``feature_summary.json`` /
        ``feature_summary.joblib`` to.
    """

    # Per-dataset target column name (kept in engineered output, but never
    # used as an input to any feature - see _assert_no_target_leakage).
    TARGET_COLUMNS: Dict[str, str] = {
        "uci": "ckd_label",
        "kaggle": "ckd_stage_label",
        "uae": "ckd_label",
    }

    def __init__(
        self,
        processed_dir: str = "data/processed",
        engineered_dir: str = "data/engineered",
        artifacts_dir: str = "artifacts/feature_engineering",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.engineered_dir = Path(engineered_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logger or _build_logger()

        self._records: List[FeatureCreationRecord] = []
        self._definitions: Dict[str, FeatureDefinitionMeta] = {}
        self._rows_processed: Dict[str, int] = {}
        self._original_feature_counts: Dict[str, int] = {}
        self._leakage_exclusions: List[LeakageExclusionRecord] = []
        self._statistical_flags: List[StatisticalLeakageFlag] = []
        self._excluded_columns_by_dataset: Dict[str, pd.DataFrame] = {}

    # -------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute feature engineering end-to-end for all three datasets."""
        self.logger.info("=" * 70)
        self.logger.info("CKD Feature Engineering - START")
        self.logger.info("=" * 70)

        uci_df = self._load_processed("uci", "uci_processed.csv")
        kaggle_df = self._load_processed("kaggle", "kaggle_processed.csv")
        uae_df = self._load_processed("uae", "uae_processed.csv")

        uci_eng = self.engineer_dataset(uci_df, "uci")
        kaggle_eng = self.engineer_dataset(kaggle_df, "kaggle")
        uae_eng = self.engineer_dataset(uae_df, "uae")

        # UAE is processed via the exact same stateless function calls as
        # UCI/Kaggle, on its own DataFrame, with no cross-dataset merge at
        # any point above - this IS the isolation guarantee, not just a
        # comment promising it.

        self._save_outputs("uci", uci_eng)
        self._save_outputs("kaggle", kaggle_eng)
        self._save_outputs("uae", uae_eng)

        self._write_feature_summary()

        self.logger.info("=" * 70)
        self.logger.info("CKD Feature Engineering - COMPLETE")
        self.logger.info("=" * 70)

        return uci_eng, kaggle_eng, uae_eng

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------

    def _load_processed(self, dataset_key: str, filename: str) -> pd.DataFrame:
        path = self.processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"[{dataset_key}] Processed file not found at {path.resolve()}. "
                f"Run preprocess.py first - feature_engineering.py never reads "
                f"raw data directly."
            )
        df = pd.read_csv(path)
        self._rows_processed[dataset_key] = len(df)
        self._original_feature_counts[dataset_key] = df.shape[1]
        self.logger.info(
            "[%s] Loaded processed dataset: %s -> shape %s",
            dataset_key, path, df.shape,
        )
        return df

    # -------------------------------------------------------------------
    # Leakage guard
    # -------------------------------------------------------------------

    def _assert_no_target_leakage(self, required_columns: List[str], feature_name: str) -> None:
        offending = TARGET_LIKE_COLUMNS.intersection(required_columns)
        if offending:
            raise TargetLeakageError(
                f"Feature '{feature_name}' attempted to use target-like column(s) "
                f"{sorted(offending)} as input. This is forbidden by the project's "
                f"data-leakage rule and indicates a bug in feature_engineering.py "
                f"itself, not bad input data."
            )

    # -------------------------------------------------------------------
    # Generic feature-creation helper
    # -------------------------------------------------------------------

    def _try_create_feature(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        feature_name: str,
        required_columns: List[str],
        compute_fn: Callable[[pd.DataFrame], pd.Series],
        formula: str,
        clinical_rationale: str,
    ) -> pd.DataFrame:
        """
        Shared scaffolding for every feature: checks column availability,
        guards against target leakage, computes the feature if possible,
        and records a FeatureCreationRecord either way. Never raises for
        missing columns - that is an expected, normal occurrence across
        these three differently-shaped datasets.
        """
        self._assert_no_target_leakage(required_columns, feature_name)

        # Register the static definition once (idempotent across datasets).
        if feature_name not in self._definitions:
            self._definitions[feature_name] = FeatureDefinitionMeta(
                name=feature_name, formula=formula, clinical_rationale=clinical_rationale
            )

        if not set(required_columns).issubset(df.columns):
            missing = sorted(set(required_columns) - set(df.columns))
            leakage_registry = LEAKAGE_PRONE_COLUMNS.get(dataset_key, {})
            excluded_for_leakage = [c for c in missing if c in leakage_registry]
            if excluded_for_leakage:
                reason = (
                    f"missing required column(s): {missing} "
                    f"(note: {excluded_for_leakage} unavailable because they were "
                    f"deliberately excluded earlier as target-leakage risks - "
                    f"see leakage_audit.excluded_columns in feature_summary.json, "
                    f"not a data-quality gap)"
                )
            else:
                reason = f"missing required column(s): {missing}"
            self._records.append(FeatureCreationRecord(
                feature_name=feature_name,
                dataset=dataset_key,
                created=False,
                reason=reason,
            ))
            return df

        result = compute_fn(df)
        df = df.copy()
        df[feature_name] = result

        n_null = int(result.isna().sum())
        self._records.append(FeatureCreationRecord(
            feature_name=feature_name,
            dataset=dataset_key,
            created=True,
            reason="created",
            source_columns_used=required_columns,
            n_values_computed=int(len(result) - n_null),
            n_values_null_after_compute=n_null,
        ))
        self.logger.info(
            "[%s] Created feature '%s' from %s (%d non-null / %d rows).",
            dataset_key, feature_name, required_columns, len(result) - n_null, len(result),
        )
        return df

    # -------------------------------------------------------------------
    # Feature definitions
    # -------------------------------------------------------------------
    # Each method below adds exactly one engineered feature (or a closely
    # related small family, e.g. bp_risk_score's per-dataset variants) and
    # is independently skippable. Order does not matter - no feature here
    # depends on another engineered feature as input, only on the
    # processed-dataset's original columns.

    def _add_bun_creatinine_ratio(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["blood_urea", "serum_creatinine"]
        return self._try_create_feature(
            df, key, "bun_creatinine_ratio", cols,
            compute_fn=lambda d: _safe_divide(
                _coerce_numeric(d["blood_urea"]), _coerce_numeric(d["serum_creatinine"])
            ),
            formula="blood_urea / serum_creatinine",
            clinical_rationale=(
                "BUN/creatinine ratio helps distinguish pre-renal azotemia "
                "(high ratio) from intrinsic renal disease (normal/low ratio) "
                "and is a standard, widely used renal-function indicator."
            ),
        )

    def _add_sodium_potassium_ratio(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["sodium", "potassium"]
        return self._try_create_feature(
            df, key, "sodium_potassium_ratio", cols,
            compute_fn=lambda d: _safe_divide(
                _coerce_numeric(d["sodium"]), _coerce_numeric(d["potassium"])
            ),
            formula="sodium / potassium",
            clinical_rationale=(
                "Electrolyte imbalance is a hallmark of declining renal "
                "function; the Na/K ratio condenses two correlated "
                "electrolyte signals into a single interpretable feature."
            ),
        )

    def _add_kidney_dysfunction_score(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["serum_creatinine", "egfr"]
        return self._try_create_feature(
            df, key, "kidney_dysfunction_score", cols,
            compute_fn=lambda d: _safe_divide(
                _coerce_numeric(d["serum_creatinine"]), _coerce_numeric(d["egfr"])
            ),
            formula="serum_creatinine / egfr",
            clinical_rationale=(
                "Creatinine rises and eGFR falls as kidney function declines, "
                "so their ratio amplifies the renal-impairment signal beyond "
                "either marker alone (both move in the same direction)."
            ),
        )

    def _add_bp_risk_score(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Dataset-specific because the three datasets capture blood pressure
        differently:
            UAE:    systolic_bp and diastolic_bp are both available ->
                    mean arterial-pressure-style average.
            Kaggle: only blood_pressure_diastolic is available (no
                    systolic column survived schema mapping) -> per the
                    project spec, use it directly as the equivalent score.
            UCI:    only a single, unsplit 'blood_pressure' reading exists
                    -> used directly as its own risk score for consistency
                    (documented explicitly, not silently assumed).
        """
        if key == "uae":
            cols = ["systolic_bp", "diastolic_bp"]
            return self._try_create_feature(
                df, key, "bp_risk_score", cols,
                compute_fn=lambda d: (
                    _coerce_numeric(d["systolic_bp"]) + _coerce_numeric(d["diastolic_bp"])
                ) / 2.0,
                formula="(systolic_bp + diastolic_bp) / 2",
                clinical_rationale=(
                    "Mean arterial-pressure-style average of systolic and "
                    "diastolic readings; elevated BP is a major driver of "
                    "CKD progression and cardiovascular risk."
                ),
            )
        if key == "kaggle":
            cols = ["blood_pressure_diastolic"]
            return self._try_create_feature(
                df, key, "bp_risk_score", cols,
                compute_fn=lambda d: _coerce_numeric(d["blood_pressure_diastolic"]),
                formula="blood_pressure_diastolic (no systolic column available in this dataset)",
                clinical_rationale=(
                    "Equivalent BP risk indicator for Kaggle, which only "
                    "retained a diastolic reading after schema "
                    "standardization; used directly per project "
                    "specification rather than estimating a missing "
                    "systolic value."
                ),
            )
        if key == "uci":
            cols = ["blood_pressure"]
            return self._try_create_feature(
                df, key, "bp_risk_score", cols,
                compute_fn=lambda d: _coerce_numeric(d["blood_pressure"]),
                formula="blood_pressure (single unsplit reading - no systolic/diastolic split in this dataset)",
                clinical_rationale=(
                    "UCI's classic CKD dataset records a single blood "
                    "pressure value rather than systolic/diastolic "
                    "components; used directly as the dataset's BP risk "
                    "indicator for cross-dataset feature-name consistency."
                ),
            )
        return df

    def _add_metabolic_risk_score(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        UAE only (the only dataset with comorbidity-history flags, BMI,
        and cholesterol together). This is a transparent, simple additive
        composite for ML/SHAP purposes - it is NOT a validated clinical
        index (e.g. not the Adult Treatment Panel metabolic syndrome
        criteria), and is documented as such.
        """
        cols = ["history_diabetes", "history_obesity", "history_dyslipidemia",
                "bmi", "cholesterol"]
        ref = CLINICAL_REFERENCE_VALUES

        def _compute(d: pd.DataFrame) -> pd.Series:
            comorbidity_sum = (
                _coerce_numeric(d["history_diabetes"]).fillna(0)
                + _coerce_numeric(d["history_obesity"]).fillna(0)
                + _coerce_numeric(d["history_dyslipidemia"]).fillna(0)
            )
            bmi_component = _coerce_numeric(d["bmi"]) / ref["bmi_obesity_threshold"]
            chol_component = (
                _coerce_numeric(d["cholesterol"]) / ref["cholesterol_borderline_high_mg_dl"]
            )
            return comorbidity_sum + bmi_component + chol_component

        return self._try_create_feature(
            df, key, "metabolic_risk_score", cols,
            compute_fn=_compute,
            formula=(
                "(history_diabetes + history_obesity + history_dyslipidemia) "
                f"+ (bmi / {ref['bmi_obesity_threshold']}) "
                f"+ (cholesterol / {ref['cholesterol_borderline_high_mg_dl']})"
            ),
            clinical_rationale=(
                "Composite of metabolic comorbidity flags, BMI normalized "
                "against the WHO obesity threshold, and total cholesterol "
                "normalized against the ATP III borderline-high cutoff. "
                "Metabolic syndrome components are well-established "
                "accelerators of CKD progression. This is an engineered "
                "heuristic for ML use, not a validated clinical score."
            ),
        )

    def _add_anemia_risk_score(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        UCI / Kaggle (both carry hemoglobin, packed_cell_volume, and
        red_blood_cell_count - UAE does not have these columns and is
        skipped automatically via the column-availability check).

        Each marker's shortfall below a normal reference value contributes
        to the score; values at/above the reference contribute 0 (anemia
        risk should not go negative). Reference values are commonly-cited
        normal ranges (see CLINICAL_REFERENCE_VALUES), not derived from
        this dataset.
        """
        cols = ["hemoglobin", "packed_cell_volume", "red_blood_cell_count"]
        ref = CLINICAL_REFERENCE_VALUES

        def _shortfall(series: pd.Series, normal_value: float) -> pd.Series:
            ratio_deficit = 1.0 - (_coerce_numeric(series) / normal_value)
            return ratio_deficit.clip(lower=0.0)

        def _compute(d: pd.DataFrame) -> pd.Series:
            return (
                _shortfall(d["hemoglobin"], ref["hemoglobin_normal_g_dl"])
                + _shortfall(d["packed_cell_volume"], ref["pcv_normal_pct"])
                + _shortfall(d["red_blood_cell_count"], ref["rbc_normal_million_per_ul"])
            )

        return self._try_create_feature(
            df, key, "anemia_risk_score", cols,
            compute_fn=_compute,
            formula=(
                "sum over {hemoglobin, packed_cell_volume, red_blood_cell_count} "
                "of max(0, 1 - value / normal_reference)"
            ),
            clinical_rationale=(
                "Anemia of chronic kidney disease is a well-documented "
                "complication driven by reduced erythropoietin production; "
                "combining the three principal red-cell markers into a "
                "single shortfall-based score captures this signal more "
                "robustly than any one marker alone."
            ),
        )

    def _add_cardiovascular_burden_score(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        UCI / Kaggle: hypertension + coronary_artery_disease + diabetes_mellitus
        UAE:          history_hypertension + history_chd + history_diabetes
        (the UAE-equivalent comorbidity-history flags), per project spec.
        """
        if key in ("uci", "kaggle"):
            cols = ["hypertension", "coronary_artery_disease", "diabetes_mellitus"]
            return self._try_create_feature(
                df, key, "cardiovascular_burden_score", cols,
                compute_fn=lambda d: (
                    _coerce_numeric(d["hypertension"]).fillna(0)
                    + _coerce_numeric(d["coronary_artery_disease"]).fillna(0)
                    + _coerce_numeric(d["diabetes_mellitus"]).fillna(0)
                ),
                formula="hypertension + coronary_artery_disease + diabetes_mellitus",
                clinical_rationale=(
                    "Simple comorbidity count (0-3) of the three classic "
                    "cardiovascular/metabolic risk factors most strongly "
                    "associated with CKD progression in the nephrology "
                    "literature."
                ),
            )
        if key == "uae":
            cols = ["history_hypertension", "history_chd", "history_diabetes"]
            df = self._try_create_feature(
                df, key, "cardiovascular_burden_score", cols,
                compute_fn=lambda d: (
                    _coerce_numeric(d["history_hypertension"]).fillna(0)
                    + _coerce_numeric(d["history_chd"]).fillna(0)
                    + _coerce_numeric(d["history_diabetes"]).fillna(0)
                ),
                formula="history_hypertension + history_chd + history_diabetes",
                clinical_rationale=(
                    "UAE-cohort equivalent of the UCI/Kaggle cardiovascular "
                    "burden score, using the matching comorbidity-history "
                    "flags so the feature name/semantics stay aligned "
                    "across datasets."
                ),
            )
            # UAE-only enrichment: this cohort uniquely also records
            # vascular-disease and smoking history, which are directly
            # cardiovascular-relevant and otherwise unused. Kept as a
            # separate, clearly-named extended feature rather than folded
            # into the cross-dataset-comparable score above, so the base
            # cardiovascular_burden_score remains an apples-to-apples
            # feature across all three datasets.
            ext_cols = ["history_hypertension", "history_chd", "history_diabetes",
                        "history_vascular", "history_smoking"]
            df = self._try_create_feature(
                df, key, "cardiovascular_burden_score_extended", ext_cols,
                compute_fn=lambda d: (
                    _coerce_numeric(d["history_hypertension"]).fillna(0)
                    + _coerce_numeric(d["history_chd"]).fillna(0)
                    + _coerce_numeric(d["history_diabetes"]).fillna(0)
                    + _coerce_numeric(d["history_vascular"]).fillna(0)
                    + _coerce_numeric(d["history_smoking"]).fillna(0)
                ),
                formula=(
                    "history_hypertension + history_chd + history_diabetes "
                    "+ history_vascular + history_smoking"
                ),
                clinical_rationale=(
                    "UAE-only extended cardiovascular burden score that "
                    "additionally incorporates vascular disease and smoking "
                    "history, both available in this cohort and clinically "
                    "relevant but absent from UCI/Kaggle."
                ),
            )
            return df
        return df

    def _add_age_creatinine_interaction(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["age", "serum_creatinine"]
        return self._try_create_feature(
            df, key, "age_creatinine_interaction", cols,
            compute_fn=lambda d: (
                _coerce_numeric(d["age"]) * _coerce_numeric(d["serum_creatinine"])
            ),
            formula="age * serum_creatinine",
            clinical_rationale=(
                "Renal impairment from a given creatinine elevation carries "
                "more clinical weight in older patients (age-related decline "
                "in baseline renal reserve); this interaction lets "
                "tree-based models split on the combined effect directly."
            ),
        )

    def _add_urea_creatinine_product(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["blood_urea", "serum_creatinine"]
        return self._try_create_feature(
            df, key, "urea_creatinine_product", cols,
            compute_fn=lambda d: (
                _coerce_numeric(d["blood_urea"]) * _coerce_numeric(d["serum_creatinine"])
            ),
            formula="blood_urea * serum_creatinine",
            clinical_rationale=(
                "Captures overall nitrogenous-waste retention burden as a "
                "joint magnitude, complementing the BUN/creatinine RATIO "
                "(which captures their relative balance, not their "
                "combined severity)."
            ),
        )

    def _add_hemoglobin_creatinine_ratio(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["hemoglobin", "serum_creatinine"]
        return self._try_create_feature(
            df, key, "hemoglobin_creatinine_ratio", cols,
            compute_fn=lambda d: _safe_divide(
                _coerce_numeric(d["hemoglobin"]), _coerce_numeric(d["serum_creatinine"])
            ),
            formula="hemoglobin / serum_creatinine",
            clinical_rationale=(
                "Links anemia severity to renal impairment in a single "
                "feature - both worsen together in progressive CKD due to "
                "reduced erythropoietin production, so their ratio tends to "
                "fall faster than either marker individually."
            ),
        )

    def _add_albumin_specific_gravity_product(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["albumin", "specific_gravity"]
        return self._try_create_feature(
            df, key, "albumin_specific_gravity_interaction", cols,
            compute_fn=lambda d: (
                _coerce_numeric(d["albumin"]) * _coerce_numeric(d["specific_gravity"])
            ),
            formula="albumin * specific_gravity",
            clinical_rationale=(
                "Urinary albumin and specific gravity are both glomerular/"
                "tubular function indicators on urinalysis; their product "
                "highlights cases where proteinuria co-occurs with abnormal "
                "urine concentrating ability."
            ),
        )

    def _add_egfr_age_ratio(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        cols = ["egfr", "age"]
        return self._try_create_feature(
            df, key, "egfr_age_ratio", cols,
            compute_fn=lambda d: _safe_divide(
                _coerce_numeric(d["egfr"]), _coerce_numeric(d["age"])
            ),
            formula="egfr / age",
            clinical_rationale=(
                "Normalizes kidney filtration capacity by age, since eGFR "
                "naturally declines with normal aging; a low ratio flags "
                "filtration loss beyond what age alone would predict."
            ),
        )

    def _add_pulse_pressure(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """UAE only - the only dataset with separate systolic/diastolic readings."""
        cols = ["systolic_bp", "diastolic_bp"]
        return self._try_create_feature(
            df, key, "pulse_pressure", cols,
            compute_fn=lambda d: (
                _coerce_numeric(d["systolic_bp"]) - _coerce_numeric(d["diastolic_bp"])
            ),
            formula="systolic_bp - diastolic_bp",
            clinical_rationale=(
                "Widened pulse pressure is a well-established marker of "
                "arterial stiffness and independently associated with "
                "cardiovascular and renal outcomes."
            ),
        )

    def _add_comorbidity_count(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """UAE only - the only dataset with a full comorbidity-history panel."""
        cols = ["history_diabetes", "history_chd", "history_vascular",
                "history_smoking", "history_hypertension",
                "history_dyslipidemia", "history_obesity"]
        return self._try_create_feature(
            df, key, "comorbidity_count", cols,
            compute_fn=lambda d: sum(
                _coerce_numeric(d[c]).fillna(0) for c in cols
            ),
            formula="sum of all history_* comorbidity flags",
            clinical_rationale=(
                "Overall comorbidity burden is one of the strongest general "
                "predictors of disease progression and mortality risk in "
                "cohort studies, independent of any single condition."
            ),
        )

    def _add_medication_burden(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """UAE only - the only dataset with medication-class flags."""
        cols = ["dld_meds", "diabetes_meds", "hypertension_meds", "acei_arb"]
        return self._try_create_feature(
            df, key, "medication_burden", cols,
            compute_fn=lambda d: sum(
                _coerce_numeric(d[c]).fillna(0) for c in cols
            ),
            formula="dld_meds + diabetes_meds + hypertension_meds + acei_arb",
            clinical_rationale=(
                "Number of distinct medication classes a patient is on is a "
                "practical proxy for disease-management intensity and "
                "multimorbidity, and ACEI/ARB use in particular is directly "
                "relevant to renal hemodynamics."
            ),
        )

    # -------------------------------------------------------------------
    # Leakage protection: exclusion of known target-proxy columns
    # -------------------------------------------------------------------

    def _exclude_leakage_columns(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """
        Removes any column registered in LEAKAGE_PRONE_COLUMNS for this
        dataset BEFORE feature creation runs, so no engineered feature can
        ever reference them (defense in depth, on top of the fact that no
        feature definition references them by construction).

        Excluded columns are NOT silently discarded: their values are kept
        in self._excluded_columns_by_dataset and written to a dedicated
        '<dataset>_excluded_leakage_columns.csv' audit file (row-position
        aligned, same pattern as the provenance file) so the exclusion is
        fully reversible/inspectable - this is a documented removal, not a
        silent one.
        """
        registry = LEAKAGE_PRONE_COLUMNS.get(dataset_key, {})
        present = [c for c in registry if c in df.columns]

        if not present:
            return df

        excluded_df = df[present].copy()
        excluded_df.insert(0, "row_position", range(len(excluded_df)))
        self._excluded_columns_by_dataset[dataset_key] = excluded_df

        for col in present:
            n_values = int(df[col].notna().sum())
            rationale = registry[col]
            self._leakage_exclusions.append(LeakageExclusionRecord(
                column=col, dataset=dataset_key, rationale=rationale,
                n_values_excluded=n_values,
            ))
            self.logger.warning(
                "[%s] Excluding column '%s' as a target-leakage risk: %s",
                dataset_key, col, rationale,
            )

        return df.drop(columns=present)

    # -------------------------------------------------------------------
    # Leakage protection: generic statistical screen (informational only)
    # -------------------------------------------------------------------

    def _audit_statistical_leakage(self, df: pd.DataFrame, dataset_key: str) -> None:
        """
        Screens low-cardinality (categorical/flag-like) columns for
        suspiciously strong association with this dataset's target, using
        normalized mutual information. This is deliberately restricted to
        low-cardinality columns - continuous lab values (serum_creatinine,
        egfr, ...) are EXPECTED to correlate strongly with the target and
        screening them here would just flag every genuinely useful
        biomarker, which is not the goal.

        Results are purely informational: they are logged and written to
        feature_summary.json for human/domain review. Nothing is removed
        based on this screen alone - statistical association cannot, on
        its own, distinguish a genuine strong predictor from a target
        proxy; that judgment call requires domain reasoning (as was used
        for ckd_binary_label / ckd_affected above).
        """
        target_col = self.TARGET_COLUMNS.get(dataset_key)
        if not target_col or target_col not in df.columns:
            return
        if not _SKLEARN_AVAILABLE:
            self.logger.warning(
                "[%s] scikit-learn not available - skipping statistical "
                "leakage screen (this is informational tooling only; it "
                "does not affect feature creation).",
                dataset_key,
            )
            return

        already_known = set(LEAKAGE_PRONE_COLUMNS.get(dataset_key, {}).keys())
        already_known |= TARGET_LIKE_COLUMNS | PROVENANCE_COLUMNS | {target_col}

        target_values = df[target_col]
        valid_target_mask = target_values.notna()

        for col in df.columns:
            if col in already_known:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique < 2 or n_unique > LEAKAGE_AUDIT_MAX_CARDINALITY:
                continue  # not categorical/flag-like, or constant - skip

            mask = valid_target_mask & df[col].notna()
            if mask.sum() < 2:
                continue

            try:
                nmi = normalized_mutual_info_score(
                    df.loc[mask, col].astype(str),
                    target_values.loc[mask].astype(str),
                )
            except ValueError:
                continue

            if nmi >= LEAKAGE_AUDIT_NMI_FLAG_THRESHOLD:
                self._statistical_flags.append(StatisticalLeakageFlag(
                    column=col, dataset=dataset_key, target_column=target_col,
                    normalized_mutual_info=float(nmi), n_unique_values=int(n_unique),
                ))
                self.logger.warning(
                    "[%s] Statistical leakage screen: column '%s' has high "
                    "normalized mutual information (%.3f) with target '%s'. "
                    "Flagged for human review in feature_summary.json - NOT "
                    "auto-removed.",
                    dataset_key, col, nmi, target_col,
                )

    # -------------------------------------------------------------------
    # Per-dataset orchestration
    # -------------------------------------------------------------------

    def engineer_dataset(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """
        Apply every applicable feature definition to a single dataset's
        DataFrame. Each call is independent - this method never looks at
        another dataset's DataFrame, which is what keeps UAE structurally
        isolated from UCI/Kaggle at the feature-engineering stage.

        Leakage protection runs FIRST, before any feature is created:
            1. Known target-proxy columns (LEAKAGE_PRONE_COLUMNS) are
               physically removed from the working DataFrame.
            2. A statistical screen flags (but does not remove) any other
               low-cardinality column with suspiciously high association
               with the target, for human review.
        """
        self.logger.info("[%s] Starting feature engineering …", dataset_key)
        df = df.copy()

        df = self._exclude_leakage_columns(df, dataset_key)
        self._audit_statistical_leakage(df, dataset_key)

        df = self._add_bun_creatinine_ratio(df, dataset_key)
        df = self._add_sodium_potassium_ratio(df, dataset_key)
        df = self._add_kidney_dysfunction_score(df, dataset_key)
        df = self._add_bp_risk_score(df, dataset_key)
        df = self._add_metabolic_risk_score(df, dataset_key)
        df = self._add_anemia_risk_score(df, dataset_key)
        df = self._add_cardiovascular_burden_score(df, dataset_key)
        df = self._add_age_creatinine_interaction(df, dataset_key)
        df = self._add_urea_creatinine_product(df, dataset_key)
        df = self._add_hemoglobin_creatinine_ratio(df, dataset_key)
        df = self._add_albumin_specific_gravity_product(df, dataset_key)
        df = self._add_egfr_age_ratio(df, dataset_key)
        df = self._add_pulse_pressure(df, dataset_key)
        df = self._add_comorbidity_count(df, dataset_key)
        df = self._add_medication_burden(df, dataset_key)

        n_created = sum(
            1 for r in self._records if r.dataset == dataset_key and r.created
        )
        self.logger.info(
            "[%s] Feature engineering complete: %d feature(s) created, final shape %s.",
            dataset_key, n_created, df.shape,
        )
        return df

    # -------------------------------------------------------------------
    # Saving outputs
    # -------------------------------------------------------------------

    def _save_outputs(self, dataset_key: str, df: pd.DataFrame) -> None:
        """
        Drops provenance columns from the engineered dataset (per the
        project's architectural rule) and saves them separately, row-order
        aligned via an explicit `row_position` column.
        """
        self.engineered_dir.mkdir(parents=True, exist_ok=True)

        present_provenance_cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
        provenance_df = df[present_provenance_cols].copy() if present_provenance_cols else pd.DataFrame()
        if not provenance_df.empty:
            provenance_df.insert(0, "row_position", range(len(provenance_df)))

        engineered_df = df.drop(columns=present_provenance_cols, errors="ignore")

        engineered_path = self.engineered_dir / f"{dataset_key}_engineered.csv"
        engineered_df.to_csv(engineered_path, index=False)
        self.logger.info(
            "[%s] Saved engineered dataset: %s (%d rows, %d cols).",
            dataset_key, engineered_path, *engineered_df.shape,
        )

        if not provenance_df.empty:
            provenance_path = self.engineered_dir / f"{dataset_key}_provenance.csv"
            provenance_df.to_csv(provenance_path, index=False)
            self.logger.info(
                "[%s] Saved provenance file: %s (join key: row_position).",
                dataset_key, provenance_path,
            )
        else:
            self.logger.warning(
                "[%s] No provenance columns (%s) found in the processed "
                "input - nothing to save separately.",
                dataset_key, sorted(PROVENANCE_COLUMNS),
            )

        excluded_df = self._excluded_columns_by_dataset.get(dataset_key)
        if excluded_df is not None and not excluded_df.empty:
            excluded_path = self.engineered_dir / f"{dataset_key}_excluded_leakage_columns.csv"
            excluded_df.to_csv(excluded_path, index=False)
            self.logger.warning(
                "[%s] Saved EXCLUDED leakage-risk columns to a separate "
                "audit file (NOT included in the engineered dataset): %s "
                "(join key: row_position). See feature_summary.json -> "
                "leakage_audit.excluded_columns for the rationale.",
                dataset_key, excluded_path,
            )

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------

    def _write_feature_summary(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        per_dataset: Dict[str, Any] = {}
        for dataset_key in ("uci", "kaggle", "uae"):
            ds_records = [r for r in self._records if r.dataset == dataset_key]
            created = [r for r in ds_records if r.created]
            skipped = [r for r in ds_records if not r.created]
            per_dataset[dataset_key] = {
                "rows_processed": self._rows_processed.get(dataset_key, 0),
                "original_feature_count": self._original_feature_counts.get(dataset_key, 0),
                "engineered_feature_count": len(created),
                "total_feature_count_after_engineering": (
                    self._original_feature_counts.get(dataset_key, 0) + len(created)
                ),
                "features_created": [r.as_dict() for r in created],
                "features_skipped": [r.as_dict() for r in skipped],
            }

        feature_definitions: Dict[str, Any] = {}
        for name, meta in self._definitions.items():
            datasets_created_in = sorted({
                r.dataset for r in self._records if r.feature_name == name and r.created
            })
            feature_definitions[name] = {
                "formula": meta.formula,
                "clinical_rationale": meta.clinical_rationale,
                "datasets_created_in": datasets_created_in,
            }

        leakage_audit: Dict[str, Any] = {}
        for dataset_key in ("uci", "kaggle", "uae"):
            exclusions = [r for r in self._leakage_exclusions if r.dataset == dataset_key]
            flags = [f for f in self._statistical_flags if f.dataset == dataset_key]
            leakage_audit[dataset_key] = {
                "excluded_columns": [r.as_dict() for r in exclusions],
                "statistical_screen_flags": [f.as_dict() for f in flags],
            }

        summary = {
            "pipeline_stage": "feature_engineering",
            "data_leakage_policy": (
                "No engineered feature reads any of: "
                f"{sorted(TARGET_LIKE_COLUMNS)}. All features are deterministic "
                "row-wise transforms of existing columns - nothing is fit or "
                "estimated from the data, so there is no possibility of "
                "train/validation leakage at this stage, and UAE is never "
                "merged with or statistically influenced by UCI/Kaggle."
            ),
            "provenance_handling": (
                f"{sorted(PROVENANCE_COLUMNS)} were dropped from every engineered "
                "dataset and saved separately as <dataset>_provenance.csv, "
                "row-order aligned via an explicit row_position column."
            ),
            "leakage_audit": {
                "policy": (
                    "Two layers of protection: (1) columns in LEAKAGE_PRONE_COLUMNS "
                    "are physically removed before feature creation and saved to "
                    "<dataset>_excluded_leakage_columns.csv for auditability "
                    "(see 'excluded_columns' below, with rationale per column); "
                    "(2) a generic statistical screen flags any other "
                    "low-cardinality column with high normalized mutual "
                    "information against the target for human review "
                    "(see 'statistical_screen_flags' below) - this screen is "
                    "informational only and never triggers automatic removal, "
                    "since strong association from a genuine clinical "
                    "biomarker is expected and desirable, not a leakage signal."
                ),
                "nmi_flag_threshold": LEAKAGE_AUDIT_NMI_FLAG_THRESHOLD,
                "max_cardinality_screened": LEAKAGE_AUDIT_MAX_CARDINALITY,
                "datasets": leakage_audit,
            },
            "clinical_reference_values_used": CLINICAL_REFERENCE_VALUES,
            "datasets": per_dataset,
            "feature_definitions": feature_definitions,
        }

        json_path = self.artifacts_dir / "feature_summary.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        self.logger.info("Saved feature summary: %s", json_path)

        joblib_path = self.artifacts_dir / "feature_summary.joblib"
        joblib.dump(summary, joblib_path)
        self.logger.info("Saved feature summary (joblib): %s", joblib_path)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    engineer = FeatureEngineer()
    uci_out, kaggle_out, uae_out = engineer.run()

    print("\n── Feature engineering complete ──")
    print(f"UCI engineered shape    : {uci_out.shape}")
    print(f"Kaggle engineered shape : {kaggle_out.shape}")
    print(f"UAE engineered shape    : {uae_out.shape}")
    print(
        "\nDatasets  -> data/engineered/\n"
        "Artifacts -> artifacts/feature_engineering/feature_summary.json"
    )