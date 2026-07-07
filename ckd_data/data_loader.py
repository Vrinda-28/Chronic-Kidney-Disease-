"""
data_loader.py
================

Production-grade data ingestion layer for the CKD (Chronic Kidney Disease)
prediction research project.

Scope (intentionally limited):
    - Load the three source datasets (UCI, Kaggle-staged, UAE external).
    - Validate file existence and schema.
    - Standardize column names via a configurable schema-mapping (no
      hardcoded mappings inside functions - see config/datasets.yaml).
    - Attach provenance (`source_dataset`, `original_row_id`) to every row.
    - Produce data-quality reports (duplicates, missing values, dtypes,
      class distribution, basic outliers) WITHOUT mutating or dropping data.
    - Keep the UAE external-validation cohort structurally isolated from
      anything that could be used for training.

Explicitly OUT of scope for this module:
    - Feature engineering, imputation, encoding.
    - Train/test splitting, resampling (SMOTE, class weighting).
    - Model training or evaluation.
    - Any mutation of label/feature values (including whitespace trimming) -
      see _check_target_label_hygiene() for why this is a read-only check.

These will live in separate modules later. This loader only ever returns
raw-but-standardized, provenance-tagged DataFrames plus metadata/reports.
"""

from __future__ import annotations

import hashlib
import logging
import logging.handlers
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Bump this whenever loading/standardization/validation LOGIC changes in a
# way that could alter output. Stored in DatasetMetadata for paper-level
# reproducibility ("which version of the loader produced this dataframe?").
LOADER_VERSION = "1.1.0"


# =============================================================================
# Exceptions
# =============================================================================

class CKDDataError(Exception):
    """Base exception for all data-loading errors in this module."""


class DatasetFileNotFoundError(CKDDataError):
    """Raised when a configured dataset CSV path does not exist on disk."""


class ConfigurationError(CKDDataError):
    """Raised when the YAML configuration is missing or malformed."""


class CriticalSchemaError(CKDDataError):
    """
    Raised when a dataset is missing its TARGET column after standardization,
    or when schema standardization produces MULTIPLE columns sharing the
    target column's standardized name (an unusable, ambiguous target).
    Missing non-target columns are reported as warnings, not raised, since
    research datasets are frequently incomplete.
    """


# =============================================================================
# Logging
# =============================================================================

def build_logger(log_dir: str, log_filename: str,
                  console_level: str = "INFO",
                  file_level: str = "DEBUG") -> logging.Logger:
    """
    Configure and return a module-level logger that writes structured
    output both to the console and to a rotating log file.

    Idempotent: calling this multiple times will not duplicate handlers.
    """
    logger = logging.getLogger("ckd_data_loader")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        # Already configured (e.g. re-imported in a notebook/session).
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
        file_path = os.path.join(log_dir, log_filename)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Logging to disk is best-effort; never let it crash data loading.
        logger.warning("Could not set up file logging at %s: %s", log_dir, exc)

    return logger


# =============================================================================
# Data classes (structured outputs)
# =============================================================================

@dataclass
class DatasetMetadata:
    """Descriptive metadata captured for every loaded dataset."""

    name: str
    display_name: str
    source_path: str
    role: str                      # "train_candidate" | "external_validation"
    row_count: int
    column_count: int
    target_column: str
    target_type: str
    class_distribution: Dict[Any, int] = field(default_factory=dict)
    missing_required_columns: List[str] = field(default_factory=list)
    unmapped_raw_columns: List[str] = field(default_factory=list)
    schema_warnings: List[str] = field(default_factory=list)
    # --- Reproducibility fields (Issue 5) ---
    loader_version: str = LOADER_VERSION
    file_checksum_sha256: str = ""
    schema_hash: str = ""
    dataset_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "source_path": self.source_path,
            "role": self.role,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "class_distribution": self.class_distribution,
            "missing_required_columns": self.missing_required_columns,
            "unmapped_raw_columns": self.unmapped_raw_columns,
            "schema_warnings": self.schema_warnings,
            "loader_version": self.loader_version,
            "file_checksum_sha256": self.file_checksum_sha256,
            "schema_hash": self.schema_hash,
            "dataset_hash": self.dataset_hash,
        }


@dataclass
class DataQualityReport:
    """Read-only quality report. Never causes rows to be dropped or modified."""

    dataset_name: str
    n_duplicate_rows: int
    n_duplicate_patients: Optional[int]
    missing_value_summary: Dict[str, int]
    missing_value_percent: Dict[str, float]
    dtype_summary: Dict[str, str]
    class_distribution: Dict[Any, int]
    outlier_report: Dict[str, int]
    # --- Issue 6 additions ---
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: Dict[str, int] = field(default_factory=dict)
    duplicate_raw_column_names: List[str] = field(default_factory=list)
    target_distribution_warnings: List[str] = field(default_factory=list)
    # --- Issue 1 / Issue 3 additions ---
    n_missing_target_labels: int = 0
    target_whitespace_anomalies: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_duplicate_patients": self.n_duplicate_patients,
            "missing_value_summary": self.missing_value_summary,
            "missing_value_percent": self.missing_value_percent,
            "dtype_summary": self.dtype_summary,
            "class_distribution": self.class_distribution,
            "outlier_report": self.outlier_report,
            "constant_columns": self.constant_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "duplicate_raw_column_names": self.duplicate_raw_column_names,
            "target_distribution_warnings": self.target_distribution_warnings,
            "n_missing_target_labels": self.n_missing_target_labels,
            "target_whitespace_anomalies": self.target_whitespace_anomalies,
        }


@dataclass
class CKDDataBundle:
    """
    Final structured output of CKDDataLoader.load_all().

    train_candidate_datasets and external_validation_dataset are kept as
    SEPARATE dictionaries on purpose. There is no method on this class
    that concatenates them - that isolation is enforced structurally,
    not just by convention.
    """

    train_candidate_datasets: Dict[str, pd.DataFrame]
    external_validation_dataset: Dict[str, pd.DataFrame]
    metadata: Dict[str, DatasetMetadata]
    quality_reports: Dict[str, DataQualityReport]

    def summary(self) -> Dict[str, Any]:
        """Lightweight, JSON-serializable summary for logging/inspection."""
        return {
            "train_candidate_datasets": list(self.train_candidate_datasets.keys()),
            "external_validation_dataset": list(self.external_validation_dataset.keys()),
            "metadata": {k: v.as_dict() for k, v in self.metadata.items()},
            "quality_reports": {k: v.as_dict() for k, v in self.quality_reports.items()},
        }


# =============================================================================
# Configuration loading
# =============================================================================

class DataConfig:
    """
    Thin wrapper around config/datasets.yaml. Centralizes every path and
    every schema mapping so the rest of the module never hardcodes them.
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise ConfigurationError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "datasets" not in raw:
            raise ConfigurationError(
                f"Config file {self.config_path} is missing a top-level 'datasets' key."
            )

        self.datasets: Dict[str, Dict[str, Any]] = raw["datasets"]
        self.schema_mapping: Dict[str, Dict[str, str]] = raw.get("schema_mapping", {})
        self.logging_config: Dict[str, Any] = raw.get("logging", {})
        self.outlier_columns: List[str] = raw.get("numeric_columns_for_outlier_check", [])
        self.high_cardinality_threshold_ratio: float = raw.get(
            "high_cardinality_threshold_ratio", 0.9
        )

        # The directory the config file lives in is used as the base for
        # relative dataset paths, so the loader works regardless of CWD.
        self.base_dir = self.config_path.parent.parent

    def resolve_path(self, relative_or_absolute_path: str) -> Path:
        p = Path(relative_or_absolute_path)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def get_dataset_config(self, key: str) -> Dict[str, Any]:
        if key not in self.datasets:
            raise ConfigurationError(f"No dataset config found for key '{key}'.")
        return self.datasets[key]

    def get_schema_mapping(self, key: str) -> Dict[str, str]:
        return self.schema_mapping.get(key, {})


# =============================================================================
# Schema standardization
# =============================================================================

class SchemaStandardizer:
    """
    Renames raw, source-specific column names to the unified schema using
    a configurable mapping dictionary (loaded from YAML, never hardcoded
    in this class's logic).
    """

    @staticmethod
    def normalize_key(raw_name: str) -> str:
        """
        Normalize a raw column name into a lookup-friendly form:
        lowercase, stripped, internal whitespace/punctuation collapsed
        to single underscores. e.g. "Blood Pressure " -> "blood_pressure".
        """
        name = str(raw_name).strip().lower()
        name = re.sub(r"[\s\-]+", "_", name)
        name = re.sub(r"[^\w]", "", name)
        return name

    def standardize(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, str],
        dataset_name: str,
        logger: logging.Logger,
        target_column: Optional[str] = None,
    ) -> "tuple[pd.DataFrame, list[str], list[str], list[tuple[str, str, str]]]":
        """
        Returns (renamed_df, unmapped_raw_columns, target_collision_names,
        auto_disambiguated).

        unmapped: raw columns with no mapping entry (kept, normalized, flagged).

        target_collision_names: non-empty only if >1 raw column mapped onto
        `target_column` itself. This is NOT auto-resolved here - it is
        surfaced so the caller can raise CriticalSchemaError, because
        guessing which raw column is "the real" target is unsafe.

        auto_disambiguated: for any OTHER standardized name that >1 raw
        column collapsed onto (e.g. both "Age" and "AgeBaseline" mapping to
        "age" - a realistic, legitimate occurrence, not necessarily an
        error), each colliding column is renamed to
        "<standardized>__raw_<normalized_raw_name>" instead of silently
        sharing one name. This is required for correctness: pandas allows
        duplicate column labels, but a duplicated label means df[col]
        returns a DataFrame instead of a Series, which silently breaks
        every downstream quality-check method (outlier_report,
        dtype_summary, class_distribution, etc). No data is dropped or
        merged - both columns are preserved, just under distinguishable
        names, and the collision is logged loudly for schema_mapping review.
        """
        normalized_mapping = {
            self.normalize_key(k): v for k, v in mapping.items()
        }

        raw_to_standard: Dict[str, str] = {}
        unmapped: List[str] = []

        for col in df.columns:
            norm = self.normalize_key(col)
            if norm in normalized_mapping:
                raw_to_standard[col] = normalized_mapping[norm]
            else:
                raw_to_standard[col] = norm
                unmapped.append(col)

        # Count how many raw columns land on each standardized name.
        name_counts: Dict[str, int] = {}
        for std_name in raw_to_standard.values():
            name_counts[std_name] = name_counts.get(std_name, 0) + 1

        target_collision_names: List[str] = []
        auto_disambiguated: List[tuple] = []
        final_columns: Dict[str, str] = {}

        for raw_col, std_name in raw_to_standard.items():
            if name_counts[std_name] > 1:
                if target_column is not None and std_name == target_column:
                    # Leave colliding target columns AS-IS (still colliding) -
                    # the caller is responsible for raising on this.
                    final_columns[raw_col] = std_name
                    if std_name not in target_collision_names:
                        target_collision_names.append(std_name)
                else:
                    disambiguated_name = f"{std_name}__raw_{self.normalize_key(raw_col)}"
                    final_columns[raw_col] = disambiguated_name
                    auto_disambiguated.append((raw_col, std_name, disambiguated_name))
            else:
                final_columns[raw_col] = std_name

        renamed = df.rename(columns=final_columns)

        if target_collision_names:
            logger.error(
                "[%s] Schema mapping produced %d raw column(s) all mapped to "
                "the TARGET column name '%s'. This is not auto-resolved - "
                "fix config/datasets.yaml schema_mapping before this dataset "
                "can be used.",
                dataset_name, name_counts.get(target_collision_names[0], 0),
                target_collision_names[0],
            )

        if auto_disambiguated:
            logger.warning(
                "[%s] %d raw column(s) collided on a standardized name and "
                "were auto-disambiguated (no data dropped, both kept under "
                "distinct names): %s. Review config/datasets.yaml "
                "schema_mapping to confirm this is intentional.",
                dataset_name, len(auto_disambiguated),
                [(r, s, d) for r, s, d in auto_disambiguated],
            )

        if unmapped:
            logger.warning(
                "[%s] %d raw column(s) had no schema mapping entry and were "
                "passed through normalized but unmapped: %s",
                dataset_name, len(unmapped), unmapped,
            )

        return renamed, unmapped, target_collision_names, auto_disambiguated


# =============================================================================
# Validation
# =============================================================================

class DataValidator:
    """File-existence and schema validation. Raises only on critical issues."""

    @staticmethod
    def validate_file_exists(path: Path, dataset_name: str) -> None:
        if not path.exists():
            raise DatasetFileNotFoundError(
                f"[{dataset_name}] Expected CSV at '{path}' but the file does not exist."
            )
        if path.stat().st_size == 0:
            raise DatasetFileNotFoundError(
                f"[{dataset_name}] File at '{path}' exists but is empty."
            )

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_columns: List[str], dataset_name: str
    ) -> List[str]:
        """Returns the list of missing required columns (does not raise)."""
        missing = [c for c in required_columns if c not in df.columns]
        return missing

    @staticmethod
    def validate_target_present(
        df: pd.DataFrame, target_column: str, dataset_name: str
    ) -> None:
        """Target column missing is treated as critical - raises."""
        if target_column not in df.columns:
            raise CriticalSchemaError(
                f"[{dataset_name}] Target column '{target_column}' is missing "
                f"after schema standardization. Available columns: "
                f"{list(df.columns)}"
            )

    @staticmethod
    def validate_target_unambiguous(
        target_collision_names: List[str], target_column: str, dataset_name: str
    ) -> None:
        """
        Raises if SchemaStandardizer reported that >1 raw column mapped onto
        the target column's standardized name. A duplicated target column is
        not a "warn and continue" situation: every downstream consumer
        (class_distribution, training) assumes a single Series, not a
        DataFrame, and would fail or silently misbehave.
        """
        if target_column in target_collision_names:
            raise CriticalSchemaError(
                f"[{dataset_name}] Schema mapping produced multiple columns "
                f"named '{target_column}' (the target column) - two or more "
                f"raw columns were mapped to the same target name. Fix the "
                f"ambiguous/duplicate entry in config/datasets.yaml "
                f"schema_mapping before proceeding."
            )

    @staticmethod
    def validate_row_count(
        df: pd.DataFrame, expected_min: Optional[int], expected_max: Optional[int],
        dataset_name: str, logger: logging.Logger,
    ) -> None:
        n = len(df)
        if expected_min is not None and n < expected_min:
            logger.warning(
                "[%s] Row count %d is below the expected minimum (%d). "
                "Confirm the correct file was loaded.",
                dataset_name, n, expected_min,
            )
        if expected_max is not None and n > expected_max:
            logger.warning(
                "[%s] Row count %d exceeds the expected maximum (%d). "
                "Confirm the correct file was loaded.",
                dataset_name, n, expected_max,
            )


# =============================================================================
# Data quality checks (report-only - never mutates or drops data)
# =============================================================================

class DataQualityChecker:
    """
    Generates descriptive data-quality reports. By design, none of these
    methods modify the input DataFrame or remove any rows - they only
    observe and report, per project requirements.
    """

    def __init__(self, outlier_columns: Optional[List[str]] = None,
                 high_cardinality_threshold_ratio: float = 0.9):
        self.outlier_columns = outlier_columns or []
        self.high_cardinality_threshold_ratio = high_cardinality_threshold_ratio

    def duplicate_rows(self, df: pd.DataFrame, exclude_cols: List[str]) -> int:
        """Exact duplicate rows, ignoring provenance columns."""
        cols = [c for c in df.columns if c not in exclude_cols]
        if not cols:
            return 0
        return int(df.duplicated(subset=cols).sum())

    def duplicate_patients(
        self, df: pd.DataFrame, id_column: Optional[str]
    ) -> Optional[int]:
        """Duplicate patient identifiers, if an ID column is configured/present."""
        if not id_column or id_column not in df.columns:
            return None
        return int(df[id_column].duplicated().sum())

    def missing_value_summary(self, df: pd.DataFrame) -> "tuple[dict, dict]":
        counts = df.isna().sum()
        percents = (counts / max(len(df), 1) * 100).round(2)
        return counts.to_dict(), percents.to_dict()

    def dtype_summary(self, df: pd.DataFrame) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def class_distribution(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Dict[Any, int]:
        if not target_column or target_column not in df.columns:
            return {}
        return df[target_column].value_counts(dropna=False).to_dict()

    def outlier_report(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Basic IQR-based outlier counts for configured numeric columns.
        Purely descriptive - used to flag columns worth inspecting later
        during feature engineering, not acted on here.
        """
        report: Dict[str, int] = {}
        for col in self.outlier_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                report[col] = 0
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            report[col] = n_outliers
        return report

    def constant_columns(self, df: pd.DataFrame, exclude_cols: List[str]) -> List[str]:
        """
        Columns with at most one distinct non-null value. Not removed here -
        zero-variance columns are still legitimate to *report*, since they
        carry no predictive signal and are worth flagging before feature
        engineering decides what to do with them.
        """
        constants = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 1:
                constants.append(col)
        return constants

    def high_cardinality_columns(
        self, df: pd.DataFrame, exclude_cols: List[str]
    ) -> Dict[str, int]:
        """
        Flags categorical/object columns whose number of unique values is
        suspiciously close to the row count (potential leaked identifiers,
        free-text fields, or accidental ID columns slipping into features).
        Returns {column: n_unique} for flagged columns only.
        """
        flagged: Dict[str, int] = {}
        n_rows = len(df)
        if n_rows == 0:
            return flagged
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype != object:
                continue
            n_unique = df[col].nunique(dropna=True)
            if n_unique == 0:
                continue
            if (n_unique / n_rows) >= self.high_cardinality_threshold_ratio and n_unique > 1:
                flagged[col] = int(n_unique)
        return flagged

    @staticmethod
    def duplicate_raw_column_names(path: Path) -> List[str]:
        """
        Detects EXACT duplicate column names in the raw CSV header, before
        pandas silently disambiguates them (e.g. "bp" and a second "bp"
        become "bp" and "bp.1" on read). This must be checked against the
        raw header line directly, since by the time the DataFrame exists
        the collision is already hidden.
        """
        import csv
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except (OSError, StopIteration):
            return []
        seen = set()
        dupes = []
        for col in header:
            key = col.strip()
            if key in seen and key not in dupes:
                dupes.append(key)
            seen.add(key)
        return dupes

    def suspicious_target_distribution(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> List[str]:
        """
        Read-only flags on the target distribution that matter for research
        validity (e.g. stratified splitting requires >=2 members per class).
        Does not drop or merge classes - only reports.
        """
        warnings_found: List[str] = []
        if not target_column or target_column not in df.columns:
            return warnings_found

        counts = df[target_column].value_counts(dropna=False)
        if counts.empty:
            return warnings_found

        singleton_classes = counts[counts == 1].index.tolist()
        if singleton_classes:
            warnings_found.append(
                f"{len(singleton_classes)} class(es) have only 1 sample: "
                f"{singleton_classes}. Stratified train/test splitting will "
                f"fail on these classes until addressed in preprocessing."
            )

        if len(counts) >= 2:
            ratio = counts.max() / max(counts.min(), 1)
            if ratio >= 10:
                warnings_found.append(
                    f"Severe class imbalance detected: majority/minority "
                    f"ratio is {ratio:.1f}:1 across classes {counts.to_dict()}. "
                    f"Flagged for the (separate, future) imbalance-handling "
                    f"stage - no resampling performed in this loader."
                )

        n_missing = int(df[target_column].isna().sum())
        if n_missing > 0:
            pct = round(n_missing / len(df) * 100, 2)
            warnings_found.append(
                f"{n_missing} row(s) ({pct}%) have a missing target label and "
                f"cannot be used for supervised training as-is. Rows are "
                f"preserved per the loader's no-row-removal policy; resolve "
                f"(drop/impute/exclude) explicitly in a preprocessing step."
            )

        return warnings_found

    def target_whitespace_anomalies(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> int:
        """
        Read-only count of non-null target values that differ from their
        own whitespace-stripped form (e.g. " ckd" vs "ckd"). Reported only -
        the loader does NOT rewrite values (see Issue 3 in the review notes).
        """
        if not target_column or target_column not in df.columns:
            return 0
        series = df[target_column].dropna()
        if series.empty:
            return 0
        as_str = series.astype(str)
        return int((as_str != as_str.str.strip()).sum())

    def run_all(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: Optional[str],
        id_column: Optional[str],
        provenance_cols: List[str],
        raw_path: Path,
    ) -> DataQualityReport:
        missing_counts, missing_pct = self.missing_value_summary(df)
        n_missing_target = (
            int(df[target_column].isna().sum())
            if target_column and target_column in df.columns else 0
        )
        return DataQualityReport(
            dataset_name=dataset_name,
            n_duplicate_rows=self.duplicate_rows(df, exclude_cols=provenance_cols),
            n_duplicate_patients=self.duplicate_patients(df, id_column),
            missing_value_summary=missing_counts,
            missing_value_percent=missing_pct,
            dtype_summary=self.dtype_summary(df),
            class_distribution=self.class_distribution(df, target_column),
            outlier_report=self.outlier_report(df),
            constant_columns=self.constant_columns(df, exclude_cols=provenance_cols),
            high_cardinality_columns=self.high_cardinality_columns(
                df, exclude_cols=provenance_cols
            ),
            duplicate_raw_column_names=self.duplicate_raw_column_names(raw_path),
            target_distribution_warnings=self.suspicious_target_distribution(
                df, target_column
            ),
            n_missing_target_labels=n_missing_target,
            target_whitespace_anomalies=self.target_whitespace_anomalies(
                df, target_column
            ),
        )


# =============================================================================
# Main loader
# =============================================================================

class CKDDataLoader:
    """
    Orchestrates loading, validating, standardizing, and reporting on the
    three CKD source datasets.

    Usage:
        loader = CKDDataLoader(config_path="config/datasets.yaml")
        bundle = loader.load_all()

        uci_df = bundle.train_candidate_datasets["uci"]
        kaggle_df = bundle.train_candidate_datasets["kaggle"]
        uae_df = bundle.external_validation_dataset["uae"]   # isolated
    """

    PROVENANCE_SOURCE_COL = "source_dataset"
    PROVENANCE_ROWID_COL = "original_row_id"

    def __init__(self, config_path: str = "config/datasets.yaml",
                 logger: Optional[logging.Logger] = None):
        self.config = DataConfig(config_path)

        log_cfg = self.config.logging_config
        self.logger = logger or build_logger(
            log_dir=str(self.config.resolve_path(log_cfg.get("log_dir", "logs"))),
            log_filename=log_cfg.get("log_filename", "data_loader.log"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )

        self.standardizer = SchemaStandardizer()
        self.validator = DataValidator()
        self.quality_checker = DataQualityChecker(
            self.config.outlier_columns,
            self.config.high_cardinality_threshold_ratio,
        )

    # -- reproducibility helpers ------------------------------------------

    @staticmethod
    def _file_checksum(path: Path) -> str:
        """SHA-256 of the raw file bytes, for exact-file provenance."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _schema_hash(df: pd.DataFrame, exclude_cols: List[str]) -> str:
        """
        Hash of the sorted, standardized column names - a structural
        fingerprint of "what schema does this dataframe have", independent
        of row order or values. Changes only if columns are added/removed/
        renamed.
        """
        cols = sorted(c for c in df.columns if c not in exclude_cols)
        return hashlib.sha256(",".join(cols).encode("utf-8")).hexdigest()

    @staticmethod
    def _dataset_hash(df: pd.DataFrame, exclude_cols: List[str]) -> str:
        """
        Content hash of the standardized dataframe (excluding provenance
        columns, which are derived/loader-generated, not source content).
        Lets a paper or downstream pipeline assert "this is byte-for-byte
        the same standardized dataset I trained/evaluated on".
        """
        cols = [c for c in df.columns if c not in exclude_cols]
        if not cols:
            return ""
        content_df = df[sorted(cols)]
        row_hashes = pd.util.hash_pandas_object(content_df, index=False)
        return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()

    # -- single dataset -------------------------------------------------

    def load_dataset(self, key: str) -> "tuple[pd.DataFrame, DatasetMetadata, DataQualityReport]":
        """
        Load and standardize a single configured dataset by its config key
        (e.g. "uci", "kaggle", "uae").
        """
        ds_cfg = self.config.get_dataset_config(key)
        source_name = ds_cfg.get("source_name", key.upper())
        display_name = ds_cfg.get("display_name", source_name)
        role = ds_cfg.get("role", "train_candidate")
        target_column = ds_cfg["target_column"]
        target_type = ds_cfg.get("target_type", "unknown")
        id_column = ds_cfg.get("id_column")
        required_columns = ds_cfg.get("required_columns", [])

        path = self.config.resolve_path(ds_cfg["path"])
        self.logger.info("[%s] Loading dataset from: %s", source_name, path)

        # 1. Validate file existence
        self.validator.validate_file_exists(path, source_name)

        # 1b. Reproducibility: checksum the raw file BEFORE any parsing.
        file_checksum = self._file_checksum(path)
        self.logger.info("[%s] Raw file SHA-256: %s", source_name, file_checksum)

        # 2. Read CSV
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse error clearly
            self.logger.error("[%s] Failed to read CSV at %s: %s", source_name, path, exc)
            raise CKDDataError(f"[{source_name}] Failed to read CSV at {path}: {exc}") from exc

        raw_row_count, raw_col_count = df.shape
        self.logger.info(
            "[%s] Raw file loaded: %d rows, %d columns.",
            source_name, raw_row_count, raw_col_count,
        )

        # 3. Provenance - captured BEFORE any column renaming, using the
        #    original CSV row order (0-indexed) as a stable identifier.
        df = df.copy()
        df[self.PROVENANCE_ROWID_COL] = df.index
        df[self.PROVENANCE_SOURCE_COL] = source_name

        # 4. Standardize schema using the configurable mapping.
        #    Provenance columns are excluded from mapping/unmapped-reporting
        #    since they are added by this loader, not part of the raw source
        #    schema, and are already in their final standardized form.
        mapping = self.config.get_schema_mapping(key)
        provenance_cols = {self.PROVENANCE_ROWID_COL, self.PROVENANCE_SOURCE_COL}
        feature_cols = [c for c in df.columns if c not in provenance_cols]

        standardized_features, unmapped_cols, target_collision_names, auto_disambiguated = (
            self.standardizer.standardize(
                df[feature_cols], mapping, source_name, self.logger,
                target_column=target_column,
            )
        )
        df = pd.concat(
            [standardized_features, df[list(provenance_cols)]], axis=1
        )

        # 5. Validate row-count sanity (warning only)
        self.validator.validate_row_count(
            df,
            ds_cfg.get("expected_min_rows"),
            ds_cfg.get("expected_max_rows"),
            source_name,
            self.logger,
        )

        # 6. Validate target column presence (critical - raises if absent)
        self.validator.validate_target_present(df, target_column, source_name)

        # 6b. Validate the target column is unambiguous (critical - raises
        #     if schema mapping collapsed >1 raw column onto it).
        self.validator.validate_target_unambiguous(
            target_collision_names, target_column, source_name
        )

        # 7. Validate required columns (non-critical - reported only)
        missing_required = self.validator.validate_required_columns(
            df, required_columns, source_name
        )
        if missing_required:
            self.logger.warning(
                "[%s] Missing %d required column(s) after standardization: %s",
                source_name, len(missing_required), missing_required,
            )

        # NOTE (Issue 3 fix): a previous revision mutated the target column
        # here via `.astype(str).str.strip()`. That was removed because:
        #   (a) `.astype(str)` silently turns NaN into the literal string
        #       "nan", which then hides missing labels from every downstream
        #       missing-value check (including the Issue 1 NaN tracking
        #       below) - a correctness bug, not just a scope violation.
        #   (b) Mutating label values is a preprocessing decision and
        #       belongs in a dedicated, audited preprocessing module, not
        #       silently inside ingestion.
        # The loader now only *reports* whitespace anomalies (see
        # DataQualityReport.target_whitespace_anomalies) without rewriting
        # anything. Actual trimming/cleaning should happen explicitly in
        # preprocessing, where it can be logged as a deliberate transform.

        # 8. Dataset-specific label-preservation rules (read-only checks).
        self._log_label_policy(df, key, source_name, target_column, target_type)

        # 9. Build metadata
        class_dist = self.quality_checker.class_distribution(df, target_column)
        schema_warnings: List[str] = []
        if auto_disambiguated:
            schema_warnings.append(
                f"{len(auto_disambiguated)} raw column(s) collided on a "
                f"standardized name and were auto-disambiguated: "
                f"{[(r, s, d) for r, s, d in auto_disambiguated]}"
            )
        if unmapped_cols:
            schema_warnings.append(
                f"{len(unmapped_cols)} raw column(s) had no schema mapping entry: "
                f"{unmapped_cols}"
            )

        metadata = DatasetMetadata(
            name=key,
            display_name=display_name,
            source_path=str(path),
            role=role,
            row_count=len(df),
            column_count=df.shape[1],
            target_column=target_column,
            target_type=target_type,
            class_distribution=class_dist,
            missing_required_columns=missing_required,
            unmapped_raw_columns=unmapped_cols,
            schema_warnings=schema_warnings,
            loader_version=LOADER_VERSION,
            file_checksum_sha256=file_checksum,
            schema_hash=self._schema_hash(df, exclude_cols=list(provenance_cols)),
            dataset_hash=self._dataset_hash(df, exclude_cols=list(provenance_cols)),
        )

        # 10. Build quality report (report-only, no mutation)
        quality_report = self.quality_checker.run_all(
            df,
            dataset_name=source_name,
            target_column=target_column,
            id_column=id_column,
            provenance_cols=[self.PROVENANCE_SOURCE_COL, self.PROVENANCE_ROWID_COL],
            raw_path=path,
        )

        self._log_summary(metadata, quality_report)

        return df, metadata, quality_report

    def _log_label_policy(
        self, df: pd.DataFrame, key: str, source_name: str,
        target_column: str, target_type: str,
    ) -> None:
        """Logs what label values were found, for manual confirmation that
        binary datasets stayed binary and staged datasets stayed staged."""
        if target_column not in df.columns:
            return
        observed_labels = sorted(
            df[target_column].dropna().unique().tolist(), key=str
        )
        self.logger.info(
            "[%s] Target column '%s' (declared type: %s) - observed labels: %s",
            source_name, target_column, target_type, observed_labels,
        )
        if target_type == "binary" and len(observed_labels) > 2:
            self.logger.warning(
                "[%s] Dataset is declared 'binary' but %d distinct label "
                "values were observed: %s. Labels are preserved as-is - "
                "no automatic collapsing/inference is performed here.",
                source_name, len(observed_labels), observed_labels,
            )
        if target_type == "multiclass" and len(observed_labels) <= 2:
            self.logger.warning(
                "[%s] Dataset is declared 'multiclass' (staged) but only "
                "%d distinct label value(s) were observed: %s. Labels are "
                "preserved as-is.",
                source_name, len(observed_labels), observed_labels,
            )

        n_missing = int(df[target_column].isna().sum())
        if n_missing > 0:
            self.logger.warning(
                "[%s] %d row(s) have a MISSING target label ('%s'). These "
                "rows are preserved (no automatic removal) - they must be "
                "explicitly excluded or imputed in a downstream "
                "preprocessing step before training.",
                source_name, n_missing, target_column,
            )

    def _log_summary(self, metadata: DatasetMetadata, report: DataQualityReport) -> None:
        self.logger.info(
            "[%s] Standardized shape: %d rows x %d cols | role=%s | target=%s",
            metadata.name, metadata.row_count, metadata.column_count,
            metadata.role, metadata.target_column,
        )
        self.logger.info(
            "[%s] Class distribution: %s", metadata.name, metadata.class_distribution
        )
        self.logger.info(
            "[%s] Duplicate rows: %d | Duplicate patient IDs: %s",
            metadata.name, report.n_duplicate_rows, report.n_duplicate_patients,
        )
        non_zero_missing = {
            k: v for k, v in report.missing_value_summary.items() if v > 0
        }
        if non_zero_missing:
            self.logger.info(
                "[%s] Columns with missing values: %s", metadata.name, non_zero_missing
            )
        non_zero_outliers = {k: v for k, v in report.outlier_report.items() if v > 0}
        if non_zero_outliers:
            self.logger.info(
                "[%s] Columns with IQR-flagged outliers: %s",
                metadata.name, non_zero_outliers,
            )
        if report.constant_columns:
            self.logger.warning(
                "[%s] Constant (zero-variance) columns detected: %s",
                metadata.name, report.constant_columns,
            )
        if report.high_cardinality_columns:
            self.logger.warning(
                "[%s] High-cardinality columns detected (possible leaked "
                "identifiers / free text): %s",
                metadata.name, report.high_cardinality_columns,
            )
        if report.duplicate_raw_column_names:
            self.logger.warning(
                "[%s] Raw CSV header contained duplicate column names "
                "(pandas auto-suffixed them on read): %s",
                metadata.name, report.duplicate_raw_column_names,
            )
        for w in report.target_distribution_warnings:
            self.logger.warning("[%s] Target distribution: %s", metadata.name, w)
        if report.target_whitespace_anomalies:
            self.logger.warning(
                "[%s] %d target label value(s) have leading/trailing "
                "whitespace (reported only, not modified by this loader).",
                metadata.name, report.target_whitespace_anomalies,
            )

    # -- all datasets -----------------------------------------------------

    def load_all(self) -> CKDDataBundle:
        """
        Load every dataset configured in config/datasets.yaml and partition
        them into train_candidate_datasets vs. external_validation_dataset
        based on each dataset's configured `role`.

        The UAE cohort (role: external_validation) is structurally placed
        in a separate dictionary and is never concatenated with anything
        in this method.
        """
        train_candidates: Dict[str, pd.DataFrame] = {}
        external_validation: Dict[str, pd.DataFrame] = {}
        metadata_map: Dict[str, DatasetMetadata] = {}
        quality_map: Dict[str, DataQualityReport] = {}

        for key in self.config.datasets.keys():
            df, metadata, report = self.load_dataset(key)
            metadata_map[key] = metadata
            quality_map[key] = report

            if metadata.role == "external_validation":
                external_validation[key] = df
                self.logger.info(
                    "[%s] Routed to EXTERNAL VALIDATION set (isolated from training).",
                    key,
                )
            else:
                train_candidates[key] = df
                self.logger.info("[%s] Routed to TRAIN-CANDIDATE set.", key)

        if not external_validation:
            self.logger.warning(
                "No dataset was routed to external_validation_dataset. "
                "Confirm the UAE dataset's 'role' in config/datasets.yaml "
                "is set to 'external_validation'."
            )

        bundle = CKDDataBundle(
            train_candidate_datasets=train_candidates,
            external_validation_dataset=external_validation,
            metadata=metadata_map,
            quality_reports=quality_map,
        )

        self.logger.info(
            "Load complete. train_candidate_datasets=%s | external_validation_dataset=%s",
            list(train_candidates.keys()), list(external_validation.keys()),
        )

        return bundle