"""
preprocess.py
=============

Production-grade preprocessing pipeline for the CKD Prediction and
Explainable AI research project.

Pipeline steps (mirroring the project specification):
    1.  Load data via the verified data_loader.py (never from raw CSV).
    2.  Remove invalid Kaggle rows (``discrete`` label, NaN target).
    3.  Clean UCI target labels (strip whitespace / tabs, verify classes).
    4.  Standardise missing-value sentinels → ``np.nan``.
    5.  Parse interval / threshold strings to numeric midpoints.
    6.  Convert medical measurement columns to ``float64``.
    7.  Standardise categorical values to canonical forms.
    8.  Impute: median for numeric, most-frequent for categorical.
        Fitted only on UCI and Kaggle; UAE is *transformed* only.
    9.  Encode binary features and target labels.
    10. Validate the processed datasets.
    11. Save fitted preprocessing artifacts to ``artifacts/preprocessing/``.
    12. Save processed datasets to ``data/processed/``.
    13. Write a ``preprocessing_summary.json`` report.

Constraints (enforced, not just documented):
    ✘ Never trains a model.
    ✘ Never splits data.
    ✘ Never applies SMOTE or synthetic sampling.
    ✘ Never fits imputers / scalers on UAE data.
    ✘ Never selects or reduces features.

Usage
-----
    python preprocess.py
    # or
    from preprocess import CKDPreprocessor
    preprocessor = CKDPreprocessor()
    uci_df, kaggle_df, uae_df = preprocessor.run()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

# Local utilities (pure, stateless helpers).
from preprocessing_utils import (
    apply_interval_parsing,
    clean_target_labels,
    convert_columns_to_numeric,
    encode_binary_features,
    encode_target_labels,
    standardise_categorical_values,
    standardise_missing_values,
    validate_no_nan_targets,
    validate_no_unexpected_categories,
    validate_numeric_dtypes,
)

# ---------------------------------------------------------------------------
# Logger bootstrap
# ---------------------------------------------------------------------------

def _build_logger(log_dir: str, log_filename: str,
                   console_level: str = "INFO",
                   file_level: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger("ckd_preprocessor")
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PreprocessingConfig:
    """
    Loads and exposes ``config/preprocessing.yaml``.

    All clinical constants, column lists, and label maps live in that file;
    nothing is hardcoded in this class or in ``CKDPreprocessor``.
    """

    def __init__(self, config_path: str = "config/preprocessing.yaml") -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing config not found: {path.resolve()}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh)

        self.missing_sentinels: List[str] = raw["missing_value_sentinels"]
        self.kaggle_invalid_labels: List[str] = raw["kaggle_invalid_label_values"]
        self.uci_valid_classes: List[str] = raw["uci_valid_classes"]
        self.numeric_columns: Dict[str, List[str]] = raw["numeric_columns"]
        self.categorical_value_map: Dict[str, str] = raw["categorical_value_map"]
        self.imputation: Dict[str, str] = raw["imputation"]
        self.binary_feature_encoding: Dict[str, int] = raw["binary_feature_encoding"]
        self.target_encoding: Dict[str, Dict[Any, int]] = raw["target_encoding"]
        self.kaggle_valid_stages: List[str] = raw["kaggle_valid_stages"]
        self.paths: Dict[str, str] = raw["paths"]
        self.logging_cfg: Dict[str, str] = raw.get("logging", {})


# ---------------------------------------------------------------------------
# Preprocessing report accumulator
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingReport:
    """Accumulates statistics across all pipeline steps for the JSON report."""

    datasets: Dict[str, Any] = field(default_factory=dict)

    def init_dataset(self, name: str, df_before: pd.DataFrame, target_col: str) -> None:
        self.datasets[name] = {
            "rows_before": len(df_before),
            "rows_after": None,
            "rows_removed": None,
            "rows_removed_reasons": {},
            "missing_values_before": int(df_before.isna().sum().sum()),
            "missing_values_after": None,
            "class_distribution_before": (
                df_before[target_col].value_counts(dropna=False).to_dict()
                if target_col in df_before.columns else {}
            ),
            "class_distribution_after": None,
            "interval_conversions": {},
            "encoding_statistics": {},
            "numeric_conversion_log": {},
            "categorical_remap_counts": {},
            "validation_issues": [],
        }

    def finalise_dataset(self, name: str, df_after: pd.DataFrame, target_col: str) -> None:
        d = self.datasets[name]
        d["rows_after"] = len(df_after)
        d["rows_removed"] = d["rows_before"] - d["rows_after"]
        d["missing_values_after"] = int(df_after.isna().sum().sum())
        d["class_distribution_after"] = (
            df_after[target_col].value_counts(dropna=False).to_dict()
            if target_col in df_after.columns else {}
        )

    def log_rows_removed(self, name: str, reason: str, n: int) -> None:
        self.datasets[name]["rows_removed_reasons"][reason] = n

    def log_interval_conversions(self, name: str, counts: Dict[str, int]) -> None:
        self.datasets[name]["interval_conversions"].update(counts)

    def log_encoding(self, name: str, col: str, count: int) -> None:
        self.datasets[name]["encoding_statistics"][col] = count

    def log_numeric_conversion(self, name: str, log: Dict[str, Any]) -> None:
        self.datasets[name]["numeric_conversion_log"].update(log)

    def log_categorical_remap(self, name: str, counts: Dict[str, int]) -> None:
        self.datasets[name]["categorical_remap_counts"].update(counts)

    def log_validation_issue(self, name: str, issue: str) -> None:
        self.datasets[name]["validation_issues"].append(issue)

    def as_dict(self) -> Dict[str, Any]:
        return {"datasets": self.datasets}


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

class CKDPreprocessor:
    """
    Orchestrates the 13-step CKD preprocessing pipeline.

    Parameters
    ----------
    config_path:
        Path to ``config/preprocessing.yaml``.
    datasets_config_path:
        Path to ``config/datasets.yaml`` (passed through to
        ``CKDDataLoader``).

    Attributes (set after ``run()``)
    ---------------------------------
    numeric_imputers:   ``{"uci": SimpleImputer, "kaggle": SimpleImputer}``
    categorical_imputers: same structure
    encoders:           ``{"binary_feature": dict, "target": dict}``
    label_mappings:     raw JSON-serialisable version of all mappings
    """

    # Target column names (must match datasets.yaml / data_loader.py output)
    _UCI_TARGET = "ckd_label"
    _KAGGLE_TARGET = "ckd_stage_label"
    _UAE_TARGET = "ckd_label"

    # Provenance columns added by data_loader.py (never encoded / imputed)
    _PROVENANCE_COLS = {"source_dataset", "original_row_id"}

    def __init__(
        self,
        config_path: str = "config/preprocessing.yaml",
        datasets_config_path: str = "config/datasets.yaml",
    ) -> None:
        self.cfg = PreprocessingConfig(config_path)
        self.datasets_config_path = datasets_config_path

        log_cfg = self.cfg.logging_cfg
        self.logger = _build_logger(
            log_dir=log_cfg.get("log_dir", "logs"),
            log_filename=log_cfg.get("log_filename", "preprocess.log"),
            console_level=log_cfg.get("console_level", "INFO"),
            file_level=log_cfg.get("file_level", "DEBUG"),
        )

        self.report = PreprocessingReport()

        # Stateful sklearn objects – populated during run()
        self.numeric_imputers: Dict[str, SimpleImputer] = {}
        self.categorical_imputers: Dict[str, SimpleImputer] = {}
        self.encoders: Dict[str, Any] = {}
        self.label_mappings: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline end-to-end.

        Returns
        -------
        (uci_processed, kaggle_processed, uae_processed)
            ML-ready DataFrames.  All artifacts and processed CSVs have been
            saved to disk before returning.
        """
        self.logger.info("=" * 70)
        self.logger.info("CKD Preprocessing Pipeline – START")
        self.logger.info("=" * 70)

        # Step 1 ── Load data via data_loader.py
        uci_df, kaggle_df, uae_df = self._step1_load_data()

        # Initialise report accumulators
        self.report.init_dataset("uci", uci_df, self._UCI_TARGET)
        self.report.init_dataset("kaggle", kaggle_df, self._KAGGLE_TARGET)
        self.report.init_dataset("uae", uae_df, self._UAE_TARGET)

        # Step 2 ── Remove invalid Kaggle rows
        kaggle_df = self._step2_remove_invalid_kaggle_rows(kaggle_df)

        # Step 3 ── Clean UCI target labels
        uci_df = self._step3_clean_uci_targets(uci_df)

        # Step 4 ── Standardise missing values
        uci_df = self._step4_standardise_missing(uci_df, "UCI")
        kaggle_df = self._step4_standardise_missing(kaggle_df, "Kaggle")
        uae_df = self._step4_standardise_missing(uae_df, "UAE")

        # Step 5 ── Parse intervals / threshold strings
        uci_df, kaggle_df = self._step5_interval_parsing(uci_df, kaggle_df)

        # Step 6 ── Type conversion
        uci_df = self._step6_type_conversion(uci_df, "uci")
        kaggle_df = self._step6_type_conversion(kaggle_df, "kaggle")
        uae_df = self._step6_type_conversion(uae_df, "uae")

        # Step 6b ── UAE unit harmonisation (creatinine µmol/L → mg/dL)
        # The UAE CSV stores serum_creatinine in µmol/L (values ~35–123), while
        # UCI and Kaggle store it in mg/dL (values ~0.4–76).  Without this
        # conversion the model sees UAE creatinine ~75 vs the UCI training
        # distribution centred on ~1.3, mapping every UAE patient to the extreme
        # CKD tail and causing 100% CKD predictions — the root cause of
        # TN=0, Accuracy=0.114.
        # Conversion factor: 1 mg/dL = 88.4 µmol/L (standard clinical constant).
        uae_df = self._step6b_uae_unit_harmonisation(uae_df)

        # Step 7 ── Categorical standardisation
        uci_df = self._step7_categorical_standardisation(uci_df, "uci")
        kaggle_df = self._step7_categorical_standardisation(kaggle_df, "kaggle")
        uae_df = self._step7_categorical_standardisation(uae_df, "uae")

        # Step 8 ── Imputation (fit on UCI + Kaggle; transform-only on UAE)
        uci_df, kaggle_df, uae_df = self._step8_imputation(uci_df, kaggle_df, uae_df)

        # Step 9 ── Encoding
        uci_df, kaggle_df, uae_df = self._step9_encoding(uci_df, kaggle_df, uae_df)

        # Step 10 ── Validation
        self._step10_validation(uci_df, kaggle_df, uae_df)

        # Step 11 ── Save artifacts
        self._step11_save_artifacts()

        # Step 12 ── Save processed datasets
        self._step12_save_datasets(uci_df, kaggle_df, uae_df)

        # Step 13 ── Write report
        self.report.finalise_dataset("uci", uci_df, self._UCI_TARGET)
        self.report.finalise_dataset("kaggle", kaggle_df, self._KAGGLE_TARGET)
        self.report.finalise_dataset("uae", uae_df, self._UAE_TARGET)
        self._step13_write_report()

        self.logger.info("=" * 70)
        self.logger.info("CKD Preprocessing Pipeline – COMPLETE")
        self.logger.info("=" * 70)

        return uci_df, kaggle_df, uae_df

    # -----------------------------------------------------------------------
    # Step 1 – Load data
    # -----------------------------------------------------------------------

    def _step1_load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load the three datasets via the existing CKDDataLoader."""
        self.logger.info("[Step 1] Loading datasets via data_loader.py …")

        # Import here to keep the module-level import clean and to ensure
        # any import errors are surfaced at step-1 time, not module-load time.
        from ckd_data.data_loader import CKDDataLoader  # type: ignore[import]

        loader = CKDDataLoader(config_path=self.datasets_config_path)
        bundle = loader.load_all()

        uci_df: pd.DataFrame = bundle.train_candidate_datasets["uci"].copy()
        kaggle_df: pd.DataFrame = bundle.train_candidate_datasets["kaggle"].copy()
        uae_df: pd.DataFrame = bundle.external_validation_dataset["uae"].copy()

        self.logger.info(
            "[Step 1] Loaded – UCI: %s | Kaggle: %s | UAE: %s",
            uci_df.shape, kaggle_df.shape, uae_df.shape,
        )
        return uci_df, kaggle_df, uae_df

    # -----------------------------------------------------------------------
    # Step 2 – Remove invalid Kaggle rows
    # -----------------------------------------------------------------------

    def _step2_remove_invalid_kaggle_rows(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove rows where ``ckd_stage_label`` is in the configured
        ``kaggle_invalid_label_values`` list, or where the target is NaN.
        """
        self.logger.info("[Step 2] Removing invalid Kaggle rows …")
        target = self._KAGGLE_TARGET
        rows_before = len(df)

        # Clean string targets first so "discrete " matches "discrete".
        if target in df.columns and df[target].dtype == object:
            df[target] = df[target].astype(str).str.strip().str.lower()
            # Restore NaN that astype(str) would turn into "nan".
            df.loc[df[target] == "nan", target] = np.nan

        # Mask: invalid label strings
        invalid_label_mask = pd.Series(False, index=df.index)
        if target in df.columns:
            invalid_label_mask = df[target].isin(
                [v.lower() for v in self.cfg.kaggle_invalid_labels]
            )

        n_invalid_label = int(invalid_label_mask.sum())
        pct_invalid = round(n_invalid_label / max(rows_before, 1) * 100, 2)
        self.logger.info(
            "[Step 2] Rows with invalid label ('discrete', etc.): %d (%.2f%%)",
            n_invalid_label, pct_invalid,
        )
        self.report.log_rows_removed(
            "kaggle", "invalid_label_values", n_invalid_label
        )

        # Mask: NaN target
        nan_target_mask = df[target].isna() if target in df.columns else pd.Series(False, index=df.index)
        n_nan = int(nan_target_mask.sum())
        pct_nan = round(n_nan / max(rows_before, 1) * 100, 2)
        self.logger.info(
            "[Step 2] Rows with NaN target: %d (%.2f%%)", n_nan, pct_nan
        )
        self.report.log_rows_removed("kaggle", "nan_target", n_nan)

        combined_mask = invalid_label_mask | nan_target_mask
        df = df[~combined_mask].reset_index(drop=True)

        rows_after = len(df)
        total_removed = rows_before - rows_after
        self.logger.info(
            "[Step 2] Kaggle rows: %d → %d  (removed %d total)",
            rows_before, rows_after, total_removed,
        )
        return df

    # -----------------------------------------------------------------------
    # Step 3 – UCI target label cleaning
    # -----------------------------------------------------------------------

    def _step3_clean_uci_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace and tab characters from the UCI
        target column.  Raises if the resulting classes are not exactly
        {ckd, notckd}.
        """
        self.logger.info("[Step 3] Cleaning UCI target labels …")
        target = self._UCI_TARGET
        if target not in df.columns:
            raise RuntimeError(
                f"[Step 3] UCI target column '{target}' is missing."
            )

        cleaned, n_changed = clean_target_labels(
            df[target], valid_classes=self.cfg.uci_valid_classes
        )
        df = df.copy()
        df[target] = cleaned
        self.logger.info(
            "[Step 3] UCI target: %d label(s) cleaned (whitespace/tab stripped).",
            n_changed,
        )
        self.logger.info(
            "[Step 3] UCI target value counts: %s",
            df[target].value_counts(dropna=False).to_dict(),
        )
        return df

    # -----------------------------------------------------------------------
    # Step 4 – Standardise missing values
    # -----------------------------------------------------------------------

    def _step4_standardise_missing(
        self, df: pd.DataFrame, dataset_label: str
    ) -> pd.DataFrame:
        """Replace all sentinel strings with ``np.nan``."""
        self.logger.info(
            "[Step 4] Standardising missing values for %s …", dataset_label
        )
        df, n_replaced = standardise_missing_values(df, self.cfg.missing_sentinels)
        self.logger.info(
            "[Step 4] %s: %d cell(s) replaced with NaN.", dataset_label, n_replaced
        )
        return df

    # -----------------------------------------------------------------------
    # Step 5 – Interval / threshold parsing
    # -----------------------------------------------------------------------

    def _step5_interval_parsing(
        self,
        uci_df: pd.DataFrame,
        kaggle_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse interval strings (e.g. ``"138 - 143"``) to midpoints, and
        threshold strings (e.g. ``"< 48.1"``) to their numeric values.

        Applied only to configured numeric columns on UCI and Kaggle.
        UAE does not have interval-value issues according to the spec.
        """
        self.logger.info("[Step 5] Parsing interval/threshold strings …")

        uci_num_cols = self.cfg.numeric_columns.get("uci", [])
        uci_df, uci_counts = apply_interval_parsing(uci_df, uci_num_cols)
        self.report.log_interval_conversions("uci", uci_counts)
        self.logger.info("[Step 5] UCI interval conversions: %s", uci_counts)

        kaggle_num_cols = self.cfg.numeric_columns.get("kaggle", [])
        kaggle_df, kaggle_counts = apply_interval_parsing(kaggle_df, kaggle_num_cols)
        self.report.log_interval_conversions("kaggle", kaggle_counts)
        self.logger.info("[Step 5] Kaggle interval conversions: %s", kaggle_counts)

        return uci_df, kaggle_df

    # -----------------------------------------------------------------------
    # Step 6 – Type conversion
    # -----------------------------------------------------------------------

    def _step6_type_conversion(
        self, df: pd.DataFrame, dataset_key: str
    ) -> pd.DataFrame:
        """Cast all configured numeric columns to ``float64``."""
        self.logger.info("[Step 6] Type conversion for '%s' …", dataset_key)
        cols = self.cfg.numeric_columns.get(dataset_key, [])
        df, log = convert_columns_to_numeric(
            df, cols, log_prefix=f"[Step 6][{dataset_key.upper()}]"
        )
        self.report.log_numeric_conversion(dataset_key, log)
        self.logger.info(
            "[Step 6] %s: converted %d column(s) to float64.", dataset_key, len(log)
        )
        return df

    # -----------------------------------------------------------------------
    # Step 7 – Categorical standardisation
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Step 6b – UAE-specific unit harmonisation
    # -----------------------------------------------------------------------

    def _step6b_uae_unit_harmonisation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert UAE ``serum_creatinine`` from µmol/L to mg/dL so it is on
        the same scale as UCI/Kaggle before feature engineering and validation.

        Evidence for µmol/L storage:
        * UAE CSV values range 35–123, consistent with the µmol/L normal
          range (53–106) for a general outpatient cohort.
        * UCI/Kaggle values range 0.4–76 mg/dL.
        * Without conversion, age_creatinine_interaction reaches ~4 000 in UAE
          vs ~68 in UCI, placing every UAE patient at the extreme CKD tail
          of the UCI-trained model — the direct cause of TN=0.

        Conversion: mg/dL = µmol/L / 88.4  (standard clinical constant,
        identical to the CKD-EPI and MDRD formula denominator).

        This method also recomputes ``kidney_dysfunction_score``
        (serum_creatinine / egfr) if present, since it was already computed
        in feature_engineering.py on the raw (unconverted) value and stored
        in the engineered CSV.  The recomputation uses the corrected
        serum_creatinine and the original egfr.

        No leakage is introduced: the conversion factor 88.4 is a universal
        clinical constant, not derived from any dataset.
        """
        df = df.copy()
        SC_COL = "serum_creatinine"
        UMOL_TO_MGDL = 88.4

        if SC_COL not in df.columns:
            self.logger.warning(
                "[Step 6b] '%s' column not found in UAE — unit harmonisation skipped.",
                SC_COL,
            )
            return df

        # Confirm this is likely µmol/L before converting (sanity guard).
        median_val = df[SC_COL].median()
        if median_val < 10:
            self.logger.info(
                "[Step 6b] UAE serum_creatinine median=%.3f — already in mg/dL range, "
                "skipping conversion.",
                median_val,
            )
            return df

        before_median = float(df[SC_COL].median())
        df[SC_COL] = df[SC_COL] / UMOL_TO_MGDL
        after_median = float(df[SC_COL].median())

        self.logger.info(
            "[Step 6b] UAE serum_creatinine converted µmol/L → mg/dL "
            "(÷ %.1f): median %.1f µmol/L → %.3f mg/dL.",
            UMOL_TO_MGDL, before_median, after_median,
        )

        # Recompute kidney_dysfunction_score if it was pre-computed and stored.
        KDS_COL = "kidney_dysfunction_score"
        EGFR_COL = "egfr"
        if KDS_COL in df.columns and EGFR_COL in df.columns:
            denom = df[EGFR_COL].replace(0, float("nan"))
            df[KDS_COL] = df[SC_COL] / denom
            self.logger.info(
                "[Step 6b] Recomputed '%s' using corrected serum_creatinine.",
                KDS_COL,
            )

        # Recompute age_creatinine_interaction if pre-computed.
        ACI_COL = "age_creatinine_interaction"
        AGE_COL = "age"
        if ACI_COL in df.columns and AGE_COL in df.columns:
            df[ACI_COL] = df[AGE_COL] * df[SC_COL]
            self.logger.info(
                "[Step 6b] Recomputed '%s' using corrected serum_creatinine.",
                ACI_COL,
            )

        # Recompute urea_creatinine_product if pre-computed.
        UCP_COL = "urea_creatinine_product"
        BU_COL = "blood_urea"
        if UCP_COL in df.columns and BU_COL in df.columns:
            df[UCP_COL] = df[BU_COL] * df[SC_COL]
            self.logger.info(
                "[Step 6b] Recomputed '%s' using corrected serum_creatinine.",
                UCP_COL,
            )

        # Recompute hemoglobin_creatinine_ratio if pre-computed.
        HCR_COL = "hemoglobin_creatinine_ratio"
        HB_COL = "hemoglobin"
        if HCR_COL in df.columns and HB_COL in df.columns:
            denom = df[SC_COL].replace(0, float("nan"))
            df[HCR_COL] = df[HB_COL] / denom
            self.logger.info(
                "[Step 6b] Recomputed '%s' using corrected serum_creatinine.",
                HCR_COL,
            )

        # Recompute bun_creatinine_ratio if pre-computed.
        BCR_COL = "bun_creatinine_ratio"
        if BCR_COL in df.columns and BU_COL in df.columns:
            denom = df[SC_COL].replace(0, float("nan"))
            df[BCR_COL] = df[BU_COL] / denom
            self.logger.info(
                "[Step 6b] Recomputed '%s' using corrected serum_creatinine.",
                BCR_COL,
            )

        return df

    def _step7_categorical_standardisation(
        self, df: pd.DataFrame, dataset_key: str
    ) -> pd.DataFrame:
        """
        Map raw categorical strings to canonical values (e.g. ``"Yes"`` →
        ``"yes"``, ``"Not Present"`` → ``"notpresent"``).
        """
        self.logger.info(
            "[Step 7] Categorical standardisation for '%s' …", dataset_key
        )
        numeric_cols = set(self.cfg.numeric_columns.get(dataset_key, []))
        target_cols = {self._UCI_TARGET, self._KAGGLE_TARGET, self._UAE_TARGET}
        provenance_cols = self._PROVENANCE_COLS

        # Only apply to object columns that are not numeric or target or provenance.
        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols
            and c not in target_cols
            and c not in provenance_cols
            and df[c].dtype == object
        ]

        df, remap_counts = standardise_categorical_values(
            df, categorical_cols, self.cfg.categorical_value_map
        )
        self.report.log_categorical_remap(dataset_key, remap_counts)
        total_remapped = sum(remap_counts.values())
        self.logger.info(
            "[Step 7] %s: %d total cell(s) remapped across %d column(s).",
            dataset_key, total_remapped, len([c for c, n in remap_counts.items() if n > 0]),
        )
        return df

    # -----------------------------------------------------------------------
    # Step 8 – Imputation
    # -----------------------------------------------------------------------

    def _step8_imputation(
        self,
        uci_df: pd.DataFrame,
        kaggle_df: pd.DataFrame,
        uae_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit SimpleImputers on UCI and Kaggle independently.
        UAE is transformed using the **UCI** imputer for matching columns
        (the spec says "never fit on UAE"; using UCI-fitted objects for the
        external validation cohort is the standard research approach for a
        dataset with matching feature names).

        Imputation strategy (from config):
            * Numeric columns → median
            * Categorical columns → most_frequent
        """
        self.logger.info("[Step 8] Imputation …")

        numeric_strategy = self.cfg.imputation["numeric_strategy"]
        cat_strategy = self.cfg.imputation["categorical_strategy"]

        uci_df = self._impute_dataset(
            uci_df, "uci", self._UCI_TARGET, numeric_strategy, cat_strategy, fit=True
        )
        kaggle_df = self._impute_dataset(
            kaggle_df, "kaggle", self._KAGGLE_TARGET, numeric_strategy, cat_strategy, fit=True
        )

        # UAE: transform-only using the UCI-fitted imputers.
        self.logger.info(
            "[Step 8] Transforming UAE using UCI-fitted imputers (no fitting on UAE)."
        )
        uae_df = self._impute_dataset(
            uae_df, "uae", self._UAE_TARGET,
            numeric_strategy, cat_strategy, fit=False,
            external_numeric_imputer=self.numeric_imputers.get("uci"),
            external_cat_imputer=self.categorical_imputers.get("uci"),
        )

        return uci_df, kaggle_df, uae_df

    def _impute_dataset(
        self,
        df: pd.DataFrame,
        dataset_key: str,
        target_col: str,
        numeric_strategy: str,
        cat_strategy: str,
        fit: bool,
        external_numeric_imputer: Optional[SimpleImputer] = None,
        external_cat_imputer: Optional[SimpleImputer] = None,
    ) -> pd.DataFrame:
        """
        Internal helper.  When *fit=True* the imputers are fitted and stored
        in ``self.numeric_imputers[dataset_key]`` and
        ``self.categorical_imputers[dataset_key]``.

        When *fit=False* the provided *external_* imputers are used and their
        columns are intersected with what actually exists in *df* to avoid
        KeyErrors on column-count mismatches between datasets.
        """
        df = df.copy()
        numeric_cols_cfg = set(self.cfg.numeric_columns.get(dataset_key, []))
        skip = self._PROVENANCE_COLS | {target_col}

        def _is_categorical(series: pd.Series) -> bool:
            """True for object, StringDtype, and CategoricalDtype columns."""
            return (
                series.dtype == object
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
            )

        # Separate columns by type.
        numeric_cols = [
            c for c in df.columns
            if c in numeric_cols_cfg and c not in skip
        ]
        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols_cfg and c not in skip
            and _is_categorical(df[c])
        ]

        self.logger.debug(
            "[Step 8][%s] Numeric cols to impute: %d | Categorical: %d",
            dataset_key, len(numeric_cols), len(categorical_cols),
        )

        # Numeric imputation.
        if numeric_cols:
            if fit:
                num_imp = SimpleImputer(strategy=numeric_strategy)
                df[numeric_cols] = num_imp.fit_transform(df[numeric_cols])
                self.numeric_imputers[dataset_key] = num_imp
                self.logger.info(
                    "[Step 8][%s] Fitted numeric imputer (%s) on %d column(s).",
                    dataset_key, numeric_strategy, len(numeric_cols),
                )
            else:
                if external_numeric_imputer is not None:
                    # Only apply the external imputer column-by-column to avoid
                    # sklearn's strict feature-name validation when UAE has a
                    # different column set than UCI.
                    trained_features = list(external_numeric_imputer.feature_names_in_)
                    trained_stats = dict(zip(
                        trained_features,
                        external_numeric_imputer.statistics_,
                    ))
                    overlap = [c for c in numeric_cols if c in trained_stats]
                    n_filled = 0
                    for col in overlap:
                        median_val = trained_stats[col]
                        df[col] = df[col].fillna(median_val)
                        n_filled += 1
                    if n_filled:
                        self.logger.info(
                            "[Step 8][%s] Applied external numeric imputer stats to %d column(s).",
                            dataset_key, n_filled,
                        )
                    # Columns in numeric_cols not covered by the external imputer
                    # get a simple median-of-self fill as a safe fallback.
                    uncovered = [c for c in numeric_cols if c not in trained_stats]
                    if uncovered:
                        fallback = SimpleImputer(strategy="median")
                        df[uncovered] = fallback.fit_transform(df[uncovered])
                        self.logger.info(
                            "[Step 8][%s] Fallback self-fitted median imputer on %d uncovered column(s): %s",
                            dataset_key, len(uncovered), uncovered,
                        )
                else:
                    # No external imputer provided – fit a fresh one as fallback.
                    fallback = SimpleImputer(strategy="median")
                    df[numeric_cols] = fallback.fit_transform(df[numeric_cols])

        # Categorical imputation.
        if categorical_cols:
            # SimpleImputer raises "boolean value of NA is ambiguous" on
            # StringDtype columns (Python 3.12+). Cast to object first so
            # NaN is a plain float and comparison inside sklearn works.
            df[categorical_cols] = df[categorical_cols].astype(object)
            if fit:
                cat_imp = SimpleImputer(strategy=cat_strategy)
                df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
                self.categorical_imputers[dataset_key] = cat_imp
                self.logger.info(
                    "[Step 8][%s] Fitted categorical imputer (%s) on %d column(s).",
                    dataset_key, cat_strategy, len(categorical_cols),
                )
            else:
                if external_cat_imputer is not None:
                    trained_features = list(external_cat_imputer.feature_names_in_)
                    trained_stats = dict(zip(
                        trained_features,
                        external_cat_imputer.statistics_,
                    ))
                    overlap = [c for c in categorical_cols if c in trained_stats]
                    for col in overlap:
                        df[col] = df[col].fillna(trained_stats[col])
                    if overlap:
                        self.logger.info(
                            "[Step 8][%s] Applied external categorical imputer stats to %d column(s).",
                            dataset_key, len(overlap),
                        )
                    uncovered = [c for c in categorical_cols if c not in trained_stats]
                    if uncovered:
                        fallback = SimpleImputer(strategy="most_frequent")
                        df[uncovered] = fallback.fit_transform(df[uncovered])
                        self.logger.info(
                            "[Step 8][%s] Fallback self-fitted categorical imputer on %d uncovered column(s): %s",
                            dataset_key, len(uncovered), uncovered,
                        )
                else:
                    fallback = SimpleImputer(strategy="most_frequent")
                    df[categorical_cols] = fallback.fit_transform(df[categorical_cols])

        return df

    # -----------------------------------------------------------------------
    # Step 9 – Encoding
    # -----------------------------------------------------------------------

    def _step9_encoding(
        self,
        uci_df: pd.DataFrame,
        kaggle_df: pd.DataFrame,
        uae_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode binary feature columns and target labels.

        Binary features: ``yes/no``, ``present/notpresent``, etc. → 0/1
        Targets:
            UCI       ``ckd`` → 1, ``notckd`` → 0
            Kaggle    ``s1`` → 0 … ``s5`` → 4
            UAE       ``0``/``1`` preserved as int
        """
        self.logger.info("[Step 9] Encoding features and targets …")

        binary_map = self.cfg.binary_feature_encoding
        self.encoders = {
            "binary_feature_encoding": deepcopy(binary_map),
            "target_encoding": deepcopy(self.cfg.target_encoding),
        }
        self.label_mappings = {
            "binary_feature_encoding": binary_map,
            "target_encoding": {
                k: {str(kk): vv for kk, vv in v.items()}
                for k, v in self.cfg.target_encoding.items()
            },
        }

        skip = self._PROVENANCE_COLS
        target_cols = {self._UCI_TARGET, self._KAGGLE_TARGET, self._UAE_TARGET}

        def _binary_feature_cols(df: pd.DataFrame, dataset_key: str) -> List[str]:
            numeric_set = set(self.cfg.numeric_columns.get(dataset_key, []))
            return [
                c for c in df.columns
                if c not in skip and c not in target_cols and c not in numeric_set
            ]

        # --- UCI ---
        uci_feat_cols = _binary_feature_cols(uci_df, "uci")
        uci_df, uci_enc_counts = encode_binary_features(uci_df, uci_feat_cols, binary_map)
        uci_df[self._UCI_TARGET] = encode_target_labels(
            uci_df[self._UCI_TARGET], self.cfg.target_encoding["uci"]
        )
        for col, cnt in uci_enc_counts.items():
            self.report.log_encoding("uci", col, cnt)
        self.logger.info("[Step 9] UCI: encoded %d feature column(s).", len(uci_feat_cols))

        # --- Kaggle ---
        kaggle_feat_cols = _binary_feature_cols(kaggle_df, "kaggle")
        kaggle_df, kaggle_enc_counts = encode_binary_features(
            kaggle_df, kaggle_feat_cols, binary_map
        )
        kaggle_df[self._KAGGLE_TARGET] = encode_target_labels(
            kaggle_df[self._KAGGLE_TARGET], self.cfg.target_encoding["kaggle"]
        )
        for col, cnt in kaggle_enc_counts.items():
            self.report.log_encoding("kaggle", col, cnt)
        self.logger.info(
            "[Step 9] Kaggle: encoded %d feature column(s).", len(kaggle_feat_cols)
        )

        # --- UAE ---
        uae_feat_cols = _binary_feature_cols(uae_df, "uae")
        uae_df, uae_enc_counts = encode_binary_features(uae_df, uae_feat_cols, binary_map)
        uae_df[self._UAE_TARGET] = encode_target_labels(
            uae_df[self._UAE_TARGET], self.cfg.target_encoding["uae"]
        )
        for col, cnt in uae_enc_counts.items():
            self.report.log_encoding("uae", col, cnt)
        self.logger.info("[Step 9] UAE: encoded %d feature column(s).", len(uae_feat_cols))

        return uci_df, kaggle_df, uae_df

    # -----------------------------------------------------------------------
    # Step 10 – Validation
    # -----------------------------------------------------------------------

    def _step10_validation(
        self,
        uci_df: pd.DataFrame,
        kaggle_df: pd.DataFrame,
        uae_df: pd.DataFrame,
    ) -> None:
        """
        Run post-processing checks.  Warnings are logged; hard failures raise
        ``RuntimeError`` so the pipeline does not silently produce bad data.
        """
        self.logger.info("[Step 10] Validating processed datasets …")
        all_passed = True

        for df, name, target, key in [
            (uci_df, "UCI", self._UCI_TARGET, "uci"),
            (kaggle_df, "Kaggle", self._KAGGLE_TARGET, "kaggle"),
            (uae_df, "UAE", self._UAE_TARGET, "uae"),
        ]:
            # a) No NaN targets
            try:
                validate_no_nan_targets(df[target], name)
                self.logger.info("[Step 10][%s] ✔ No NaN targets.", name)
            except ValueError as exc:
                msg = str(exc)
                self.logger.error("[Step 10][%s] ✘ %s", name, msg)
                self.report.log_validation_issue(key, msg)
                all_passed = False

            # b) No 'discrete' rows remain in Kaggle
            if name == "Kaggle":
                n_disc = int((df[target].astype(str).str.lower() == "discrete").sum())
                if n_disc:
                    msg = f"{n_disc} 'discrete' row(s) still present in Kaggle target."
                    self.logger.error("[Step 10][%s] ✘ %s", name, msg)
                    self.report.log_validation_issue(key, msg)
                    all_passed = False
                else:
                    self.logger.info("[Step 10][%s] ✔ No 'discrete' rows.", name)

            # c) Numeric dtype check
            expected_numeric = self.cfg.numeric_columns.get(key, [])
            non_numeric = validate_numeric_dtypes(df, expected_numeric, name)
            if non_numeric:
                msg = f"Non-numeric dtype in expected numeric columns: {non_numeric}"
                self.report.log_validation_issue(key, msg)

            # d) No unexpected encoded values in binary columns
            # (just log; not a hard failure since some cols may have
            # legitimate numeric-integer values after encoding)
            expected_encoded_values = {
                c: [0, 1] for c in df.columns
                if c not in self._PROVENANCE_COLS
                and c not in {self._UCI_TARGET, self._KAGGLE_TARGET, self._UAE_TARGET}
                and c not in set(self.cfg.numeric_columns.get(key, []))
            }
            unexpected = validate_no_unexpected_categories(
                df, list(expected_encoded_values.keys()),
                expected_encoded_values, name,
            )
            if unexpected:
                msg = f"Unexpected encoded values in: {unexpected}"
                self.report.log_validation_issue(key, msg)

            self.logger.info(
                "[Step 10][%s] Shape after preprocessing: %s | NaN count: %d",
                name, df.shape, int(df.isna().sum().sum()),
            )

        if not all_passed:
            raise RuntimeError(
                "[Step 10] Validation failed.  See logs and preprocessing_summary.json "
                "for details.  Fix the pipeline before proceeding to training."
            )
        self.logger.info("[Step 10] All validation checks passed.")

    # -----------------------------------------------------------------------
    # Step 11 – Save artifacts
    # -----------------------------------------------------------------------

    def _step11_save_artifacts(self) -> None:
        """
        Persist fitted preprocessing objects to ``artifacts/preprocessing/``.

        Saved objects
        -------------
        * ``numeric_imputer.joblib``     – dict of fitted SimpleImputers
        * ``categorical_imputer.joblib`` – dict of fitted SimpleImputers
        * ``encoders.joblib``            – encoding maps
        * ``label_mappings.json``        – human-readable JSON copy
        """
        self.logger.info("[Step 11] Saving preprocessing artifacts …")
        artifacts_dir = Path(self.cfg.paths["artifacts_dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.numeric_imputers,
            artifacts_dir / "numeric_imputer.joblib",
        )
        joblib.dump(
            self.categorical_imputers,
            artifacts_dir / "categorical_imputer.joblib",
        )
        joblib.dump(
            self.encoders,
            artifacts_dir / "encoders.joblib",
        )

        label_mappings_path = artifacts_dir / "label_mappings.json"
        with open(label_mappings_path, "w", encoding="utf-8") as fh:
            json.dump(self.label_mappings, fh, indent=2, default=str)

        self.logger.info(
            "[Step 11] Artifacts saved to: %s", artifacts_dir.resolve()
        )

    # -----------------------------------------------------------------------
    # Step 12 – Save processed datasets
    # -----------------------------------------------------------------------

    def _step12_save_datasets(
        self,
        uci_df: pd.DataFrame,
        kaggle_df: pd.DataFrame,
        uae_df: pd.DataFrame,
    ) -> None:
        """Save ML-ready DataFrames as CSV to ``data/processed/``."""
        self.logger.info("[Step 12] Saving processed datasets …")
        processed_dir = Path(self.cfg.paths["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)

        for df, filename in [
            (uci_df, "uci_processed.csv"),
            (kaggle_df, "kaggle_processed.csv"),
            (uae_df, "uae_processed.csv"),
        ]:
            path = processed_dir / filename
            df.to_csv(path, index=False)
            self.logger.info("[Step 12] Saved: %s  (%d rows)", path.resolve(), len(df))

    # -----------------------------------------------------------------------
    # Step 13 – Write report
    # -----------------------------------------------------------------------

    def _step13_write_report(self) -> None:
        """Write ``preprocessing_summary.json``."""
        self.logger.info("[Step 13] Writing preprocessing report …")
        report_path = Path(self.cfg.paths["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = self.report.as_dict()

        # Add top-level metadata.
        report_dict["pipeline_version"] = "1.0.0"
        report_dict["config_file"] = "config/preprocessing.yaml"

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, default=_json_serialise)

        self.logger.info(
            "[Step 13] Report saved to: %s", report_path.resolve()
        )


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_serialise(obj: Any) -> Any:
    """Custom JSON serialiser for types that ``json.dump`` can't handle."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.NA.__class__):
        return None
    return str(obj)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocessor = CKDPreprocessor(
        config_path="config/preprocessing.yaml",
        datasets_config_path="config/datasets.yaml",
    )
    uci_out, kaggle_out, uae_out = preprocessor.run()

    print("\n── Preprocessing complete ──")
    print(f"UCI processed shape    : {uci_out.shape}")
    print(f"Kaggle processed shape : {kaggle_out.shape}")
    print(f"UAE processed shape    : {uae_out.shape}")
    print(
        f"\nArtifacts  → artifacts/preprocessing/\n"
        f"Datasets   → data/processed/\n"
        f"Report     → artifacts/preprocessing/preprocessing_summary.json"
    )