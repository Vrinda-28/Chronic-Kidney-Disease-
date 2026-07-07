"""
preprocessing_utils.py
======================

Shared, stateless utility functions and constants for the CKD preprocessing
pipeline.  All functions here are pure (no side-effects, no global state)
and are independently unit-testable.

Scope:
    - Missing-value standardisation helpers.
    - Interval / threshold string parsing.
    - Categorical string normalisation.
    - Type-conversion helpers with detailed logging.
    - Validation helpers used across preprocessing steps.

Out of scope (lives in preprocess.py):
    - Scikit-learn imputer / encoder objects (stateful, need to be fitted).
    - Dataset-level orchestration.
    - File I/O, artifact saving.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ckd_preprocessor")

# ---------------------------------------------------------------------------
# Sentinel patterns for missing values
# ---------------------------------------------------------------------------

def standardise_missing_values(
    df: pd.DataFrame,
    sentinels: List[str],
) -> Tuple[pd.DataFrame, int]:
    """
    Replace all sentinel strings (e.g. "?", "NA", "NULL", empty / blank
    strings) with ``np.nan`` across the entire DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame (any mix of dtypes).
    sentinels:
        List of string tokens to treat as missing.  Loaded from
        ``config/preprocessing.yaml``.

    Returns
    -------
    (cleaned_df, n_replacements):
        Cleaned copy of *df* and the total number of cells that were
        replaced.

    Notes
    -----
    * Leading / trailing whitespace is stripped from every string cell
      *before* the sentinel check so that ``" ?"`` and ``"?"`` are both
      caught.
    * The function never mutates the caller's DataFrame (returns a copy).
    """
    df = df.copy()
    sentinel_set = set(s.strip() for s in sentinels)
    n_replaced = 0

    for col in df.columns:
        # Handle both legacy object dtype and Python 3.12+ StringDtype.
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            # astype(str) turns existing NaN into "nan" – track them first.
            is_already_null = df[col].isna()
            stripped = df[col].astype(str).str.strip()
            mask = (~is_already_null) & (stripped.isin(sentinel_set) | (stripped == ""))
            n_replaced += int(mask.sum())
            # Cast to object so np.nan can be stored regardless of StringDtype.
            new_col = df[col].astype(object)
            new_col[mask] = np.nan
            df[col] = new_col

    return df, n_replaced


# ---------------------------------------------------------------------------
# Interval / threshold string parser
# ---------------------------------------------------------------------------

# Pre-compiled patterns in evaluation order.
_INTERVAL_RE = re.compile(
    r"^\s*(?P<lo>[0-9]+(?:\.[0-9]+)?)\s*[-–]\s*(?P<hi>[0-9]+(?:\.[0-9]+)?)\s*$"
)
_THRESHOLD_RE = re.compile(
    r"^\s*(?:<=?|>=?|=)\s*(?P<val>[0-9]+(?:\.[0-9]+)?)\s*$"
)
_PLAIN_RE = re.compile(
    r"^\s*(?P<val>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def parse_numeric_string(raw: Any) -> Optional[float]:
    """
    Parse a single value that may be a numeric string, an interval, or a
    threshold expression and return a ``float`` (or ``None`` on failure).

    Supported formats
    -----------------
    * ``"138 - 143"``  → midpoint ``140.5``
    * ``"133 - 138"``  → midpoint ``135.5``
    * ``"1.019 - 1.021"`` → midpoint ``1.020``
    * ``"< 48.1"``    → ``48.1``
    * ``"<= 48.1"``   → ``48.1``
    * ``"> 48.1"``    → ``48.1``
    * ``">= 48.1"``   → ``48.1``
    * ``"= 227.944"`` → ``227.944``
    * ``"200"``        → ``200.0``
    * Already a ``float`` / ``int`` → returned unchanged.
    * ``np.nan`` / ``None`` → ``None``.

    Returns
    -------
    float or None
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)

    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "", "?", "na"}:
        return None

    m = _INTERVAL_RE.match(s)
    if m:
        lo, hi = float(m.group("lo")), float(m.group("hi"))
        return (lo + hi) / 2.0

    m = _THRESHOLD_RE.match(s)
    if m:
        return float(m.group("val"))

    m = _PLAIN_RE.match(s)
    if m:
        return float(m.group("val"))

    return None


def apply_interval_parsing(
    df: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply :func:`parse_numeric_string` to every cell in *columns* and
    return the updated DataFrame plus a per-column count of how many cells
    were parsed from a non-trivially-numeric string (intervals / thresholds).

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Subset of column names to apply parsing to.  Columns absent from
        *df* are silently skipped.

    Returns
    -------
    (updated_df, conversion_counts)
    """
    df = df.copy()
    conversion_counts: Dict[str, int] = {}

    for col in columns:
        if col not in df.columns:
            continue
        n_interval = 0
        parsed_vals: List[Optional[float]] = []
        for raw_val in df[col]:
            parsed = parse_numeric_string(raw_val)
            parsed_vals.append(parsed)
            # Count only those that were non-trivially converted
            # (i.e. the raw value was a string and not just a plain number).
            if (
                parsed is not None
                and isinstance(raw_val, str)
                and (_INTERVAL_RE.match(raw_val.strip()) or _THRESHOLD_RE.match(raw_val.strip()))
            ):
                n_interval += 1
        df[col] = parsed_vals
        conversion_counts[col] = n_interval

    return df, conversion_counts


# ---------------------------------------------------------------------------
# Numeric type conversion
# ---------------------------------------------------------------------------

def convert_columns_to_numeric(
    df: pd.DataFrame,
    columns: List[str],
    log_prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Coerce *columns* to ``float64`` using :func:`pd.to_numeric` with
    ``errors='coerce'`` (non-parseable values become ``NaN``).

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Column names to convert.  Missing columns are skipped with a warning.
    log_prefix:
        Prepended to every log message (e.g. the dataset name).

    Returns
    -------
    (updated_df, conversion_log)
        *conversion_log* maps column name → dict with keys:
        ``already_numeric``, ``converted``, ``coerced_to_nan``, ``skipped``.
    """
    df = df.copy()
    conversion_log: Dict[str, Dict[str, int]] = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("%s Column '%s' not found – skipped.", log_prefix, col)
            conversion_log[col] = {"skipped": 1}
            continue

        series = df[col]
        already_numeric = pd.api.types.is_numeric_dtype(series)

        if already_numeric:
            df[col] = series.astype("float64")
            conversion_log[col] = {
                "already_numeric": 1,
                "converted": 0,
                "coerced_to_nan": 0,
            }
            continue

        n_before_null = int(series.isna().sum())
        converted = pd.to_numeric(series, errors="coerce").astype("float64")
        n_after_null = int(converted.isna().sum())
        n_coerced = n_after_null - n_before_null

        df[col] = converted
        conversion_log[col] = {
            "already_numeric": 0,
            "converted": int((~series.isna()).sum()) - n_coerced,
            "coerced_to_nan": n_coerced,
        }

        if n_coerced:
            logger.debug(
                "%s Column '%s': %d value(s) could not be parsed to float "
                "and were coerced to NaN.",
                log_prefix, col, n_coerced,
            )

    return df, conversion_log


# ---------------------------------------------------------------------------
# Categorical standardisation
# ---------------------------------------------------------------------------

def standardise_categorical_values(
    df: pd.DataFrame,
    columns: List[str],
    value_map: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    For every object-dtype column in *columns*, lower-case and strip each
    non-null cell, then look it up in *value_map*.  Values that have no
    mapping entry are left unchanged (but logged at DEBUG level).

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Subset of columns to standardise.
    value_map:
        ``{raw_lower: canonical}`` mapping loaded from YAML.

    Returns
    -------
    (updated_df, remap_counts)
        *remap_counts* maps column name → number of cells that were
        actually remapped to a different canonical form.
    """
    df = df.copy()
    remap_counts: Dict[str, int] = {}

    # Normalise keys to lower-case for case-insensitive matching.
    normalised_map = {k.lower().strip(): v for k, v in value_map.items()}

    for col in columns:
        if col not in df.columns:
            continue
        # Handle both legacy object dtype and Python 3.12+ StringDtype.
        if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
            continue

        n_remapped = 0
        new_values: List[Any] = []
        for raw in df[col]:
            if pd.isna(raw):
                new_values.append(np.nan)
                continue
            key = str(raw).lower().strip()
            if key in normalised_map:
                canonical = normalised_map[key]
                if canonical != str(raw).strip():
                    n_remapped += 1
                new_values.append(canonical)
            else:
                new_values.append(str(raw).strip())

        df[col] = new_values
        remap_counts[col] = n_remapped

    return df, remap_counts


# ---------------------------------------------------------------------------
# Target label cleaning
# ---------------------------------------------------------------------------

def clean_target_labels(
    series: pd.Series,
    valid_classes: Optional[List[str]] = None,
) -> Tuple[pd.Series, int]:
    """
    Strip leading/trailing whitespace and tab characters from a target
    label Series and optionally verify the resulting unique values against
    *valid_classes*.

    Parameters
    ----------
    series:
        Raw target column.
    valid_classes:
        If provided, the cleaned series must contain **only** values in this
        list (NaN excluded).  Raises ``ValueError`` if not.

    Returns
    -------
    (cleaned_series, n_cleaned)
        *n_cleaned* = number of values that changed after stripping.
    """
    original = series.copy()
    cleaned = series.astype(str).str.strip()
    # Restore NaN that was coerced to "nan" by astype(str).
    was_null = original.isna()
    cleaned[was_null] = np.nan

    # Lower-case for canonical comparison only.
    cleaned = cleaned.str.lower()

    n_changed = int((cleaned.fillna("__NULL__") != original.astype(str).str.lower().fillna("__NULL__")).sum())

    if valid_classes is not None:
        observed = set(cleaned.dropna().unique())
        invalid = observed - set(valid_classes)
        if invalid:
            raise ValueError(
                f"Target column contains invalid label(s) after cleaning: "
                f"{sorted(invalid)}.  Expected only: {sorted(valid_classes)}."
            )

    return cleaned, n_changed


# ---------------------------------------------------------------------------
# Encoding helpers (stateless – no sklearn objects)
# ---------------------------------------------------------------------------

def encode_binary_features(
    df: pd.DataFrame,
    columns: List[str],
    encoding_map: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Map canonical categorical string values in *columns* to integer codes
    using *encoding_map*.  NaN values are preserved as NaN.

    Parameters
    ----------
    df:
        Input DataFrame (post-imputation).
    columns:
        Columns to encode.
    encoding_map:
        ``{"yes": 1, "no": 0, ...}`` loaded from YAML.

    Returns
    -------
    (updated_df, encoding_counts)
        *encoding_counts* maps column name → number of non-null cells encoded.
    """
    df = df.copy()
    encoding_counts: Dict[str, int] = {}
    lower_map = {k.lower(): v for k, v in encoding_map.items()}

    for col in columns:
        if col not in df.columns:
            continue
        n_encoded = 0
        new_vals: List[Any] = []
        for val in df[col]:
            if pd.isna(val):
                new_vals.append(np.nan)
                continue
            key = str(val).lower().strip()
            if key in lower_map:
                new_vals.append(lower_map[key])
                n_encoded += 1
            else:
                new_vals.append(val)
        df[col] = new_vals
        encoding_counts[col] = n_encoded

    return df, encoding_counts


def encode_target_labels(
    series: pd.Series,
    label_map: Dict[Any, int],
) -> pd.Series:
    """
    Map target label strings/integers to integer codes using *label_map*.
    Raises ``KeyError`` if any non-NaN value has no entry in the map.

    Parameters
    ----------
    series:
        Cleaned target label column.
    label_map:
        Mapping such as ``{"ckd": 1, "notckd": 0}``.

    Returns
    -------
    pd.Series with integer dtype (NaN preserved as ``pd.NA``).
    """
    # Normalise keys to lower-string for safety.
    norm_map: Dict[str, int] = {}
    for k, v in label_map.items():
        norm_map[str(k).lower().strip()] = int(v)

    def _map(val: Any) -> Any:
        if pd.isna(val):
            return pd.NA
        key = str(val).lower().strip()
        if key not in norm_map:
            raise KeyError(
                f"Target value '{val}' has no entry in the label encoding map.  "
                f"Available keys: {list(norm_map.keys())}"
            )
        return norm_map[key]

    return series.map(_map)


# ---------------------------------------------------------------------------
# Post-preprocessing validation helpers
# ---------------------------------------------------------------------------

def validate_no_nan_targets(series: pd.Series, dataset_name: str) -> None:
    """Raises ``ValueError`` if any NaN remains in the target column."""
    n = int(series.isna().sum())
    if n:
        raise ValueError(
            f"[{dataset_name}] {n} NaN value(s) remain in the target column "
            f"after preprocessing.  All missing targets must be resolved."
        )


def validate_numeric_dtypes(
    df: pd.DataFrame,
    expected_numeric_cols: List[str],
    dataset_name: str,
) -> List[str]:
    """
    Returns a list of columns that were expected to be numeric but are not.
    Does NOT raise – caller decides whether to warn or raise.
    """
    non_numeric = [
        col for col in expected_numeric_cols
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        logger.warning(
            "[%s] The following columns were expected numeric but are not: %s",
            dataset_name, non_numeric,
        )
    return non_numeric


def validate_no_unexpected_categories(
    df: pd.DataFrame,
    columns: List[str],
    allowed_values: Dict[str, List[Any]],
    dataset_name: str,
) -> Dict[str, List[Any]]:
    """
    For each column in *columns* that has an entry in *allowed_values*,
    check that no non-NaN value falls outside the allowed set.

    Returns a dict mapping column name → list of unexpected values found.
    """
    unexpected: Dict[str, List[Any]] = {}
    for col in columns:
        if col not in df.columns or col not in allowed_values:
            continue
        allowed_set = set(str(v) for v in allowed_values[col])
        observed = set(str(v) for v in df[col].dropna().unique())
        bad = sorted(observed - allowed_set)
        if bad:
            unexpected[col] = bad
            logger.warning(
                "[%s] Column '%s' contains unexpected value(s) after encoding: %s",
                dataset_name, col, bad,
            )
    return unexpected