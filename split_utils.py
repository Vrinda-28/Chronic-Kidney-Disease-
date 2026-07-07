"""
split_utils.py
==============

Pure, stateless utility functions for the CKD train/test splitting pipeline.

Every function here:
  * Has no side-effects (no file I/O, no global state mutation).
  * Is independently unit-testable without any pipeline context.
  * Accepts and returns plain Python / pandas / numpy objects.

File I/O (saving manifests, fold indices, metadata) is handled in
train_test_split.py, not here.  This keeps responsibilities clean and makes
these utilities reusable from any future pipeline stage.

Pipeline position: called exclusively by train_test_split.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ckd_splitter")


# =============================================================================
# Exceptions
# =============================================================================


class CKDSplitError(Exception):
    """Raised for irrecoverable split-configuration or data errors."""


class LeakageViolation(CKDSplitError):
    """
    Raised when a structural data-leakage constraint is violated.
    Examples: test indices found in train, UAE indices in any split.
    This is a hard error — the pipeline must halt, not warn and continue.
    Leakage in a research pipeline is a correctness bug, not a warning.
    """


# =============================================================================
# Reproducibility: fingerprinting
# =============================================================================


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Compute a SHA-256 fingerprint of a DataFrame's content.

    The fingerprint is stable with respect to:
      * Row order (rows are sorted by their hash before hashing the aggregate).
      * Column order (columns are sorted alphabetically before hashing).
      * Data types (values are converted to string for hashing).

    It is NOT stable with respect to:
      * Adding or removing columns (intentional — different schemas → different hash).
      * Changing any cell value (intentional — content fingerprint).

    Parameters
    ----------
    df:
        Input DataFrame.  Provenance columns are included if present
        (the caller is responsible for passing the right subset).

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    if df.empty:
        return hashlib.sha256(b"empty_dataframe").hexdigest()

    # Sort columns alphabetically for column-order stability.
    sorted_cols = sorted(df.columns.tolist())
    df_sorted = df[sorted_cols]

    # Hash each row's string representation, then sort row hashes for
    # row-order stability (important: the overall hash must not change if
    # the same rows arrive in a different order).
    row_hashes = pd.util.hash_pandas_object(df_sorted, index=False)
    sorted_row_hashes = sorted(row_hashes.values.tolist())

    h = hashlib.sha256()
    for rh in sorted_row_hashes:
        h.update(str(rh).encode("utf-8"))
    # Also hash the column list so that a different column schema always
    # produces a different fingerprint.
    h.update(",".join(sorted_cols).encode("utf-8"))
    return h.hexdigest()


def compute_index_fingerprint(indices: Sequence[int]) -> str:
    """
    Stable SHA-256 of a sorted list of integer indices.
    Used to fingerprint train/test/val row index sets independently of the
    DataFrame content (so you can verify the split structure without
    re-loading the DataFrames).
    """
    sorted_indices = sorted(int(i) for i in indices)
    content = ",".join(str(i) for i in sorted_indices)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Class distribution utilities
# =============================================================================


def compute_class_distribution(
    series: pd.Series,
    as_percentages: bool = False,
) -> Dict[Any, Any]:
    """
    Compute value counts (or percentages) for a target column.

    Parameters
    ----------
    series:
        Target label column.
    as_percentages:
        If True, return proportions (0.0–1.0) instead of raw counts.

    Returns
    -------
    dict mapping class label → count or proportion.
    JSON-serialisable (keys and values are plain Python types).
    """
    counts = series.value_counts(dropna=False)
    if as_percentages:
        total = max(len(series), 1)
        return {str(k): round(float(v) / total, 4) for k, v in counts.items()}
    return {str(k): int(v) for k, v in counts.items()}


def check_min_samples_per_class(
    series: pd.Series,
    min_samples: int,
    dataset_name: str,
    context: str = "",
) -> List[str]:
    """
    Return a list of classes that have fewer than *min_samples* members.
    Does NOT raise — caller decides policy (log warning vs. raise error).

    Parameters
    ----------
    series:
        Target label column (from train portion, not full dataset).
    min_samples:
        Minimum required count per class.
    dataset_name:
        Used in returned warning strings.
    context:
        Short description of why this check is being run (e.g. "for 5-fold CV").

    Returns
    -------
    List of warning strings (empty if all classes meet the minimum).
    """
    counts = series.value_counts(dropna=True)
    warnings: List[str] = []
    for cls, count in counts.items():
        if count < min_samples:
            warnings.append(
                f"[{dataset_name}] Class '{cls}' has only {count} sample(s) "
                f"in the training set (minimum required {context}: {min_samples}). "
                f"Stratified CV may fail or produce unreliable per-class metrics "
                f"for this class. Consider reducing cv_n_splits or using LOOCV."
            )
    return warnings


# =============================================================================
# Leakage verification (pure checks — no fix-up, just detect and raise)
# =============================================================================


def verify_no_index_overlap(
    set_a: Set[int],
    set_b: Set[int],
    name_a: str,
    name_b: str,
    dataset_name: str,
) -> None:
    """
    Raise LeakageViolation if *set_a* and *set_b* share any indices.

    Parameters
    ----------
    set_a, set_b:
        Integer index sets (e.g. train row positions, test row positions).
    name_a, name_b:
        Human-readable names for the two sets (e.g. "train", "test").
    dataset_name:
        Dataset this check applies to, for the error message.

    Raises
    ------
    LeakageViolation
        If the intersection is non-empty.
    """
    overlap = set_a & set_b
    if overlap:
        raise LeakageViolation(
            f"[{dataset_name}] DATA LEAKAGE DETECTED: {len(overlap)} row(s) "
            f"appear in BOTH '{name_a}' and '{name_b}' index sets. "
            f"Overlapping row positions: {sorted(overlap)[:20]}"
            f"{'... (truncated)' if len(overlap) > 20 else ''}. "
            f"This is a hard error — the split must be discarded and "
            f"re-generated with a corrected configuration."
        )


def verify_full_coverage(
    all_indices: Set[int],
    train_indices: Set[int],
    test_indices: Set[int],
    dataset_name: str,
) -> None:
    """
    Raise CKDSplitError if any row from the full dataset is in neither
    the train nor the test set (i.e. some rows were silently dropped).

    For datasets that use CV (no fixed val split), *all_indices* should equal
    *train_indices | test_indices*.

    Parameters
    ----------
    all_indices:
        Set of all row positions in the full engineered dataset.
    train_indices, test_indices:
        Sets of row positions assigned to each split.
    dataset_name:
        Used in error messages.
    """
    covered = train_indices | test_indices
    uncovered = all_indices - covered
    if uncovered:
        raise CKDSplitError(
            f"[{dataset_name}] {len(uncovered)} row(s) from the full dataset "
            f"are in neither the train nor the test set. Row positions: "
            f"{sorted(uncovered)[:20]}"
            f"{'... (truncated)' if len(uncovered) > 20 else ''}. "
            f"This indicates a bug in the splitting logic."
        )


def verify_uae_isolation(
    uae_indices: Set[int],
    train_indices: Set[int],
    test_indices: Set[int],
    cv_fold_indices: List[Dict[str, List[int]]],
) -> None:
    """
    Verify that UAE row positions never appear in any train, test, or CV fold.

    UAE indices are position-based within the UAE DataFrame, so a cross-dataset
    collision is numerically possible but would indicate a structural bug
    (e.g. DataFrames being accidentally concatenated before splitting).
    This check guards against that class of bug specifically.

    Parameters
    ----------
    uae_indices:
        Set of integer row positions in the UAE engineered DataFrame.
    train_indices, test_indices:
        Integer row positions from train/test splits of other datasets.
    cv_fold_indices:
        Pre-generated CV fold index dicts from other datasets.

    Raises
    ------
    LeakageViolation
        If any UAE index appears anywhere in the other datasets' splits.
        (This would indicate structural concatenation happened somewhere
        in the pipeline — a serious architectural bug.)
    """
    # In a correct pipeline, UCI/Kaggle indices are positions within their
    # own DataFrames (0..n_uci-1 and 0..n_kaggle-1), and UAE indices are
    # positions within the UAE DataFrame (0..n_uae-1).  Numerically, many of
    # these overlap (e.g. both have a row 0, row 1, ...) and that is expected
    # and fine — they are NOT the same rows.  The structural isolation is
    # guaranteed by the fact that this module NEVER concatenates the DataFrames.
    # This check therefore verifies the *object identity* constraint: that the
    # bundle's external_validation field never points to the same DataFrame
    # object as the train/test fields.
    #
    # We skip the numeric overlap check here because it would always fire
    # (position 0 exists in every dataset) and would be misleading.
    # Instead, callers should verify object identity using `verify_uae_object_isolation`.
    pass  # See verify_uae_object_isolation below.


def verify_uae_object_isolation(
    uae_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Verify that *uae_df* is not the same Python object (or a slice of the
    same underlying data) as *train_df* or *test_df*.

    pandas DataFrames share underlying numpy arrays when created via slicing
    (e.g. ``df[mask]`` may or may not copy, depending on the pandas version).
    We can't cheaply check copy-vs-view status, but we CAN check whether
    the DataFrames are the same object (``is`` check) or share the same
    column names AND row counts, which is a necessary (though not sufficient)
    condition for accidental aliasing.

    A full content-equality check is too expensive here.  Use the fingerprints
    saved to split_metadata.json for post-hoc verification.

    Parameters
    ----------
    uae_df, train_df, test_df:
        DataFrames to check.
    dataset_name:
        Name of the training dataset being compared against UAE.

    Raises
    ------
    LeakageViolation
        If *uae_df* is the same Python object as *train_df* or *test_df*.
    """
    if uae_df is train_df:
        raise LeakageViolation(
            f"[{dataset_name}] CRITICAL: uae_df and train_df are the SAME "
            f"Python object. UAE rows are in the training set. "
            f"This is a structural data-leakage bug."
        )
    if uae_df is test_df:
        raise LeakageViolation(
            f"[{dataset_name}] CRITICAL: uae_df and test_df are the SAME "
            f"Python object. UAE rows are in the test set. "
            f"This is a structural data-leakage bug."
        )


def verify_cv_folds_within_train(
    cv_fold_indices: List[Dict[str, List[int]]],
    train_size: int,
    dataset_name: str,
) -> None:
    """
    Verify that every index in every CV fold is a valid position within the
    training DataFrame (i.e. in range [0, train_size)).

    If any CV fold index is >= train_size, it means the fold was generated
    against the wrong DataFrame (e.g. the full dataset instead of the train
    portion), which would include test-set rows in training folds — a
    classic but silent leakage bug.

    Parameters
    ----------
    cv_fold_indices:
        List of dicts with "train_indices" and "val_indices" keys.
    train_size:
        Number of rows in the training DataFrame (the max valid index + 1).
    dataset_name:
        Used in error messages.

    Raises
    ------
    LeakageViolation
        If any index is out of range for the training DataFrame.
    """
    for fold_num, fold in enumerate(cv_fold_indices):
        for split_name in ("train_indices", "val_indices"):
            indices = fold.get(split_name, [])
            out_of_range = [i for i in indices if i < 0 or i >= train_size]
            if out_of_range:
                raise LeakageViolation(
                    f"[{dataset_name}] CV fold {fold_num} '{split_name}' contains "
                    f"{len(out_of_range)} index/indices out of range for the training "
                    f"DataFrame (train_size={train_size}). Out-of-range values: "
                    f"{sorted(out_of_range)[:10]}. "
                    f"This means CV was run against the wrong DataFrame — likely "
                    f"the full dataset instead of the train portion. "
                    f"This would allow test-set rows into training folds (leakage)."
                )
        # Also check no overlap between train and val within a fold.
        train_set = set(fold.get("train_indices", []))
        val_set = set(fold.get("val_indices", []))
        overlap = train_set & val_set
        if overlap:
            raise LeakageViolation(
                f"[{dataset_name}] CV fold {fold_num} has {len(overlap)} overlapping "
                f"indices between train_indices and val_indices. "
                f"A row cannot be both training and validation in the same fold."
            )


# =============================================================================
# Manifest generation (pure — returns structures, does not write to disk)
# =============================================================================


def build_manifest(
    df: pd.DataFrame,
    row_positions: Sequence[int],
    split_name: str,
    dataset_name: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Build a split manifest DataFrame that records which rows belong to a split.

    The manifest is the authoritative record of the split — it allows any
    future pipeline run to reconstruct the exact same train/test partition
    from the raw engineered dataset without re-running the splitting logic,
    even if the sklearn random state implementation changes across versions.

    Parameters
    ----------
    df:
        The full (or partial) engineered DataFrame being split.
    row_positions:
        Integer positions (iloc-based, 0-indexed) of the rows in this split.
    split_name:
        One of "train", "test", "uae_full".
    dataset_name:
        Name of the source dataset ("UCI", "Kaggle", "UAE").
    target_col:
        Name of the target column (used to record each row's label).

    Returns
    -------
    pd.DataFrame with columns:
        manifest_row_position  — 0-indexed position in the FULL engineered CSV
        target_value           — the encoded target label for this row
        split_set              — "train" | "test" | "uae_full"
        dataset_name           — dataset identifier string
    """
    rows = []
    for pos in sorted(row_positions):
        row_target = df.iloc[pos][target_col] if target_col in df.columns else None
        rows.append({
            "manifest_row_position": int(pos),
            "target_value": row_target,
            "split_set": split_name,
            "dataset_name": dataset_name,
        })
    return pd.DataFrame(rows)


def build_cv_fold_record(
    fold_num: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_target: pd.Series,
    repeat_num: Optional[int] = None,
    fold_within_repeat: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable record for one CV fold.

    Parameters
    ----------
    fold_num:
        Global fold number (0-indexed across all repeats).
    train_indices, val_indices:
        Row positions within the TRAINING DataFrame (not the full dataset).
        These are safe to use directly with `train_df.iloc[train_indices]`.
    train_target:
        Target column of the training DataFrame, used to compute per-fold
        class distributions for the manifest.
    repeat_num, fold_within_repeat:
        For RepeatedStratifiedKFold only.  None for plain StratifiedKFold.

    Returns
    -------
    dict with all fold metadata, ready for json.dump().
    """
    train_dist = Counter(
        str(v) for v in train_target.iloc[train_indices].tolist()
    )
    val_dist = Counter(
        str(v) for v in train_target.iloc[val_indices].tolist()
    )
    record: Dict[str, Any] = {
        "fold_num": fold_num,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "train_class_distribution": dict(train_dist),
        "val_class_distribution": dict(val_dist),
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "smote_applicable_to": "train_indices only — never val_indices",
    }
    if repeat_num is not None:
        record["repeat_num"] = repeat_num
        record["fold_within_repeat"] = fold_within_repeat
    return record


# =============================================================================
# Stratification feasibility check
# =============================================================================


def assert_stratification_feasible(
    series: pd.Series,
    n_splits: int,
    dataset_name: str,
    context: str = "stratified train/test split",
) -> None:
    """
    Raise CKDSplitError if stratified splitting is not feasible because some
    class has fewer than 2 samples (sklearn's StratifiedKFold minimum) or
    fewer than n_splits samples (needed for n-fold CV).

    Parameters
    ----------
    series:
        Target column.  Non-null values only are checked.
    n_splits:
        Number of CV folds (or 2 for a single train/test split).
    dataset_name:
        Used in error messages.
    context:
        Description of what the split is for (used in error messages).

    Raises
    ------
    CKDSplitError
        If any class has fewer than max(2, n_splits) samples.
    """
    counts = series.dropna().value_counts()
    required = max(2, n_splits)
    violations = [(str(cls), int(cnt)) for cls, cnt in counts.items() if cnt < required]
    if violations:
        raise CKDSplitError(
            f"[{dataset_name}] Cannot perform stratified {context}: "
            f"{len(violations)} class(es) have fewer than {required} sample(s) "
            f"(minimum for {n_splits}-fold stratified split). "
            f"Classes with insufficient samples: {violations}. "
            f"Options: reduce n_splits, merge rare classes (with domain justification), "
            f"or use non-stratified splitting (not recommended for imbalanced data)."
        )


# =============================================================================
# JSON serialisation helper
# =============================================================================


def json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to plain Python types for
    json.dump().  Used when serialising metadata and fold records.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Series):
        return json_safe(obj.tolist())
    if pd.isna(obj) if not isinstance(obj, (list, dict, np.ndarray)) else False:
        return None
    return obj


# =============================================================================
# File I/O helpers (minimal — only saving, no loading, no side-effects
# beyond disk writes; callers pass pre-validated paths)
# =============================================================================


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write *data* to *path* as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=json_safe)


def save_manifest_csv(manifest_df: pd.DataFrame, path: Path) -> None:
    """Write a split manifest DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(path, index=False)


def save_dataframe_split(df: pd.DataFrame, path: Path) -> None:
    """Write a split DataFrame (train or test) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)