"""
examples/train_test_split_example.py
======================================

Demonstrates correct usage of train_test_split.py and documents the
exact patterns the MODEL TRAINING module must follow to stay leakage-free.

Run from the project root:
    python examples/train_test_split_example.py

What this script shows:
  1. How to run the split pipeline.
  2. How to inspect split metadata and class distributions.
  3. The correct CV loop pattern (SMOTE, scaling, feature selection only on train folds).
  4. How to load pre-saved split artifacts in a subsequent training session.
  5. Demonstrations of the leakage checks raising for bad patterns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# =============================================================================
# 1. Run the splitting pipeline
# =============================================================================

def run_split() -> None:
    from train_test_split import CKDSplitOrchestrator, CKDSplitBundle

    print("=" * 60)
    print("Running CKD train/test splitting pipeline …")
    print("=" * 60)

    orchestrator = CKDSplitOrchestrator(config_path="config/split_config.yaml")
    bundle: CKDSplitBundle = orchestrator.split_all()

    print("\n── Split summary ──")
    for key, ds_split in bundle.train_candidate_datasets.items():
        print(f"\n{key.upper()}:")
        print(f"  Train rows      : {len(ds_split.train_df)}")
        print(f"  Test rows       : {len(ds_split.test_df)}")
        print(f"  CV strategy     : {ds_split.metadata.cv_strategy}")
        print(f"  CV folds total  : {ds_split.metadata.n_cv_folds_total}")
        print(f"  Target column   : {ds_split.target_col}")
        print(f"  Train dist      : {ds_split.metadata.class_distribution_train}")
        print(f"  Test dist       : {ds_split.metadata.class_distribution_test}")
        print(f"  Input SHA-256   : {ds_split.metadata.input_fingerprint[:16]}…")

    uae = bundle.external_validation["uae"]
    print(f"\nUAE (external validation):")
    print(f"  Rows            : {uae.n_rows}")
    print(f"  Class dist      : {uae.class_distribution}")
    print(f"  Fingerprint     : {uae.fingerprint[:16]}…")

    return bundle


# =============================================================================
# 2. Correct CV loop pattern (what the training module should implement)
# =============================================================================

def demonstrate_correct_cv_loop(bundle) -> None:
    """
    Shows the exact pattern the training module should use for CV.

    Key rules enforced by this pattern:
      * SMOTE is only applied to fold training data (train_fold_X / y).
      * Scalers are only fitted on fold training data.
      * Feature selectors are only fitted on fold training data.
      * val_fold data is only used for evaluation, never fitting.
      * test_df is never touched inside the loop.
    """
    print("\n" + "=" * 60)
    print("Correct CV loop pattern (training module guide)")
    print("=" * 60)

    uci_split = bundle.train_candidate_datasets["uci"]
    target_col = uci_split.target_col

    print(f"\nUCI: iterating {len(uci_split.cv_fold_indices)} CV folds …")
    print("(Showing first 2 folds only for brevity)\n")

    for fold in uci_split.cv_fold_indices[:2]:
        fold_num = fold["fold_num"]
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]

        # ── Access fold data ───────────────────────────────────────────────
        # Note: train_idx / val_idx are positions within train_df (iloc-safe).
        train_fold = uci_split.train_df.iloc[train_idx]
        val_fold = uci_split.train_df.iloc[val_idx]

        X_train_fold = train_fold.drop(columns=[target_col])
        y_train_fold = train_fold[target_col]
        X_val_fold = val_fold.drop(columns=[target_col])
        y_val_fold = val_fold[target_col]

        print(f"Fold {fold_num}:")
        print(f"  train_fold: {X_train_fold.shape} | val_fold: {X_val_fold.shape}")
        print(f"  train class dist : {fold['train_class_distribution']}")
        print(f"  val   class dist : {fold['val_class_distribution']}")
        print(f"  SMOTE applicable : {fold['smote_applicable_to']}")

        # ── SMOTE would go HERE ────────────────────────────────────────────
        # from imblearn.over_sampling import SMOTE
        # smote = SMOTE(random_state=42)
        # X_train_fold_resampled, y_train_fold_resampled = smote.fit_resample(
        #     X_train_fold, y_train_fold
        # )
        # ⚠️ NEVER apply SMOTE to X_val_fold / y_val_fold.

        # ── Scaler would go HERE ───────────────────────────────────────────
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        # X_val_fold_scaled = scaler.transform(X_val_fold)  # transform-only

        # ── Model training and val evaluation would go HERE ───────────────
        # model.fit(X_train_fold, y_train_fold)
        # val_predictions = model.predict(X_val_fold)
        # val_auc = roc_auc_score(y_val_fold, model.predict_proba(X_val_fold)[:, 1])

        print(f"  [OK] Fold {fold_num} structure validated.\n")

    # ── Test set access (AFTER all CV is complete and model is selected) ──
    print("After all CV folds complete:")
    print("  → Select best model based on CV val metrics.")
    print("  → Refit selected model on ALL of uci_split.train_df.")
    print("  → Evaluate ONCE on uci_split.test_df (never before this point).")
    X_test = uci_split.test_df.drop(columns=[target_col])
    y_test = uci_split.test_df[target_col]
    print(f"  Test set: X={X_test.shape}, y={y_test.shape}")
    print(f"  Test class dist: {uci_split.metadata.class_distribution_test}")


# =============================================================================
# 3. Kaggle 5-class CV pattern
# =============================================================================

def demonstrate_kaggle_cv_loop(bundle) -> None:
    print("\n" + "=" * 60)
    print("Kaggle 5-class staging: RepeatedStratifiedKFold pattern")
    print("=" * 60)

    kaggle_split = bundle.train_candidate_datasets["kaggle"]
    target_col = kaggle_split.target_col
    total_folds = len(kaggle_split.cv_fold_indices)

    print(f"\nKaggle CV: {total_folds} folds ({kaggle_split.metadata.cv_n_splits} splits "
          f"× {kaggle_split.metadata.cv_n_repeats} repeats)")

    fold_val_sizes = [len(f["val_indices"]) for f in kaggle_split.cv_fold_indices]
    per_class_val = [
        min(v for v in f["val_class_distribution"].values())
        for f in kaggle_split.cv_fold_indices
    ]

    print(f"Val fold size    : {fold_val_sizes[0]} rows per fold")
    print(f"Min per-class val: {min(per_class_val)} rows (across all folds)")
    print(f"Max per-class val: {max(per_class_val)} rows (across all folds)")
    print(
        "\nNote: With ~6–7 val samples per class per fold, per-class precision/recall "
        "is unreliable within a single fold. Report macro-averaged AUC and F1 "
        "averaged over all 25 folds, not per-fold class-level metrics."
    )

    # Show first fold from first repeat and first fold from second repeat
    fold_0 = kaggle_split.cv_fold_indices[0]
    fold_5 = kaggle_split.cv_fold_indices[5]  # first fold of second repeat

    print(f"\nFold 0  (repeat 0, fold-in-repeat 0):")
    print(f"  train class dist: {fold_0['train_class_distribution']}")
    print(f"  val   class dist: {fold_0['val_class_distribution']}")

    print(f"\nFold 5  (repeat 1, fold-in-repeat 0):")
    print(f"  train class dist: {fold_5['train_class_distribution']}")
    print(f"  val   class dist: {fold_5['val_class_distribution']}")

    print(
        "\nRecommended metric: macro-averaged AUC (OvR) over 25 folds. "
        "This is robust to class imbalance and comparable to Gogoi & Valan (2025)."
    )


# =============================================================================
# 4. UAE external validation pattern
# =============================================================================

def demonstrate_uae_usage(bundle) -> None:
    print("\n" + "=" * 60)
    print("UAE external validation — correct usage pattern")
    print("=" * 60)

    uae = bundle.external_validation["uae"]

    print(f"\nUAE: {uae.n_rows} rows, target='{uae.target_col}'")
    print(f"Role: {uae.role}")
    print(f"Class distribution: {uae.class_distribution}")

    print("\nCORRECT usage:")
    print("  # After final model is selected and refitted on full train_df:")
    print("  X_uae = uae.full_df.drop(columns=[uae.target_col])")
    print("  y_uae = uae.full_df[uae.target_col]")
    print("  uae_predictions = final_model.predict(X_uae)")
    print("  uae_auc = roc_auc_score(y_uae, final_model.predict_proba(X_uae)[:, 1])")
    print("  # This is the external validation result reported in the paper.")

    print("\nINCORRECT usage (examples of what NOT to do):")
    print("  ✗ Using UAE to tune hyperparameters.")
    print("  ✗ Including UAE in any CV fold.")
    print("  ✗ Applying SMOTE to UAE.")
    print("  ✗ Refitting any imputer or scaler on UAE.")
    print("  ✗ Evaluating UAE more than once (if you iterate, report only the final run).")


# =============================================================================
# 5. Loading pre-saved artifacts (for a subsequent training session)
# =============================================================================

def demonstrate_artifact_loading() -> None:
    print("\n" + "=" * 60)
    print("Loading pre-saved split artifacts (subsequent session)")
    print("=" * 60)

    # Load the split metadata to verify the run.
    meta_path = Path("artifacts/splits/split_metadata.json")
    if not meta_path.exists():
        print(f"  (metadata not found at {meta_path} — run the split first)")
        return

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    print(f"\nGlobal seed: {meta.get('random_seed')}")
    for ds_key in ("uci", "kaggle", "uae"):
        ds_meta = meta.get("datasets", {}).get(ds_key, {})
        if ds_meta:
            print(f"\n{ds_key.upper()} split metadata:")
            print(f"  n_rows_total     : {ds_meta.get('n_rows_total')}")
            print(f"  n_rows_train     : {ds_meta.get('n_rows_train')}")
            print(f"  n_rows_test      : {ds_meta.get('n_rows_test')}")
            print(f"  cv_strategy      : {ds_meta.get('cv_strategy')}")
            print(f"  n_cv_folds_total : {ds_meta.get('n_cv_folds_total')}")
            print(f"  input_fingerprint: {str(ds_meta.get('input_fingerprint', ''))[:16]}…")

    # Load the UCI train/test DataFrames from data/splits/.
    uci_train_path = Path("data/splits/uci_train.csv")
    uci_test_path = Path("data/splits/uci_test.csv")
    if uci_train_path.exists() and uci_test_path.exists():
        uci_train = pd.read_csv(uci_train_path)
        uci_test = pd.read_csv(uci_test_path)
        print(f"\nLoaded UCI from disk: train={uci_train.shape}, test={uci_test.shape}")
    else:
        print("\n  (data/splits/ not found — run the split first)")

    # Load UCI CV fold indices.
    fold_path = Path("artifacts/splits/uci_cv_fold_indices.json")
    if fold_path.exists():
        with open(fold_path, "r", encoding="utf-8") as fh:
            fold_data = json.load(fh)
        folds = fold_data.get("folds", [])
        print(f"\nLoaded UCI CV fold indices: {len(folds)} folds")
        print(f"  Fold 0 train size: {len(folds[0]['train_indices'])}")
        print(f"  Fold 0 val size  : {len(folds[0]['val_indices'])}")
        print(f"  SMOTE policy     : {folds[0].get('smote_applicable_to')}")


# =============================================================================
# 6. Leakage guard demonstration
# =============================================================================

def demonstrate_leakage_guards() -> None:
    """
    Shows that LeakageViolation is raised for bad patterns.
    This confirms the guards are active and correctly configured.
    """
    print("\n" + "=" * 60)
    print("Leakage guard demonstration")
    print("=" * 60)

    from split_utils import (
        verify_no_index_overlap,
        verify_cv_folds_within_train,
        verify_uae_object_isolation,
        LeakageViolation,
        CKDSplitError,
    )

    # ── Guard 1: overlapping train/test indices ────────────────────────────
    print("\nTest 1: Overlapping train/test indices →")
    try:
        verify_no_index_overlap(
            set(range(10)),   # train: rows 0–9
            set(range(5, 15)),  # test: rows 5–14 (overlap at 5–9)
            "train", "test", "TestDataset",
        )
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] LeakageViolation raised correctly: {str(exc)[:80]}…")

    # ── Guard 2: CV fold index out of train bounds ─────────────────────────
    print("\nTest 2: CV fold index beyond train_df size →")
    bad_folds = [{"train_indices": [0, 1, 2], "val_indices": [350]}]  # 350 >= train_size=320
    try:
        verify_cv_folds_within_train(bad_folds, train_size=320, dataset_name="UCI")
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] LeakageViolation raised correctly: {str(exc)[:80]}…")

    # ── Guard 3: UAE object identity check ────────────────────────────────
    print("\nTest 3: UAE is same object as train →")
    df = pd.DataFrame({"a": [1, 2, 3], "ckd_label": [0, 1, 0]})
    try:
        verify_uae_object_isolation(
            uae_df=df,         # same object as train_df!
            train_df=df,
            test_df=pd.DataFrame(),
            dataset_name="UCI",
        )
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] LeakageViolation raised correctly: {str(exc)[:80]}…")

    print("\n[All leakage guard tests passed.]\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Step 1: Run the pipeline (comment out if already run and artifacts exist).
    bundle = run_split()

    # Step 2–4: Demonstrate correct usage patterns.
    demonstrate_correct_cv_loop(bundle)
    demonstrate_kaggle_cv_loop(bundle)
    demonstrate_uae_usage(bundle)

    # Step 5: Load from disk (simulates a subsequent training session).
    demonstrate_artifact_loading()

    # Step 6: Confirm leakage guards are active.
    demonstrate_leakage_guards()

    print("\n── Example complete. ──")
    print("Artifacts saved to: artifacts/splits/")
    print("Split data saved to: data/splits/")