"""
examples/model_training_example.py
====================================

Demonstrates correct usage of model_training.py and documents the
exact patterns and output structure to expect.

Run from the project root:
    python examples/model_training_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd


def run_training() -> None:
    from model_training import CKDModelTrainer

    print("=" * 60)
    print("Running CKD model training pipeline …")
    print("=" * 60)

    trainer = CKDModelTrainer(config_path="config/model_config.yaml")
    results = trainer.train_all()

    print("\n── Results summary ──")
    for task_key, task_result in results.items():
        if task_key == "uae":
            continue
        print(f"\n{task_key.upper()} — {task_result.task_type}")
        print(f"  Best model      : {task_result.best_model_name}")
        print(f"  Primary metric  : {task_result.primary_metric}")
        print(f"  Best CV score   : {task_result.best_cv_score:.4f}")
        print("  All model CV scores:")
        for name, res in task_result.model_results.items():
            print(f"    {name:<20}: {res.primary_cv_score:.4f}")

        print("  Test set metrics (best model):")
        for k, v in task_result.test_metrics_best.items():
            if isinstance(v, float):
                print(f"    {k:<30}: {v:.4f}")

    uci = results.get("uci")
    if uci and uci.uae_metrics and not uci.uae_metrics.get("skipped"):
        print("\nUAE External Validation:")
        for k, v in uci.uae_metrics.items():
            if isinstance(v, float):
                print(f"  {k:<30}: {v:.4f}")

    return results


def inspect_artifacts() -> None:
    """Load and inspect saved artifacts from disk."""
    print("\n" + "=" * 60)
    print("Inspecting saved artifacts")
    print("=" * 60)

    # UCI best model
    uci_dir = Path("artifacts/models/uci")
    best_model_path = uci_dir / "best_model.json"
    if best_model_path.exists():
        with open(best_model_path) as fh:
            best = json.load(fh)
        best_name = best.get("best_model")
        print(f"\nUCI best model: {best_name}")
        print(f"CV {best.get('primary_metric')}: {best.get('best_cv_score'):.4f}")

        # CV summary for best model
        cv_summary_path = uci_dir / best_name / "cv_summary.json"
        if cv_summary_path.exists():
            with open(cv_summary_path) as fh:
                cv_summary = json.load(fh)
            print(f"\nCV summary for {best_name}:")
            for metric, stats in cv_summary.items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {metric:<25}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Feature importances
        fi_path = uci_dir / best_name / "feature_importance.json"
        if fi_path.exists():
            with open(fi_path) as fh:
                fi = json.load(fh)
            print(f"\nTop 10 features ({best_name}):")
            for i, (feat, score) in enumerate(list(fi.items())[:10]):
                print(f"  {i+1:2}. {feat:<40}: {score:.6f}")

        # Selected features
        sf_path = uci_dir / best_name / "selected_features.json"
        if sf_path.exists():
            with open(sf_path) as fh:
                sf = json.load(fh)
            print(f"\nUnion features selected: {sf.get('n_union_features')} features")

        # Test predictions
        preds_path = uci_dir / best_name / "test_predictions.csv"
        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            print(f"\nTest predictions shape: {preds.shape}")
            print(preds.head(5).to_string(index=False))

    # Kaggle best model
    kaggle_dir = Path("artifacts/models/kaggle")
    kaggle_best_path = kaggle_dir / "best_model.json"
    if kaggle_best_path.exists():
        with open(kaggle_best_path) as fh:
            kaggle_best = json.load(fh)
        print(f"\nKaggle best model: {kaggle_best.get('best_model')}")
        print(f"CV {kaggle_best.get('primary_metric')}: {kaggle_best.get('best_cv_score'):.4f}")

    # Global summary
    global_path = Path("artifacts/models/best_model_summary.json")
    if global_path.exists():
        print(f"\nGlobal best model summary saved at: {global_path}")


def demonstrate_leakage_guards() -> None:
    """Confirm leakage guards raise correctly."""
    print("\n" + "=" * 60)
    print("Leakage guard demonstration")
    print("=" * 60)

    from model_utils import (
        LeakageViolation,
        assert_no_target_in_features,
        assert_no_uae_in_training,
        assert_val_not_in_train_indices,
    )

    # Guard 1: target in features
    print("\nTest 1: Target column in features →")
    try:
        assert_no_target_in_features(
            ["age", "serum_creatinine", "ckd_label"],
            "ckd_label",
            context="example",
        )
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] {str(exc)[:80]}…")

    # Guard 2: UAE == train
    print("\nTest 2: UAE same object as train →")
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "ckd_label": [0, 1]})
    try:
        assert_no_uae_in_training(df, df, context="example")
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] {str(exc)[:80]}…")

    # Guard 3: val in train indices
    print("\nTest 3: Val index in train indices →")
    try:
        assert_val_not_in_train_indices(
            train_indices=[0, 1, 2, 3, 4],
            val_indices=[3, 4, 5],       # 3 and 4 overlap
            fold_num=0,
            dataset_name="UCI",
        )
        print("  [FAIL] Should have raised LeakageViolation.")
    except LeakageViolation as exc:
        print(f"  [PASS] {str(exc)[:80]}…")

    print("\n[All leakage guard tests passed.]")


if __name__ == "__main__":
    results = run_training()
    inspect_artifacts()
    demonstrate_leakage_guards()

    print("\n── Example complete ──")
    print("Artifacts → artifacts/models/")
    print("Best model summary → artifacts/models/best_model_summary.json")