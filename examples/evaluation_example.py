"""
examples/evaluation_example.py
================================

Minimal demonstration of the CKD evaluation pipeline.

Usage:
    cd /path/to/Mom\ Kidney\ CKD
    python examples/evaluation_example.py

This example shows how to:
  1. Run the full evaluation pipeline programmatically
  2. Load results for downstream use
  3. Run specific tasks only
  4. Disable SHAP/CI for speed
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import CKDEvaluator


def run_full_evaluation():
    """Run complete evaluation for both UCI and Kaggle."""
    print("=" * 60)
    print("CKD Evaluation — Full Run")
    print("=" * 60)

    evaluator = CKDEvaluator(config_path="config/evaluation_config.yaml")
    results = evaluator.run(
        tasks=["uci", "kaggle"],
        run_shap=True,      # Set False for ~10x faster run
        run_ci=True,        # Set False to skip bootstrap CIs
    )

    # Inspect UCI results
    uci = results.get("uci", {})
    if uci:
        print("\nUCI Binary CKD — Best Model:", uci.get("model_name"))
        test_m = uci.get("test_metrics", {})
        for key in ("roc_auc", "f1", "sensitivity", "specificity", "mcc"):
            v = test_m.get(key, {})
            if isinstance(v, dict):
                pt = v.get("point", "N/A")
                lo, hi = v.get("ci_lower", ""), v.get("ci_upper", "")
                ci = f" [95% CI: {lo:.4f}–{hi:.4f}]" if lo else ""
                print(f"  {key:<20}: {pt:.4f}{ci}")
            elif isinstance(v, (int, float)):
                print(f"  {key:<20}: {v:.4f}")

    # Inspect Kaggle results
    kaggle = results.get("kaggle", {})
    if kaggle:
        print("\nKaggle 5-Class Staging — Best Model:", kaggle.get("model_name"))
        test_m = kaggle.get("test_metrics", {})
        for key in ("balanced_accuracy", "macro_f1", "cohen_kappa", "macro_roc_auc"):
            v = test_m.get(key) or test_m.get(f"test_{key}")
            if isinstance(v, (int, float)):
                print(f"  {key:<25}: {v:.4f}")

    print("\nArtifacts → artifacts/evaluation/")
    return results


def run_uci_only_no_shap():
    """Quick evaluation of UCI only without SHAP (much faster)."""
    print("\n" + "=" * 60)
    print("CKD Evaluation — UCI Only, No SHAP")
    print("=" * 60)

    evaluator = CKDEvaluator(config_path="config/evaluation_config.yaml")
    results = evaluator.run(
        tasks=["uci"],
        run_shap=False,
        run_ci=False,
    )
    print("Done. Check artifacts/evaluation/uci/")
    return results


if __name__ == "__main__":
    # Run full evaluation by default
    results = run_full_evaluation()

    # Uncomment below for quick test:
    # results = run_uci_only_no_shap()
