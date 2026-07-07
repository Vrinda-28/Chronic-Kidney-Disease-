"""
examples/external_validation_example.py
=========================================

Minimal demonstration of UAE external validation.

Usage:
    cd /path/to/Mom\ Kidney\ CKD
    python examples/external_validation_example.py

This example shows how to:
  1. Run UAE external validation programmatically
  2. Inspect Youden's J results vs default threshold
  3. Understand the population shift numbers
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from external_validation import CKDExternalValidator


def run_external_validation():
    """Run UAE external validation with full bootstrap CIs."""
    print("=" * 60)
    print("CKD UAE External Validation")
    print("=" * 60)
    print()
    print("Design:")
    print("  • Track A: 8-feature reduced CatBoost (trained on UCI only)")
    print("  • UAE: 491 cardiology outpatients (NEVER seen in training)")
    print("  • CKD prevalence in UAE: 11.4% (vs ~62% in UCI training)")
    print()

    validator = CKDExternalValidator(config_path="config/evaluation_config.yaml")
    result = validator.run(run_ci=True)   # Set False for faster run

    # Summarise key numbers
    pop = result.get("population_shift", {}).get("uae_cohort", {})
    print(f"\nUAE Cohort: {pop.get('n_patients')} patients | "
          f"CKD: {pop.get('n_ckd')} ({pop.get('ckd_prevalence_pct'):.1f}%)")

    at05 = result.get("at_threshold_0.5", {})
    at_y = result.get("at_youden_threshold", {})
    ythr = result.get("youden_threshold", "N/A")

    def _pt(v):
        if isinstance(v, dict):
            return f"{v.get('point', 'N/A'):.4f}"
        return f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

    print(f"\n{'Metric':<20} {'τ=0.50':>10} {'τ=Youden':>12}")
    print("-" * 46)
    for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "mcc", "accuracy"):
        print(f"  {k:<18} {_pt(at05.get(k)):>10} {_pt(at_y.get(k)):>12}")
    print(f"\n  Youden's J threshold: τ = {ythr}")

    calib = result.get("calibration", {})
    print(f"\n  ECE = {calib.get('ece', 'N/A'):.4f} | Brier = {calib.get('brier_score', 'N/A'):.4f}")

    print("\nArtifacts → artifacts/evaluation/uae/")
    print("Summary   → artifacts/evaluation/summary_report.md")
    return result


if __name__ == "__main__":
    result = run_external_validation()
