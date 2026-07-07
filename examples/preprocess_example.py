"""
examples/preprocess_example.py
================================

Standalone, runnable example demonstrating how to invoke the CKD
preprocessing pipeline and inspect its outputs.

Run from the project root:
    python examples/preprocess_example.py

Prerequisites:
    - data_loader.py must exist (verified, do not modify).
    - config/datasets.yaml must exist (verified, do not modify).
    - config/preprocessing.yaml must exist.
    - Raw CSVs must exist at the paths in config/datasets.yaml.
    - pip packages: pandas, numpy, scikit-learn, joblib, pyyaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Ensure project root is on sys.path so sibling imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess import CKDPreprocessor  # noqa: E402


def main() -> None:
    print("=" * 65)
    print("  CKD Preprocessing Pipeline – Example Run")
    print("=" * 65)

    # ── 1. Instantiate and run the preprocessor ───────────────────────────
    preprocessor = CKDPreprocessor(
        config_path=str(PROJECT_ROOT / "config" / "preprocessing.yaml"),
        datasets_config_path=str(PROJECT_ROOT / "config" / "datasets.yaml"),
    )

    uci_df, kaggle_df, uae_df = preprocessor.run()

    # ── 2. Inspect processed DataFrames ───────────────────────────────────
    print("\n── Processed dataset shapes ──")
    print(f"UCI    : {uci_df.shape}  (target: ckd_label)")
    print(f"Kaggle : {kaggle_df.shape}  (target: ckd_stage_label)")
    print(f"UAE    : {uae_df.shape}  (target: ckd_label)")

    print("\n── UCI head (first 3 rows) ──")
    print(uci_df.head(3).to_string())

    print("\n── Kaggle head (first 3 rows) ──")
    print(kaggle_df.head(3).to_string())

    print("\n── UAE head (first 3 rows) ──")
    print(uae_df.head(3).to_string())

    # ── 3. Inspect target distributions ───────────────────────────────────
    print("\n── Target distributions after preprocessing ──")
    print("UCI ckd_label:")
    print(uci_df["ckd_label"].value_counts().to_string())

    print("\nKaggle ckd_stage_label:")
    print(kaggle_df["ckd_stage_label"].value_counts().to_string())

    print("\nUAE ckd_label:")
    print(uae_df["ckd_label"].value_counts().to_string())

    # ── 4. Confirm no NaN targets ─────────────────────────────────────────
    print("\n── NaN target counts ──")
    print(f"UCI    NaN targets : {uci_df['ckd_label'].isna().sum()}")
    print(f"Kaggle NaN targets : {kaggle_df['ckd_stage_label'].isna().sum()}")
    print(f"UAE    NaN targets : {uae_df['ckd_label'].isna().sum()}")

    # ── 5. Confirm no residual missing values in numeric columns ──────────
    print("\n── Residual missing values (numeric columns) ──")
    for name, df, key in [
        ("UCI", uci_df, "uci"),
        ("Kaggle", kaggle_df, "kaggle"),
        ("UAE", uae_df, "uae"),
    ]:
        total_nan = int(df.isna().sum().sum())
        print(f"{name:6s} total NaN cells : {total_nan}")

    # ── 6. Load and inspect saved artifacts ───────────────────────────────
    artifacts_dir = PROJECT_ROOT / "artifacts" / "preprocessing"

    print("\n── Saved artifacts ──")
    for filename in [
        "numeric_imputer.joblib",
        "categorical_imputer.joblib",
        "encoders.joblib",
        "label_mappings.json",
    ]:
        path = artifacts_dir / filename
        exists = path.exists()
        size = f"{path.stat().st_size:,} bytes" if exists else "NOT FOUND"
        print(f"  {filename:<35} {'✔' if exists else '✘'}  {size}")

    print("\n── Label mappings ──")
    label_mappings_path = artifacts_dir / "label_mappings.json"
    if label_mappings_path.exists():
        with open(label_mappings_path, "r", encoding="utf-8") as fh:
            mappings = json.load(fh)
        print(json.dumps(mappings, indent=2))

    print("\n── Numeric imputer (UCI) statistics ──")
    numeric_imputers = joblib.load(artifacts_dir / "numeric_imputer.joblib")
    uci_num_imp = numeric_imputers.get("uci")
    if uci_num_imp is not None:
        features = list(uci_num_imp.feature_names_in_)
        statistics = uci_num_imp.statistics_.tolist()
        print(pd.Series(statistics, index=features).to_string())

    # ── 7. Load and summarise the preprocessing report ────────────────────
    report_path = artifacts_dir / "preprocessing_summary.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        print("\n── Preprocessing summary ──")
        for ds_name, ds_report in report.get("datasets", {}).items():
            print(f"\n  [{ds_name.upper()}]")
            print(f"    Rows before : {ds_report.get('rows_before')}")
            print(f"    Rows after  : {ds_report.get('rows_after')}")
            print(f"    Rows removed: {ds_report.get('rows_removed')}")
            print(f"    Missing before: {ds_report.get('missing_values_before')}")
            print(f"    Missing after : {ds_report.get('missing_values_after')}")
            print(f"    Class dist before: {ds_report.get('class_distribution_before')}")
            print(f"    Class dist after : {ds_report.get('class_distribution_after')}")
            if ds_report.get("interval_conversions"):
                print(f"    Interval conversions: {ds_report['interval_conversions']}")
            if ds_report.get("validation_issues"):
                print(f"    ⚠ Validation issues: {ds_report['validation_issues']}")

    # ── 8. Confirm UAE was never merged with training data ─────────────────
    print("\n── UAE isolation check ──")
    # The UAE DataFrame must share no row from UCI or Kaggle provenance.
    if "source_dataset" in uae_df.columns:
        sources = uae_df["source_dataset"].unique().tolist()
        print(f"UAE source_dataset values: {sources}")
        assert all(s == "UAE" for s in sources), (
            "UAE rows contain non-UAE provenance – isolation violated!"
        )
        print("✔ UAE dataset provenance is isolated (only 'UAE' rows present).")
    else:
        print("  (source_dataset column not present in UAE – provenance check skipped)")

    print("\n" + "=" * 65)
    print("  Example run complete.  See logs/preprocess.log for full detail.")
    print("=" * 65)


if __name__ == "__main__":
    main()