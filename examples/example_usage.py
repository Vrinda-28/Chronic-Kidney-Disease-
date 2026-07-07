"""
example_usage.py
=================
Minimal example showing how to invoke CKDDataLoader.

Run from the project root:
    python examples/example_usage.py
"""

import json
import sys
from pathlib import Path

# Allow running this script directly from the examples/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ckd_data.data_loader import CKDDataLoader, CKDDataError


def main() -> None:
    loader = CKDDataLoader(config_path="config/datasets.yaml")

    try:
        bundle = loader.load_all()
    except CKDDataError as exc:
        print(f"Data loading failed: {exc}")
        sys.exit(1)

    # ---- Train-candidate datasets (UCI, Kaggle) ----
    print("\n=== TRAIN-CANDIDATE DATASETS ===")
    for name, df in bundle.train_candidate_datasets.items():
        print(f"\n[{name}] shape={df.shape}")
        print(df[["source_dataset", "original_row_id"]].head(3))

    # ---- External validation dataset (UAE) — kept fully separate ----
    print("\n=== EXTERNAL VALIDATION DATASET (never merged) ===")
    for name, df in bundle.external_validation_dataset.items():
        print(f"\n[{name}] shape={df.shape}")
        print(df[["source_dataset", "original_row_id"]].head(3))

    # ---- Metadata & quality reports ----
    print("\n=== METADATA SUMMARY ===")
    print(json.dumps(
        {k: v.as_dict() for k, v in bundle.metadata.items()},
        indent=2, default=str,
    ))

    print("\n=== DATA QUALITY REPORTS ===")
    print(json.dumps(
        {k: v.as_dict() for k, v in bundle.quality_reports.items()},
        indent=2, default=str,
    ))

    # Sanity assertion: external validation data must never appear in
    # train_candidate_datasets, and vice versa.
    overlap = set(bundle.train_candidate_datasets) & set(bundle.external_validation_dataset)
    assert not overlap, f"Isolation violated! Overlapping keys: {overlap}"
    print("\nIsolation check passed: UAE external validation set is structurally separate.")


if __name__ == "__main__":
    main()