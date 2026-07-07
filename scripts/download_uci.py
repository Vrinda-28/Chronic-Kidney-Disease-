from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    print("Downloading UCI CKD dataset...")

    ckd = fetch_ucirepo(id=336)

    X = ckd.data.features
    y = ckd.data.targets

    df = pd.concat([X, y], axis=1)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "uci_ckd.csv"

    df.to_csv(output_file, index=False)

    print("Done!")
    print("Saved to:", output_file)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()