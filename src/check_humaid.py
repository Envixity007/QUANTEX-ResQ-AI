from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "humAID"

for file in sorted(DATA_DIR.glob("*.parquet")):
    df = pd.read_parquet(file)

    print(f"\n{'=' * 60}")
    print(f"FILE: {file.name}")
    print(f"Total samples: {len(df)}")

    print("\nClass distribution:")
    print(df["class_label"].value_counts().to_string())

    print("\nMissing values:")
    print(df[["tweet_text", "class_label"]].isna().sum())

    print("\nUnique classes:", df["class_label"].nunique())
    print("Classes:", sorted(df["class_label"].dropna().unique()))