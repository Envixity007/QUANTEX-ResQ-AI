from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "humAID"

for file in DATA_DIR.glob("*.parquet"):
    print(f"\n{'=' * 50}")
    print(f"FILE: {file.name}")

    df = pd.read_parquet(file)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nData types:")
    print(df.dtypes)