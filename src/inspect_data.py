import pandas as pd
from pathlib import Path

file_path = Path("data/processed/ibm final data.xlsx")

df = pd.read_excel(file_path)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head().to_string())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)