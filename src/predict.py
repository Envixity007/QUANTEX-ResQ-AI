import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/severity_model.joblib")
LABEL_PATH = Path("models/severity_labels.joblib")
DATA_PATH = Path("data/processed/ibm final data.xlsx")

# Load saved model and labels
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)

# Load a sample record
df = pd.read_excel(DATA_PATH)
sample = df.drop(columns=[
    "severity",
    "event_id",
    "event_name",
    "deaths",
    "injured",
    "total_affected",
    "damage_usd_thousand",
    "injured_missing",
    "total_affected_missing",
], errors="ignore").iloc[[0]].copy()

# Match training-time date conversion
for col in sample.columns:
    if pd.api.types.is_datetime64_any_dtype(sample[col]):
        sample[col] = pd.to_datetime(sample[col]).map(
            lambda x: x.timestamp() if pd.notna(x) else float("nan")
        )

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
severity = label_encoder.inverse_transform([prediction])[0]

print("Predicted severity:", severity)
print("\nClass probabilities:")

for label, probability in zip(
    label_encoder.classes_, probabilities
):
    print(f"{label}: {probability:.1%}")