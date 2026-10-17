import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Paths
DATA_PATH = Path("data/processed/ibm final data.xlsx")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nSeverity distribution:")
print(df["severity"].value_counts())

# Remove rows without target labels
df = df.dropna(subset=["severity"]).copy()

# Remove IDs and direct target leakage.
# The outcome fields may have been used to assign severity,
# so we exclude them from the input features.
drop_columns = [
    "severity",
    "event_id",
    "event_name",
    "deaths",
    "injured",
    "total_affected",
    "damage_usd_thousand",
    "injured_missing",
    "total_affected_missing",
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["severity"].astype(str).str.strip()

# Convert date columns into numerical features
for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col] = pd.to_datetime(X[col]).map(
            lambda x: x.timestamp() if pd.notna(x) else np.nan
        )

# Handle any remaining non-numeric columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )),
])

# Stratify so each split retains class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded,
)

# Train
print("\nTraining XGBoost model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0,
))

# Save model and label encoder
joblib.dump(model, MODEL_DIR / "severity_model.joblib")
joblib.dump(label_encoder, MODEL_DIR / "severity_labels.joblib")

print("\nModel saved to models/")