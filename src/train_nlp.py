from pathlib import Path
import json

import joblib
import pandas as pd
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Paths and configuration
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "humAID"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_FEATURES = 20000


# -----------------------------
# Load dataset splits
# -----------------------------

def find_split(keyword):
    matches = list(DATA_DIR.glob(f"*{keyword}*.parquet"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {keyword} Parquet file, found: {matches}"
        )

    return pd.read_parquet(matches[0])


train_df = find_split("train")
val_df = find_split("validation")
test_df = find_split("test")

required_columns = {"tweet_text", "class_label"}

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} split is missing required columns.")

    df.dropna(subset=["tweet_text", "class_label"], inplace=True)
    df["tweet_text"] = df["tweet_text"].astype(str)

print("Device:", DEVICE)
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))
print("Classes:", train_df["class_label"].nunique())


# -----------------------------
# Encode class labels
# -----------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_df["class_label"])
y_val = encoder.transform(val_df["class_label"])
y_test = encoder.transform(test_df["class_label"])

num_classes = len(encoder.classes_)


# -----------------------------
# Convert text to TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train = vectorizer.fit_transform(train_df["tweet_text"])
X_val = vectorizer.transform(val_df["tweet_text"])
X_test = vectorizer.transform(test_df["tweet_text"])

# PyTorch model uses dense tensors.
# Keep an eye on RAM if you increase MAX_FEATURES.
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_val = torch.tensor(X_val.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# -----------------------------
# Define neural network
# -----------------------------

class DisasterTextClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)


model = DisasterTextClassifier(
    input_size=X_train.shape[1],
    output_size=num_classes,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# -----------------------------
# Training and validation
# -----------------------------

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

    model.eval()

    with torch.no_grad():
        val_logits = model(X_val.to(DEVICE))
        val_loss = criterion(val_logits, y_val.to(DEVICE)).item()
        val_predictions = val_logits.argmax(dim=1).cpu()

    val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {total_loss / len(train_df):.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# -----------------------------
# Evaluate on held-out test set
# -----------------------------

if best_state is None:
    raise RuntimeError("Training did not produce a model checkpoint.")

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    test_logits = model(X_test.to(DEVICE))
    test_predictions = test_logits.argmax(dim=1).cpu().numpy()

test_labels = y_test.numpy()

print("\nTest Accuracy:", accuracy_score(test_labels, test_predictions))

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=list(range(num_classes)),
        target_names=encoder.classes_,
        zero_division=0,
    )
)


# -----------------------------
# Save model artifacts
# -----------------------------

torch.save(
    {
        "state_dict": model.cpu().state_dict(),
        "input_size": X_train.shape[1],
        "num_classes": num_classes,
    },
    MODEL_DIR / "humaid_text_model.pt",
)

joblib.dump(vectorizer, MODEL_DIR / "humaid_tfidf.joblib")
joblib.dump(encoder, MODEL_DIR / "humaid_labels.joblib")

with open(MODEL_DIR / "humaid_metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {"classes": encoder.classes_.tolist()},
        f,
        indent=2,
    )

print("\nSaved NLP model and supporting artifacts to:", MODEL_DIR)