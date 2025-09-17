import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter

# ---------------------------------------------------------------------
# Set reproducible but flexible seed
# ---------------------------------------------------------------------
BASE_SEED = 42
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET
np.random.seed(SEED)

# ---------------------------------------------------------------------
# 1. Load boosted structured + note features
# ---------------------------------------------------------------------
print("🔹 Loading data...")
df = pd.read_csv("mimic_enriched_features_w_notes.csv")

# ---------------------------------------------------------------------
# 2. Feature Selection
# ---------------------------------------------------------------------
feature_cols = [
    "approx_age", "gender", "insurance_group", "admission_type", "length_of_stay",
    "was_in_icu", "seen_by_psych", "polypharmacy_flag", "diagnosis_count", "medication_count",
    "psych_or_pain_rx_count", "transfer_count", "note_count", "avg_note_length", "sentiment", "note_cluster"
] + [
    f"tfidf_{term}" for term in [
        'pain', 'anxiety', 'depression', 'headache', 'fatigue', 'sleep',
        'sad', 'crying', 'hopeless', 'tired', 'insomnia', 'nausea', 'vomiting'
    ]
] + [f"topic_{i+1}" for i in range(5)]

label_col = "multiclass_label"

# ---------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------
print("🔹 Preprocessing...")

df[feature_cols] = df[feature_cols].fillna(0)
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

# Encode categorical variables
df["gender"] = df["gender"].map({"M": 1, "F": 0}).fillna(-1)
df["insurance_group"] = df["insurance_group"].astype("category").cat.codes
df["admission_type"] = df["admission_type"].astype("category").cat.codes

binary_cols = ["was_in_icu", "seen_by_psych", "polypharmacy_flag"]
for col in binary_cols:
    df[col] = df[col].astype(int)

# ---------------------------------------------------------------------
# 4. Build visit-level sequences with masks
# ---------------------------------------------------------------------
print("🔹 Building sequences...")

SEQUENCE_LENGTH = 10
sequences, labels, masks = [], [], []

for subject_id, group in df.sort_values("admittime").groupby("subject_id"):
    visit_features = group[feature_cols].values
    label_sequence = group[label_col].values

    if len(visit_features) == 0:
        continue

    if len(visit_features) >= SEQUENCE_LENGTH:
        visit_features = visit_features[-SEQUENCE_LENGTH:]
        label = label_sequence[-1]
        mask = [1] * SEQUENCE_LENGTH
    else:
        pad_len = SEQUENCE_LENGTH - len(visit_features)
        visit_features = np.pad(visit_features, ((pad_len, 0), (0, 0)), mode='constant')  # Pad with zeros at the beginning
        mask = [0] * pad_len + [1] * len(visit_features[-(SEQUENCE_LENGTH - pad_len):])
        label = label_sequence[-1]

    sequences.append(visit_features)
    labels.append(label)
    masks.append(mask)

X = np.stack(sequences)
y = np.array(labels)
masks = np.array(masks)

# ---------------------------------------------------------------------
# 5. Train/test split + optional oversampling
# ---------------------------------------------------------------------
print("🔹 Splitting train/val...")

X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
    X, y, masks, stratify=y, test_size=0.2, random_state=SEED
)

# Diagnostic: check class coverage
def check_class_coverage(y, label):
    classes, counts = np.unique(y, return_counts=True)
    print(f"📦 {label}:", dict(zip(classes, counts)))
    missing = set([0, 1, 2]) - set(classes)
    if missing:
        print(f"⚠️ WARNING: {label} missing classes: {missing}")
    return missing

missing = check_class_coverage(y_train, "Train")
check_class_coverage(y_val, "Validation")

if missing:
    raise ValueError(f"🚨 Training data missing class(es): {missing}")

# Optional oversampling of co-morbid (class 2)
OVERSAMPLE_TARGET = int(os.getenv("OVERSAMPLE_TARGET", 100))
train_counts = Counter(y_train)
minority_class = 2
if train_counts[minority_class] < OVERSAMPLE_TARGET:
    print("⚠️ Oversampling class 2 (co-morbid)...")
    X_min = X_train[y_train == minority_class]
    y_min = y_train[y_train == minority_class]
    m_min = m_train[y_train == minority_class]

    n_samples = OVERSAMPLE_TARGET - len(y_min)
    X_upsampled, y_upsampled, m_upsampled = resample(
        X_min, y_min, m_min,
        replace=True, n_samples=n_samples, random_state=SEED
    )

    X_train = np.concatenate([X_train, X_upsampled])
    y_train = np.concatenate([y_train, y_upsampled])
    m_train = np.concatenate([m_train, m_upsampled])

    print("✅ Oversampling complete.")
    check_class_coverage(y_train, "Train (post-oversample)")

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
print("🔹 Saving sequences...")

out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X_train_transformer.npy", X_train)
np.save(f"{out_dir}/y_train_transformer.npy", y_train)
np.save(f"{out_dir}/X_val_transformer.npy", X_val)
np.save(f"{out_dir}/y_val_transformer.npy", y_val)
np.save(f"{out_dir}/mask_train_transformer.npy", m_train)
np.save(f"{out_dir}/mask_val_transformer.npy", m_val)

# Save feature columns for future reference
with open(f"{out_dir}/feature_cols_transformer.txt", "w") as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# ---------------------------------------------------------------------
# 7. Summary Report
# ---------------------------------------------------------------------
print("\n📋 Transformer Sequence Summary")
print("--------------------------------------------------")
print(f"🔢 Features per visit: {X.shape[2]}")
print(f"👩‍💻 Training samples: {X_train.shape[0]}")
print(f"👨‍💻 Validation samples: {X_val.shape[0]}")
print(f"🔎 Seed used: {SEED}")
print("✅ Saved Transformer-ready sequences with attention masks.")

