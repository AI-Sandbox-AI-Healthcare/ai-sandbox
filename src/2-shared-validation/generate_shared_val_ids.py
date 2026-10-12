#!/usr/bin/env python3
# generate_shared_val_ids.py
# --------------------------------------------------------------
# Generate shared, stratified train/validation/test splits across patients
# --------------------------------------------------------------

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# CLI arguments
# --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed_offset", type=int, default=0)
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
BASE_SEED = 42
SEED = BASE_SEED + args.seed_offset
np.random.seed(SEED)
SPLIT_DIR = Path("../../analysis/data/derivedData")
SPLIT_PREFIX = args.metric_prefix

# --------------------------------------------------------------
# Load patient-level data
# --------------------------------------------------------------
feat_path = Path("../../analysis/data/derivedData/synthea_enriched_features_w_notes.csv")
if not feat_path.exists():
    raise FileNotFoundError("Missing ../../analysis/data/derivedData/synthea_enriched_features_w_notes.csv")

df = pd.read_csv(feat_path)

# Clean labels
df = df.dropna(subset=["binary_label"])
df["binary_label"] = df["binary_label"].astype(int)

# Remove class -1 (neither MH nor pain)
df_patient = df[df["binary_label"].isin([0, 1])].copy()
df_patient["id"] = df_patient["id"].astype(str)
df_patient = df_patient.sort_values("id").drop_duplicates("id")

eligible_ids = df_patient["id"].values
labels = df_patient["binary_label"].values

# --------------------------------------------------------------
# Split off a stratified 15% test set, then 15% validation from
# the remaining 85%, leaving 70% for training.
# --------------------------------------------------------------
train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
    eligible_ids,
    labels,
    test_size=0.15,
    stratify=labels,
    random_state=SEED
)
train_ids, val_ids, train_labels, val_labels = train_test_split(
    train_val_ids,
    train_val_labels,
    test_size=0.15 / 0.85,
    stratify=train_val_labels,
    random_state=SEED
)

# --------------------------------------------------------------
# Verify and save
# --------------------------------------------------------------
split_ids = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids,
}
split_labels = {
    "train": train_labels,
    "val": val_labels,
    "test": test_labels,
}

for split_name, ids in split_ids.items():
    path = SPLIT_DIR / f"shared_{split_name}_ids_{SPLIT_PREFIX}.npy"
    np.save(path, ids)
    print(f"✅ Saved shared {split_name} IDs to {path} ({len(ids)} IDs)")

all_split_ids = np.concatenate(list(split_ids.values()))
if len(all_split_ids) != len(eligible_ids) or len(np.unique(all_split_ids)) != len(eligible_ids):
    raise RuntimeError("Generated split IDs are not a disjoint partition of eligible patients.")
metadata_path = SPLIT_DIR / f"shared_split_metadata_{SPLIT_PREFIX}.json"
metadata_path.write_text(json.dumps({
    "seed": SEED,
    "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "patient_counts": {name: len(ids) for name, ids in split_ids.items()},
    "class_counts": {
        name: {str(cls): int((split_y == cls).sum()) for cls in np.unique(labels)}
        for name, split_y in split_labels.items()
    },
}, indent=2) + "\n")
print(f"✅ Saved split metadata to {metadata_path}")

# --------------------------------------------------------------
# Diagnostics: split sizes and class distribution
# --------------------------------------------------------------
for split_name, split_y in split_labels.items():
    counts = pd.Series(split_y).value_counts().sort_index()
    print(f"📊 {split_name.title()}: {len(split_y)} patients ({len(split_y) / len(eligible_ids):.1%})")
    for cls, count in counts.items():
        print(f"  Class {cls}: {count} ({count / len(split_y):.1%})")
