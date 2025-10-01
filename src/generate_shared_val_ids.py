#Let's diagnose each script:
#!/usr/bin/env python3
# generate_shared_val_ids.py
# --------------------------------------------------------------
# Generate a shared, stratified validation split across patients
# --------------------------------------------------------------

import argparse
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
VAL_IDS_PATH = Path(f"shared_val_ids_{args.metric_prefix}.npy")

# --------------------------------------------------------------
# Load eligible patient IDs from a *patient-level* DataFrame
# --------------------------------------------------------------
feat_path = Path("mimic_enriched_features_w_notes.csv")
if not feat_path.exists():
    raise FileNotFoundError("Missing mimic_enriched_features_w_notes.csv")

df = pd.read_csv(feat_path)
df = df.dropna(subset=["multiclass_label"])
df["multiclass_label"] = df["multiclass_label"].astype(int)

# Aggregate to patient-level (last visit per subject)
df_patient = df.sort_values("admittime").groupby("subject_id").last().reset_index()
eligible_ids = df_patient["subject_id"].values
labels = df_patient["multiclass_label"].values

# --------------------------------------------------------------
# Perform stratified split
# --------------------------------------------------------------
_, val_ids = train_test_split(
    eligible_ids,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)

# --------------------------------------------------------------
# Save
# --------------------------------------------------------------
np.save(VAL_IDS_PATH, val_ids)
print(f"✅ Saved shared validation IDs to {VAL_IDS_PATH} ({len(val_ids)} IDs)")