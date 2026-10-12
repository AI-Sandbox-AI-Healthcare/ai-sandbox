"""tfidf_logreg_notes.py
Multiclass text-only baseline: TF-IDF vectoriser + LogisticRegression
on concatenated clinical notes.
Uses ResourceLogger for cost tracking and writes metrics.
"""

import os, csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from resource_logger import ResourceLogger
import argparse

# ------------------------------------------------------------------
# 0. Setup
# ------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
BASE = "../../analysis/data/derivedData"
Metrics_BASE = "../../analysis/results/metrics"
Plot_BASE = "../../analysis/results/figures/tfidf_logreg"

# Create directories if not present
os.makedirs(Metrics_BASE, exist_ok=True)
os.makedirs(Plot_BASE, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
args = parser.parse_args()
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")

# ------------------------------------------------------------------
# 1. Load note sequences per patient
# ------------------------------------------------------------------
notes_path = f"{BASE}/note_sequences_per_patient.npy"
notes_dict = np.load(notes_path, allow_pickle=True).item()

# Flatten notes → single doc per patient
texts = {}
for subj, adm_lists in notes_dict.items():
    all_notes = " ".join([" ".join(notes) for notes in adm_lists])
    texts[subj] = all_notes

# ------------------------------------------------------------------
# 2. Load labels (binary) aggregated per patient
# ------------------------------------------------------------------
feat = pd.read_csv(
    f"{BASE}/synthea_enriched_features.csv",
    usecols=["id", "binary_label"]
)
labels = feat.groupby("id")['binary_label'].max().astype(int)

# Keep only patients with notes
subj_ids = sorted(set(texts.keys()) & set(labels.index))
corpus   = [texts[s] for s in subj_ids]
y        = labels.loc[subj_ids].values
subj_ids = np.array(subj_ids)

print(f"Patients with notes & valid label: {len(subj_ids)}")
print("Class distribution:", np.bincount(y))

# ------------------------------------------------------------------
# 3. Shared train/validation/test split
# ------------------------------------------------------------------
train_ids = np.load(f"{BASE}/shared_train_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
val_ids = np.load(f"{BASE}/shared_val_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
test_ids = np.load(f"{BASE}/shared_test_ids_{METRIC_PREFIX}.npy", allow_pickle=True)
is_train = np.isin(subj_ids, train_ids)
is_val = np.isin(subj_ids, val_ids)
is_test = np.isin(subj_ids, test_ids)

train_idx = np.where(is_train)[0]
val_idx = np.where(is_val)[0]
test_idx = np.where(is_test)[0]

X_train_txt = [corpus[i] for i in train_idx]
y_train     = y[train_idx]
X_val_txt = [corpus[i] for i in val_idx]
y_val = y[val_idx]
X_test_txt = [corpus[i] for i in test_idx]
y_test = y[test_idx]
subj_test = subj_ids[test_idx]

# ------------------------------------------------------------------
# 4. Vectorise + train
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(X_train_txt)
X_val = vectorizer.transform(X_val_txt)
X_test = vectorizer.transform(X_test_txt)

best_auc = -np.inf
best_c = None
logreg = None
with ResourceLogger(tag=f"tfidf_logreg_notes_{METRIC_PREFIX}"):
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        candidate = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            random_state=SEED,
            C=c_value
        )
        candidate.fit(X_train, y_train)
        val_prob = candidate.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"Validation AUC for C={c_value}: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_c = c_value
            logreg = candidate

    if logreg is None:
        raise RuntimeError("TF-IDF validation did not select a LogisticRegression model.")

    print(f"Selected C={best_c} using validation AUC={best_auc:.4f}")
    prob = logreg.predict_proba(X_test)
    preds = logreg.predict(X_test)

# ------------------------------------------------------------------
# 5. Save probabilities for stacking
# ------------------------------------------------------------------
np.savez_compressed(
    f"{BASE}/tfidf_probs_{METRIC_PREFIX}.npz",
    probs=prob,
    y_true=y_test,
    subject_ids=subj_test
)

# ------------------------------------------------------------------
# 6. Metrics & save
# ------------------------------------------------------------------
accuracy = accuracy_score(y_test, preds)

prob_pos = prob[:, 1]  # select positive-class probability

roc_auc  = roc_auc_score(y_test, prob_pos)
report   = classification_report(y_test, preds, output_dict=True, zero_division=0)

METRIC_CSV = f"{Metrics_BASE}/tfidf_metrics_{METRIC_PREFIX}.csv"
with open(METRIC_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric","Precision","Recall","F1"])
    for cls in ["0","1"]:
        row = report.get(cls, {"precision":0,"recall":0,"f1-score":0})
        writer.writerow([cls,row["precision"],row["recall"],row["f1-score"]])
    writer.writerow(["Accuracy", accuracy, "", ""])
    writer.writerow(["ROC_AUC", roc_auc, "", ""])

print(f"\nMetrics saved to {METRIC_CSV}")
print(f"Accuracy: {accuracy:.4f} | AUC: {roc_auc:.4f}")
print(classification_report(y_test, preds, digits=4, zero_division=0))

# ------------------------------------------------------------------
# 7. ROC Curves
# ------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, prob_pos)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("TF-IDF Logistic Regression ROC (Binary)")
plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(f"{Plot_BASE}/tfidf_roc_curve_{METRIC_PREFIX}.png")
plt.close()

# ------------------------------------------------------------------
# 8. Confusion Matrix
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("TF-IDF LogReg Confusion Matrix (Binary)")
plt.tight_layout()
plt.savefig(f"{Plot_BASE}/tfidf_confusion_{METRIC_PREFIX}.png")
plt.close()
