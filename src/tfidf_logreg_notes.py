"""tfidf_logreg_notes.py
Lightweight text‑only baseline: TF‑IDF vectoriser + LogisticRegression on concatenated clinical notes.
Uses ResourceLogger for cost tracking and writes metrics.
"""
import os, csv, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from resource_logger import ResourceLogger

SEED = 42
np.random.seed(SEED)
BASE = "./"

# ------------------------------------------------------------------
# 1. Load note sequences per patient
# ------------------------------------------------------------------
notes_path = f"{BASE}/note_sequences_per_patient.npy"
notes_dict = np.load(notes_path, allow_pickle=True).item()

# Flatten notes -> single doc per patient
texts = {}
for subj, adm_lists in notes_dict.items():
    all_notes = " ".join([" ".join(notes) for notes in adm_lists])
    texts[subj] = all_notes

# ------------------------------------------------------------------
# 2. Load labels (co_occurrence) aggregated per patient
# ------------------------------------------------------------------
feat = pd.read_csv(f"{BASE}/mimic_enriched_features.csv", usecols=["subject_id","co_occurrence"])
labels = feat.groupby("subject_id")['co_occurrence'].max().astype(int)

# Keep only patients with text
subj_ids = list(set(texts.keys()) & set(labels.index))
corpus   = [texts[s] for s in subj_ids]
y        = labels.loc[subj_ids].values

print(f"Patients with notes & label: {len(subj_ids)}")

# ------------------------------------------------------------------
# 3. Split, vectorise, train
# ------------------------------------------------------------------
X_train_txt, X_test_txt, y_train, y_test = train_test_split(corpus, y, stratify=y, test_size=0.2, random_state=SEED)

vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
X_train = vectorizer.fit_transform(X_train_txt)
X_test  = vectorizer.transform(X_test_txt)

with ResourceLogger(tag="tfidf_logreg_notes"):
    logreg = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    logreg.fit(X_train, y_train)
    prob = logreg.predict_proba(X_test)[:,1]
    preds = (prob>0.5).astype(int)
    np.save("tfidf_probs.npy", prob)

# ------------------------------------------------------------------
# 4. Metrics & save
# ------------------------------------------------------------------
roc_auc = roc_auc_score(y_test, prob)
accuracy = (preds==y_test).mean()

METRIC_CSV = f"{BASE}/tfidf_metrics.csv"
with open(METRIC_CSV, "w", newline="") as f:
    csv.writer(f).writerows([[
        "Metric", "Value"
    ],[
        "AUC", f"{roc_auc:.4f}"
    ],[
        "Accuracy", f"{accuracy:.4f}"
    ]])

print(f"Metrics → {METRIC_CSV}")
print("\n📊 Final TF-IDF + Logistic Regression Evaluation:")
print(f"  AUC:      {roc_auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, preds, zero_division=0))

# ROC
fpr,tpr,_ = roc_curve(y_test, prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("TF‑IDF LogReg ROC")
plt.legend()
plt.grid()
plt.show()