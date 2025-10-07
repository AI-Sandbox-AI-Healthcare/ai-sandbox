"""tabular_logreg_baseline.py
Fast logistic‑regression baseline on structured features with resource logging.
"""
import os, csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from resource_logger import ResourceLogger

# ------------------------------------------------------------------
# 0. Paths & reproducibility
# ------------------------------------------------------------------
SEED = 42
BASE = "./"
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. Load + preprocess
# ------------------------------------------------------------------
df = pd.read_csv(f"{BASE}/mimic_enriched_features.csv")
df = df.dropna(subset=["approx_age", "gender"])
df["gender"] = df["gender"].map({"M":1,"F":0})
df["insurance"] = df["insurance"].astype("category").cat.codes
df["admission_type"] = df["admission_type"].astype("category").cat.codes
df["length_of_stay"].fillna(df["length_of_stay"].median(), inplace=True)
for b in ["medication_count","was_in_icu"]:
    df[b].fillna(0, inplace=True)
for b in ["seen_by_psych","on_psych_or_pain_meds","was_in_icu"]:
    df[b] = df[b].astype(int)

FEATURES=["approx_age","gender","insurance","admission_type","length_of_stay","was_in_icu","seen_by_psych","on_psych_or_pain_meds","diagnosis_count","medication_count"]
X=df[FEATURES]
y=df["co_occurrence"].astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2,random_state=SEED)
scaler=StandardScaler(); X_train=scaler.fit_transform(X_train); X_test=scaler.transform(X_test)

# ------------------------------------------------------------------
# 2. Train + evaluate
# ------------------------------------------------------------------
with ResourceLogger(tag="tabular_logreg"):
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    logreg.fit(X_train, y_train)
    prob = logreg.predict_proba(X_test)[:,1]
    preds = (prob>0.5).astype(int)
    np.save("logreg_probs.npy", prob)

# Metrics
roc_auc = roc_auc_score(y_test, prob)
accuracy = (preds==y_test.values).mean()

METRIC_CSV=f"{BASE}/logreg_metrics.csv"
with open(METRIC_CSV,"w",newline="") as f:
    csv.writer(f).writerows([[
        "Metric", "Value"
    ],[
        "AUC", f"{roc_auc:.4f}"
    ],[
        "Accuracy", f"{accuracy:.4f}"
    ]])

print(f"Metrics → {METRIC_CSV}")
print("\n📊 Final Logistic Regression Evaluation:")
print(f"  AUC:      {roc_auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, preds, zero_division=0))

# ROC
fpr,tpr,_=roc_curve(y_test, prob)
plt.figure()
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Logistic Regression ROC")
plt.legend()
plt.grid()
plt.show()
