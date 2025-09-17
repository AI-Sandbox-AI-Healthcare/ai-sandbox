"""train_gru_mimic.py
GRU baseline on structured sequences with resource logging and metrics.
"""
import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from resource_logger import ResourceLogger
from tqdm import tqdm
import argparse

# Base seed
BASE_SEED = 42
# Get iteration offset from environment (defaults to 0)
OFFSET = int(os.getenv("SEED_OFFSET", 0))
SEED = BASE_SEED + OFFSET

np.random.seed(SEED)
torch.manual_seed(SEED)

# Dataset
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), torch.tensor(int(self.y[i]), dtype=torch.long)

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=1, dropout=0.3, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

BASE = "./"
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default=None)
args = parser.parse_args()

# Allow fallback to environment variable
METRIC_PREFIX = args.metric_prefix or os.getenv("METRIC_PREFIX", "iter1")
X_tr = np.load(f"{BASE}/X_train_seq.npy")
y_tr = np.load(f"{BASE}/y_train_seq.npy")
X_val = np.load(f"{BASE}/X_val_seq.npy")
y_val = np.load(f"{BASE}/y_val_seq.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute class weights to address imbalance
class_counts = np.bincount(y_tr)
class_weights = 1.0 / class_counts
norm_weights = class_weights / class_weights.sum()
weight_tensor = torch.tensor(norm_weights, dtype=torch.float32).to(device)

train_loader = DataLoader(SeqDS(X_tr, y_tr), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(SeqDS(X_val, y_val), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

model = GRUClassifier(input_dim=X_tr.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss(weight=weight_tensor)

BEST = f"{BASE}/gru_model_{METRIC_PREFIX}.pt"
METRIC = f"{BASE}/gru_multiclass_metrics_{METRIC_PREFIX}.csv"

with ResourceLogger(tag=f"gru_multiclass_{METRIC_PREFIX}"):
    best = float('inf'); pat = 3; cnt = 0
    for ep in range(15):
        model.train(); tl = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep+1:02d} Training"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb); loss = crit(logits, yb); loss.backward(); opt.step(); tl += loss.item()
        print(f"Ep{ep+1:02d} Train {tl/len(train_loader):.4f}")

        model.eval(); vl = 0; preds = []; trues = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Final Evaluation"):
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                vl += crit(logit, yb).item()
                preds.extend(torch.argmax(logit, dim=1).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        vl /= len(val_loader)
        print(f"Ep{ep+1:02d} Val {vl:.4f}")
        if vl < best:
            best = vl; cnt = 0; torch.save(model.state_dict(), BEST); print("  ✓ checkpoint saved")
        else:
            cnt += 1
            if cnt >= pat: print("Early stopping"); break

# eval
model.load_state_dict(torch.load(BEST)); model.eval(); preds = []; trues = []
with torch.no_grad():
    for xb, yb in tqdm(val_loader, desc=f"Epoch {ep+1:02d} Validation"):
        xb = xb.to(device)
        log = model(xb)
        preds.extend(torch.argmax(log, dim=1).cpu().numpy())
        trues.extend(yb.numpy())

acc = accuracy_score(trues, preds)
report = classification_report(trues, preds, zero_division=0, output_dict=True)

with open(METRIC, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-score"])
    for cls in sorted(report.keys()):
        if cls not in ["accuracy", "macro avg", "weighted avg"]:
            row = report[cls]
            writer.writerow([cls, f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1-score']:.4f}"])

print(f"Metrics → {METRIC}")
print("\n📊 Final GRU Multiclass Evaluation:")
print(classification_report(trues, preds, zero_division=0))
print("Class Distribution:", dict(zip(*np.unique(y_tr, return_counts=True))))

cm = confusion_matrix(trues, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - GRU Multiclass")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Save softmax probabilities for stacking
# ---------------------------------------------------------------------
subject_ids = np.load(f"{BASE}/subject_ids_val_seq.npy")  # <-- must match split

probs = []
with torch.no_grad():
    for xb, _ in tqdm(val_loader, desc=f"Saving Probs {METRIC_PREFIX}"):
        xb = xb.to(device)
        logits = model(xb)
        prob_batch = torch.softmax(logits, dim=1)
        probs.extend(prob_batch.cpu().numpy())

np.savez_compressed(
    f"{BASE}/gru_probs_{METRIC_PREFIX}.npz",
    probs=np.array(probs),
    y_true=y_val,
    subject_ids=subject_ids
)
print(f"Saved GRU probs → gru_probs_{METRIC_PREFIX}.npz")