"""train_lstm_mimic.py
Clean LSTM baseline that
1. Loads structured sequence arrays
2. Trains LSTM with early stopping
3. Logs resources via ResourceLogger
4. Saves metrics + ROC curve for benchmarking
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from resource_logger import ResourceLogger

# ---------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# 1. Dataset & Model
# ---------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, seq, labels):
        self.X, self.y = seq, labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()  # logits

# ---------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------
BASE = "./"
X_train = np.load(f"{BASE}/X_train_seq.npy")
y_train = np.load(f"{BASE}/y_train_seq.npy")
X_val   = np.load(f"{BASE}/X_val_seq.npy")
y_val   = np.load(f"{BASE}/y_val_seq.npy")

train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(SequenceDataset(X_val,   y_val),   batch_size=32, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------
# 3. Model / Optim / Loss
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pos_weight = torch.tensor([len(y_train)/y_train.sum() - 1]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

BEST_PATH = f"{BASE}/mimic_lstm_model.pt"
METRIC_CSV = f"{BASE}/lstm_metrics.csv"

# ---------------------------------------------------------------------
# 4. Train + validate with ResourceLogger
# ---------------------------------------------------------------------
with ResourceLogger(tag="lstm"):
    best_val, patience, counter = float("inf"), 3, 0
    for epoch in range(15):
        # --- Train ---
        model.train(); train_loss=0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | TrainLoss {train_loss/len(train_loader):.4f}")

        # --- Validate ---
        model.eval(); val_loss=0; probs=[]; trues=[]
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item()
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                trues.extend(yb.cpu().numpy())
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:02d} | ValLoss  {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss; counter=0
            torch.save(model.state_dict(), BEST_PATH)
            print("  ✓ checkpoint saved")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    
    np.save("lstm_probs.npy", probs)  
    np.save("lstm_y_true.npy", trues)
    
# ---------------------------------------------------------------------
# 5. Final evaluation
# ---------------------------------------------------------------------
model.load_state_dict(torch.load(BEST_PATH)); model.eval()
probs, trues = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.numpy())
probs = np.array(probs); trues = trues = np.array(trues).astype(int)
preds = (probs>0.5).astype(int)
roc_auc = roc_auc_score(trues, probs)
accuracy= (preds==trues).mean()

with open(METRIC_CSV, "w", newline="") as f:
    csv.writer(f).writerows([["Metric", "Value"], ["AUC", f"{roc_auc:.4f}"], ["Accuracy", f"{accuracy:.4f}"]])

print(f"Metrics → {METRIC_CSV}")
print("\n📊 Final LSTM Evaluation:")
print(f"  AUC:      {roc_auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(trues, preds, zero_division=0))


fpr, tpr, _ = roc_curve(trues, probs)
plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}"); plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("LSTM ROC"); plt.legend(); plt.grid(); plt.show()
