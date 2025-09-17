"""train_transformer_mimic.py
Transformer baseline on structured sequences with resource logging and metrics.
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from resource_logger import ResourceLogger

# ------------------------------------------------------------------
# 0. Reproducibility
# ------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------
# 1. Dataset & Model
# ------------------------------------------------------------------
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

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim=64, heads=4, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, model_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(model_dim, 1)
    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze()  # logits

# ------------------------------------------------------------------
# 2. Data loading
# ------------------------------------------------------------------
BASE = "./"
X_train = np.load(f"{BASE}/X_train_seq.npy")
y_train = np.load(f"{BASE}/y_train_seq.npy")
X_val   = np.load(f"{BASE}/X_val_seq.npy")
y_val   = np.load(f"{BASE}/y_val_seq.npy")

train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(SequenceDataset(X_val,   y_val),   batch_size=32, shuffle=False, num_workers=2)

# ------------------------------------------------------------------
# 3. Model / loss / opt
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pos_weight = torch.tensor([len(y_train)/y_train.sum() - 1]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

BEST_PATH   = f"{BASE}/mimic_transformer_model.pt"
METRIC_CSV  = f"{BASE}/transformer_metrics.csv"

# ------------------------------------------------------------------
# 4. Train + validate with ResourceLogger
# ------------------------------------------------------------------
with ResourceLogger(tag="transformer"):
    best_val, patience, counter = float('inf'), 3, 0
    for epoch in range(20):
        # --- train ---
        model.train(); tr_loss=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | TrainLoss {tr_loss/len(train_loader):.4f}")

        # --- val ---
        model.eval(); val_loss=0; probs=[]; trues=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
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
    np.save("transformer_probs.npy", probs)

# ------------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------------
model.load_state_dict(torch.load(BEST_PATH)); model.eval()
probs, trues = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        trues.extend(yb.numpy())
probs = np.array(probs); trues = np.array(trues).astype(int)
preds = (probs>0.5).astype(int)
roc_auc = roc_auc_score(trues, probs)
accuracy= (preds==trues).mean()

with open(METRIC_CSV, "w", newline="") as f:
    csv.writer(f).writerows([
        ["Metric","Value"],
        ["AUC",f"{roc_auc:.4f}"],
        ["Accuracy",f"{accuracy:.4f}"]
    ])
print(f"Metrics → {METRIC_CSV}")
print("\n📊 Final Transformer Evaluation:")
print(f"  AUC:      {roc_auc:.4f}")
print(f"  Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(trues, preds, zero_division=0))

# ROC
fpr,tpr,_ = roc_curve(trues, probs)
plt.figure(); plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}"); plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Transformer ROC"); plt.legend(); plt.grid(); plt.show()