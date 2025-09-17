# wilcoxon_test.py
import pandas as pd
from scipy.stats import wilcoxon

# Load the full summary
df = pd.read_csv("results_summary_all_iterations.csv")

# Separate models
lstm_scores = df[df["model"] == "train_lstm_mimic"]["macro_f1"]
transformer_scores = df[df["model"] == "train_transformer_mimic"]["macro_f1"]

# Sanity check
assert len(lstm_scores) == len(transformer_scores), "Mismatch in number of runs!"

# Wilcoxon signed-rank test
stat, p = wilcoxon(lstm_scores, transformer_scores, alternative="two-sided")

print("\n📊 Wilcoxon Signed-Rank Test")
print("--------------------------------------------------")
print(f"Statistic = {stat:.4f}")
print(f"P-value   = {p:.4e}")
if p < 0.05:
    print("✅ Difference is statistically significant!")
else:
    print("⚠️ No statistically significant difference detected.")
