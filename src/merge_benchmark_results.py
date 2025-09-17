# Merge results
echo "=== [Final] Merging all iteration results =========================="
python3 - <<'PY'
import glob
import pandas as pd
import os
from collections import defaultdict

csvs = glob.glob("*_metrics_iter*.csv")
if not csvs:
    print("⚠️ No iteration metric CSVs found. Skipping merge.")
    exit(1)

grouped = defaultdict(list)
for path in csvs:
    base = os.path.basename(path)
    parts = base.split("_iter")
    if len(parts) == 2:
        tag = "iter" + parts[1].replace(".csv", "")
        grouped[tag].append((parts[0], path))

all_summaries = []
for tag, files in grouped.items():
    dfs = []
    for model_name, path in files:
        df = pd.read_csv(path)
        df.insert(0, "model", model_name)
        df.insert(0, "iteration", tag)
        dfs.append(df)
    df_iter = pd.concat(dfs, ignore_index=True)
    df_iter.to_csv(f"results_summary_{tag}.csv", index=False)
    all_summaries.append(df_iter)

df_all = pd.concat(all_summaries, ignore_index=True)
df_all.to_csv("results_summary_all_iterations.csv", index=False)
print("✅ Saved → results_summary_all_iterations.csv")
PY

represented in this:
# merge_all_metrics.py
import pandas as pd
import glob
import os

BASE = "./"
output_file = os.path.join(BASE, "results_summary_all_iterations.csv")

# Detect all relevant *_metrics_iter*.csv files
metric_files = sorted(glob.glob(os.path.join(BASE, "*_metrics_iter*.csv")))

if not metric_files:
    print("❌ No metrics files found. Please check the directory and filenames.")
    exit(1)

combined = []

for file in metric_files:
    filename = os.path.basename(file)
    parts = filename.split("_metrics_")
    if len(parts) != 2:
        continue
    model = parts[0]
    run = parts[1].replace(".csv", "")

    df = pd.read_csv(file)

    # Case: AUC + Accuracy table (tabular summary)
    if list(df.columns) == ["Model", "AUC", "Accuracy"]:
        df.insert(0, "run", run)
        df.insert(0, "model", model)
        df["Class"] = ""
        df["Precision"] = ""
        df["Recall"] = ""
        df["F1-score"] = ""
    else:
        df.insert(0, "run", run)
        df.insert(0, "model", model)
        df["Model"] = ""
        df["AUC"] = ""
        df["Accuracy"] = ""

    combined.append(df)

# Concatenate all and save
results_df = pd.concat(combined, ignore_index=True)
results_df.to_csv(output_file, index=False)
print(f"✅ Saved combined metrics → {output_file}")