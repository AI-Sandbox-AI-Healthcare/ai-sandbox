# AI-Sandbox

## Overview
This project explores AI Sandboxes as a unified framework for AI education. The AI Sandbox is designed as a modular, reproducible, and policy-aware learning environment where students can safely experiment with complete AI workflows while maintaining scientific rigor, transparency, and consistency.

Building on the AI Sandbox developed through the awarded NAIRR Classroom Pilot `NAIRR250184`, this research examines how sandbox-based learning can improve AI education by combining infrastructure, pedagogy, reproducibility, and governance into one integrated model. The repository supports that goal through an end-to-end healthcare AI workflow built around Synthea-based data processing, feature engineering, benchmarking, and model training.

## Run Code Instructions
Use the commands below to set up the environment and run the pipeline from the project's `src/` directory. The workflow begins with environment setup, continues through preprocessing and feature generation, then runs the full modeling pipeline and repeated benchmark iterations.

### 1. Create and activate a virtual environment

```bash
sudo apt install python3-venv
python3 -m venv myenv
source ~/Desktop/myenv/bin/activate
```

### 2. Move into the source directory

```bash
cd ~/ai-sandbox/src
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run preprocessing and feature-building scripts

```bash
python3 process_noteevents_text.py
python3 synthea_extract.py
python3 boosted_synthea_text_features.py
```

These scripts must be run from inside `src/1-pre-processing/`. They load raw Synthea CSVs, build the patient-level feature dataset, aggregate medication text into per-patient note sequences, and generate additional text-derived features (TF-IDF, sentiment, LDA topics, PCA/UMAP embeddings) used by all downstream models.

### 5. Run the full model pipeline

```bash
chmod +x run_all_models.sh
./run_all_models.sh
```

This script must be run from inside `src/shell-scripts/`. It runs the end-to-end modeling workflow: shared validation split generation, visit sequence building, CPU baseline training (Random Forest, XGBoost, TF-IDF LogReg), GPU model training (LSTM, GRU, Transformer, ClinicalBERT), stacking meta-learner evaluation, and timing/summary logs for one iteration.

### 6. Run benchmark iterations

```bash
chmod +x run_benchmark_iterations.sh
./run_benchmark_iterations.sh
```

This script repeats the full pipeline across multiple iterations with different random seeds, resumes from the last completed iteration when possible, runs post-processing (merge metrics, Wilcoxon tests, MLflow logging), organizes output artifacts, and regenerates the project README.

## File Organization

```text
.
|-- README.md                              # project overview and documentation
|-- .gitignore                             # git ignore rules
|-- analysis/                             # generated artifacts, experiment outputs, and logs
|   |-- cache/                            # cached intermediate files (tokenized notes, pycache)
|   |-- data/
|   |   |-- rawData/                      # source Synthea CSV datasets
|   |   `-- derivedData/                  # processed arrays and features created by the pipeline
|   |-- experiments/                      # MLflow runs and CatBoost training logs
|   |-- logs/                             # per-iteration timing logs and completion markers
|   |   `-- resource_usage.csv            # runtime and resource tracking output
|   |-- models/                           # saved model artifacts (.pt, .pkl files)
|   `-- results/
|       |-- figures/                      # generated plots (ROC curves, confusion matrices, SHAP)
|       |-- metrics/                      # per-iteration and global benchmark CSVs
|       `-- model_cards/                  # model documentation markdown files
`-- src/                                  # all source code, organized by pipeline stage
    |-- 1-pre-processing/                 # raw data ingestion and feature engineering
    |   |-- synthea_extract.py            # loads Synthea CSVs, engineers structured patient features
    |   |-- process_noteevents_text.py    # builds per-patient medication note sequences
    |   |-- boosted_synthea_text_features.py  # adds TF-IDF, sentiment, LDA, PCA/UMAP features
    |   `-- clinicalbert_tokenize_notes.py    # parallel tokenization of notes for ClinicalBERT
    |-- 2-shared-validation/              # reproducible train/validation split
    |   `-- generate_shared_val_ids.py    # stratified patient split shared across all models
    |-- 3-visit-sequences/                # sequential feature arrays for sequence models
    |   |-- lstm_sequences.py             # builds padded sequences for LSTM/GRU
    |   `-- transformer_sequences.py      # builds masked sequences for Transformer/ClinicalBERT
    |-- 4-cpu-baselines/                  # tabular and text CPU-only baselines
    |   |-- synthea_classification.py     # Random Forest and XGBoost with resampling search
    |   `-- tfidf_logreg_notes.py         # TF-IDF + Logistic Regression on note text
    |-- 5-gpu-baselines/                  # GPU sequence and BERT models
    |   |-- train_lstm_synthea.py         # binary LSTM classifier
    |   |-- train_gru_synthea.py          # binary GRU classifier
    |   |-- train_transformer_synthea.py  # Transformer encoder classifier
    |   |-- precompute_bert_embeddings.py # precomputes ClinicalBERT CLS embeddings
    |   |-- clinicalbert_training.py      # ClinicalBERT + structured fusion model
    |   |-- clinicalbert_dataset.py       # PyTorch dataset classes for ClinicalBERT
    |   `-- clinicalbert_model.py         # ClinicalBERT model architecture definition
    |-- 6-stacking-meta-learner/          # ensemble stacking over base model outputs
    |   |-- stacking_meta_learner.py      # evaluates candidate meta-learners, selects best
    |   `-- best_stacking_meta_learner_across_iterations.py  # picks best model across all runs
    |-- 7-benchmark-iterations/           # post-run aggregation and statistical tests
    |   |-- merge_benchmark_results.py    # merges per-iteration metric CSVs into summaries
    |   |-- wilcoxon_test.py              # pairwise Wilcoxon signed-rank tests across models
    |   `-- summarize_benchmark.py        # final performance and resource usage summary
    |-- 8-resource-logger/                # shared runtime instrumentation
    |   `-- resource_logger.py            # context manager logging wall time, CPU, GPU, disk
    |-- shell-scripts/                    # orchestration scripts
    |   |-- run_all_models.sh             # end-to-end pipeline for a single iteration
    |   |-- run_benchmark_iterations.sh   # runs N iterations, resumes on failure
    |   |-- run_summarize_benchmarks.sh   # timing summary and MLflow logging
    |   |-- organize_artifacts.sh         # moves plots and CSVs into standard output folders
    |   `-- generate_readme.sh            # auto-generates a benchmark results README
    |-- postgres_create_tables_synthea.sql  # PostgreSQL schema setup for Synthea
    |-- postgres_import_data_synthea.sql    # PostgreSQL data import script
    |-- requirements.txt                  # Python dependencies
    `-- zombie/                           # unused or archived files
```
