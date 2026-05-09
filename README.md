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

These scripts aggregate medication text by patient, build the patient-level analysis dataset, and generate additional text-derived features for modeling.

### 5. Run the full model pipeline

```bash
chmod +x run_all_models.sh
./run_all_models.sh
```

This script runs the end-to-end modeling workflow, including preprocessing, sequence generation, model training, embedding generation, timing logs, and summary outputs.

### 6. Run benchmark iterations

```bash
chmod +x run_benchmark_iterations.sh
./run_benchmark_iterations.sh
```

This script repeats the benchmarking workflow across iterations, resumes incomplete runs when possible, organizes artifacts, and updates the project outputs.

## File Organization

```text
.
|-- README.md                    # project overview and documentation
|-- .gitignore                   # git ignore rules
|-- analysis/                    # generated artifacts, experiment outputs, and logs
|   |-- cache/                   # cached intermediate files
|   |-- data/
|   |   |-- README.md            # notes about project data
|   |   |-- rawData/             # source datasets
|   |   `-- derivedData/         # processed datasets created from source data
|   |-- experiments/             # experiment-specific artifacts
|   |-- logs/
|   |   `-- resource_usage.csv   # runtime and resource tracking output
|   |-- models/                  # saved model artifacts
|   `-- results/
|       |-- figures/             # generated figures
|       |-- metrics/             # benchmark and iteration summary outputs
|       `-- model_cards/         # model documentation artifacts
`-- src/                         # Python, SQL, and shell scripts for data prep, training, and benchmarking
    |-- *.py                     # modeling, preprocessing, evaluation, and utility scripts
    |-- *.sql                    # PostgreSQL setup and import scripts
    |-- *.sh                     # workflow and artifact organization scripts
    |-- requirements.txt         # Python dependencies
    `-- zombie/                  # unused files
```
