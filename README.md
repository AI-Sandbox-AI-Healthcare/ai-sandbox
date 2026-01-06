# AI Sandbox

The **AI Sandbox** is a modular, reproducible experimentation environment for applied artificial intelligence in healthcare and synthetic electronic health record (EHR) data. It was developed as part of a broader effort aligned with the **NAIRR Classroom Pilot** to lower barriers to advanced AI experimentation while preserving rigor, transparency, and reproducibility.

The Sandbox supports **end-to-end benchmarking**, from data preprocessing and feature engineering through model training, validation, and stacking-based meta-learning, using both CPU- and GPU-based workflows.

---

## Motivation and Project Goals

Modern AI education and research increasingly depend on access to **complex data, scalable compute, and reproducible workflows**, yet these resources are often fragmented or inaccessible to students and early-stage researchers. The AI Sandbox addresses this gap by providing a **structured, auditable, and extensible environment** for hands-on AI experimentation.

Grounded in the goals of the NAIRR Classroom Pilot, this project is motivated by the need to:

* **Democratize access to advanced AI workflows**
  Enable learners and researchers to engage with realistic, end-to-end machine learning pipelines without requiring bespoke infrastructure or ad hoc setups.

* **Support reproducibility and benchmarking as first-class practices**
  Emphasize repeated runs, variance estimation, logging, and result aggregation rather than one-off model performance.

* **Bridge education and research**
  Use the same pipeline for instructional settings (e.g., classrooms, capstones) and for method development, evaluation, and comparison.

* **Expose learners to responsible, real-world AI systems**
  Model the full lifecycle of applied AI—including preprocessing, feature construction, validation, and meta-learning—rather than isolated algorithms.

The AI Sandbox is intentionally designed to be **infrastructure-agnostic**: while it is well suited for use on platforms such as Jetstream2 via NAIRR or ACCESS, it does not require specialized infrastructure development within the project itself.

---

## Repository Structure (High-Level)

```
.
├── src/
│   ├── process_noteevents_text.py
│   ├── synthea_extract.py
│   ├── boosted_synthea_text_features.py
│   ├── run_all_models.sh
│   └── run_benchmark_iterations.sh
│
├── analysis/
│   ├── data/
│   ├── results/
│   └── logs/
│
├── requirements.txt
└── README.md
```

* **`src/`** contains all processing, modeling, and orchestration scripts
* **`analysis/data/`** stores intermediate and processed datasets
* **`analysis/results/`** stores model outputs and benchmark summaries
* **`analysis/logs/`** stores timing, resource, and execution logs

---

## 1. Setup Instructions

Follow the steps below to set up the environment and run the full pipeline.

### 1.1 Create and Activate a Virtual Environment

Ensure Python virtual environment tools are installed:

```bash
sudo apt install python3-venv
```

Create a virtual environment:

```bash
python3 -m venv myenv
```

Activate the environment:

```bash
source ~/Desktop/myenv/bin/activate
```

---

### 1.2 Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

> ⚠️ GPU-based models require a CUDA-enabled environment and a compatible PyTorch installation.

---

## 2. Running the Data Processing Pipeline

Once the environment is ready, run the processing scripts **from the `src/` directory**, in order:

```bash
python3 src/process_noteevents_text.py
python3 src/synthea_extract.py
python3 src/boosted_synthea_text_features.py
```

### Script Descriptions

#### `process_noteevents_text.py`

* Loads medication records from a Synthea dataset (`medications.csv`)
* Cleans column names
* Aggregates medication descriptions per patient into a single block of text
* Constructs **ordered medication-note sequences** per patient
* Outputs patient-level medication text sequences for downstream NLP modeling

---

#### `synthea_extract.py`

* Loads and merges multiple Synthea datasets
* Identifies pain-related patient cohorts
* Constructs patient-level structured features including:

  * Demographics
  * Encounters
  * Conditions
  * Medications
  * Procedures
  * Key vitals
* Cleans data, fills missing values, and prepares features for machine-learning use

---

#### `boosted_synthea_text_features.py`

* Builds **enhanced text-derived features** from medication notes
* Generates:

  * TF-IDF features
  * Sentiment scores
  * Topic models
  * Clustering-based representations
* Reduces high-dimensional text outputs into clean numeric features
* Merges text-derived features into the structured Synthea dataset

---

## 3. Run the Full Modeling Pipeline

Make the main execution script executable and run it:

```bash
chmod +x src/run_all_models.sh
./src/run_all_models.sh
```

### What `run_all_models.sh` Does

This script runs the **full machine-learning pipeline end to end**, including:

* Data preprocessing and validation
* Sequence construction
* CPU and GPU model training
* Transformer-based (BERT) embedding generation
* Stacking meta-learner training
* Timing and resource logging
* Automatic merging of results into summary files

Outputs are written to:

* `analysis/results/`
* `analysis/logs/`

---

## 4. Run Benchmark Iterations

To perform **multi-run benchmarking with safe resume**:

```bash
chmod +x src/run_benchmark_iterations.sh
./src/run_benchmark_iterations.sh
```

### What `run_benchmark_iterations.sh` Does

* Runs multiple iterations of the full benchmarking pipeline
* Automatically resumes if some iterations were already completed
* Handles failures safely
* Logs timing and performance metrics
* Merges results across iterations
* Generates summary artifacts and reports
* Organizes outputs in `analysis/`
* Updates benchmark summaries for reproducibility

This mode is intended for **variance estimation, robustness checks, and comparative evaluation** rather than single-run performance reporting.

---

## Intended Use

The AI Sandbox is designed for:

* NAIRR-aligned classroom pilots
* Capstone and independent projects
* Method benchmarking and ablation studies
* Reproducible AI education and research workflows

---

## License

*Add license information here.*
