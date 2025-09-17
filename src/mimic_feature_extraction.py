#!/usr/bin/env python3
"""
MIMIC-III Feature Extraction Script

This script:
- Connects to the MIMIC-III database.
- Extracts relevant data from `DIAGNOSES_ICD`, `ADMISSIONS`, and `PATIENTS`.
- Flags mental health and chronic pain conditions.
- Merges everything into a single DataFrame for further analysis.
- Saves the processed dataset to a CSV file.

Requirements:
- Replace the connection details (`HOST`, `DBNAME`, `USER`, `PASSWORD`) with your credentials.
- Ensure PostgreSQL (`psycopg2` or `SQLAlchemy`) is installed.

"""

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------
# 1. Define Database Connection
# ---------------------------------------------------------------------
HOST = "localhost"
DBNAME = "mimic"
USER = "gchism"
PASSWORD = "Chism1154!"  # Ensure to use a secure password
SCHEMA = "mimiciii"
PORT = 5432

# Use SQLAlchemy for efficient connection management
DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
engine = create_engine(DATABASE_URL)

# Ensure schema is set before queries
with engine.begin() as conn:
    conn.execute(text(f"SET search_path TO {SCHEMA};"))

print("Connected to the MIMIC-III database.")

# ---------------------------------------------------------------------
# 2. Define ICD-9 Code Sets for Mental Health & Chronic Pain
# ---------------------------------------------------------------------
MENTAL_HEALTH_CODES = {
    'depression': ['2962', '2963', '311'],
    'bipolar': ['2964', '2965', '2966', '2967'],
    'anxiety': ['3000', '3002', '3003'],
    'ptsd': ['30981'],
    'psychotic': ['295', '297', '298'],
    'personality': ['301'],
    'substance_use': ['303', '304', '305'],
}

CHRONIC_PAIN_CODES = {
    'chronic_pain_general': ['3382', '3384'],
    'back_pain': ['724'],
    'arthritis': ['714', '715'],
    'fibromyalgia': ['7291'],
    'migraine': ['346'],
    'headache': ['7840'],
    'diabetic_neuropathy': ['2506', '3572'],
}

# Flatten sets for quick lookups
MENTAL_HEALTH_SET = {code for codes in MENTAL_HEALTH_CODES.values() for code in codes}
CHRONIC_PAIN_SET = {code for codes in CHRONIC_PAIN_CODES.values() for code in codes}

def is_mental_health_code(icd9_code: str) -> bool:
    return icd9_code.replace('.', '')[:4] in MENTAL_HEALTH_SET

def is_chronic_pain_code(icd9_code: str) -> bool:
    return icd9_code.replace('.', '')[:4] in CHRONIC_PAIN_SET

# ---------------------------------------------------------------------
# 3. Extract Diagnoses Data
# ---------------------------------------------------------------------
query_diagnoses = """
    SELECT subject_id, hadm_id, icd9_code
    FROM mimiciii.diagnoses_icd
    WHERE icd9_code IS NOT NULL;
"""

diag_df = pd.read_sql(query_diagnoses, engine)
print(f"Fetched {len(diag_df)} rows from DIAGNOSES_ICD.")

# Aggregate by (subject_id, hadm_id)
diag_agg = diag_df.groupby(['subject_id', 'hadm_id'])['icd9_code'].apply(list).reset_index()
diag_agg['has_mental_health'] = diag_agg['icd9_code'].apply(lambda codes: any(is_mental_health_code(code) for code in codes))
diag_agg['has_chronic_pain'] = diag_agg['icd9_code'].apply(lambda codes: any(is_chronic_pain_code(code) for code in codes))
diag_agg['icd9_code'] = diag_agg['icd9_code'].apply(lambda x: ', '.join(x))  # Convert list to string

# ---------------------------------------------------------------------
# 4. Merge with ADMISSIONS & PATIENTS Data
# ---------------------------------------------------------------------
query_admissions = """
    SELECT subject_id, hadm_id, insurance, admission_type
    FROM mimiciii.admissions;
"""
adm_df = pd.read_sql(query_admissions, engine)
print(f"Fetched {len(adm_df)} rows from ADMISSIONS.")

query_patients = """
    SELECT subject_id, gender, dob, dod, expire_flag, anchor_age, anchor_year
    FROM mimiciii.patients;
"""
pat_df = pd.read_sql(query_patients, engine)
print(f"Fetched {len(pat_df)} rows from PATIENTS.")

# Merge everything together
merged_df = diag_agg.merge(adm_df, on=['subject_id', 'hadm_id'], how='left')
merged_df = merged_df.merge(pat_df, on='subject_id', how='left')

# ---------------------------------------------------------------------
# 5. Additional Feature Engineering
# ---------------------------------------------------------------------
merged_df['approx_age'] = merged_df['anchor_age']
merged_df['co_occurrence'] = merged_df['has_mental_health'] & merged_df['has_chronic_pain']

final_cols = [
    'subject_id', 'hadm_id', 'icd9_code', 
    'has_mental_health', 'has_chronic_pain', 'co_occurrence',
    'insurance', 'admission_type', 'gender', 'approx_age'
]
processed_df = merged_df[final_cols].copy()

# ---------------------------------------------------------------------
# 6. Save to CSV
# ---------------------------------------------------------------------
output_csv = "mimic_feature_engineered.csv"
processed_df.to_csv(output_csv, index=False)
print(f"Feature-engineered data saved to {output_csv}.")

# Close the database connection
engine.dispose()
print("Database connection closed.")
