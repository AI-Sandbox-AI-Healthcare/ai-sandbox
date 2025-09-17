import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------
# 1. Database Connection
# ---------------------------------------------------------------------
HOST = "localhost"
DBNAME = "mimic"
USER = "gchism"
PASSWORD = "Chism1154!"  # Use secure password practices
SCHEMA = "mimiciii"
PORT = 5432

DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text(f"SET search_path TO {SCHEMA};"))

print("Connected to the MIMIC-III database.")

# ---------------------------------------------------------------------
# 2. Define ICD-9 Code Sets
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

MENTAL_HEALTH_SET = {code for codes in MENTAL_HEALTH_CODES.values() for code in codes}
CHRONIC_PAIN_SET = {code for codes in CHRONIC_PAIN_CODES.values() for code in codes}

def is_mental_health_code(code):
    return code.replace('.', '')[:4] in MENTAL_HEALTH_SET

def is_chronic_pain_code(code):
    return code.replace('.', '')[:4] in CHRONIC_PAIN_SET

# ---------------------------------------------------------------------
# 3. Load Tables
# ---------------------------------------------------------------------
print("Loading tables...")
diag_df = pd.read_sql("""
    SELECT subject_id, hadm_id, icd9_code
    FROM diagnoses_icd WHERE icd9_code IS NOT NULL;
""", engine)

adm_df = pd.read_sql("""
    SELECT subject_id, hadm_id, admittime, dischtime, insurance, admission_type
    FROM admissions;
""", engine)

pat_df = pd.read_sql("""
    SELECT subject_id, gender, dob, dod, expire_flag
    FROM patients;
""", engine)

transfers_df = pd.read_sql("""
    SELECT subject_id, hadm_id, icustay_id, intime, outtime, curr_careunit
    FROM transfers;
""", engine)

services_df = pd.read_sql("""
    SELECT subject_id, hadm_id, curr_service
    FROM services;
""", engine)

prescriptions_df = pd.read_sql("""
    SELECT subject_id, hadm_id, drug
    FROM prescriptions;
""", engine)

print("All tables loaded.")

# ---------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------
print("Processing features...")

diag_agg = diag_df.groupby(['subject_id', 'hadm_id'])['icd9_code'].apply(list).reset_index()
diag_agg['has_mental_health'] = diag_agg['icd9_code'].apply(lambda codes: any(is_mental_health_code(c) for c in codes))
diag_agg['has_chronic_pain'] = diag_agg['icd9_code'].apply(lambda codes: any(is_chronic_pain_code(c) for c in codes))
diag_agg['diagnosis_count'] = diag_agg['icd9_code'].apply(len)
diag_agg['icd9_code'] = diag_agg['icd9_code'].apply(lambda x: ', '.join(x))

merged_df = diag_agg.merge(adm_df, on=['subject_id', 'hadm_id'], how='left') \
                    .merge(pat_df, on='subject_id', how='left') \
                    .merge(services_df, on=['subject_id', 'hadm_id'], how='left')

merged_df['approx_age'] = 2101 - pd.to_datetime(merged_df['dob']).dt.year
merged_df['co_occurrence'] = merged_df['has_mental_health'] & merged_df['has_chronic_pain']
merged_df['seen_by_psych'] = merged_df['curr_service'].fillna('').str.contains("PSY", case=False)
merged_df['length_of_stay'] = (pd.to_datetime(merged_df['dischtime']) - pd.to_datetime(merged_df['admittime'])).dt.days
merged_df['in_hospital_mortality'] = pd.to_datetime(merged_df['dod']).between(
    pd.to_datetime(merged_df['admittime']), pd.to_datetime(merged_df['dischtime'])
)

# Medication flags
merged_df['on_psych_or_pain_meds'] = merged_df['hadm_id'].isin(
    prescriptions_df[
        prescriptions_df['drug'].str.lower().str.contains("morphine|prozac", na=False)
    ]['hadm_id']
)

med_count = prescriptions_df.groupby(['subject_id', 'hadm_id']).size().reset_index(name='medication_count')
merged_df = merged_df.merge(med_count, on=['subject_id', 'hadm_id'], how='left')

# ICU flags
icu_keywords = ['MICU', 'SICU', 'CCU', 'ICU']
transfers_df['was_in_icu'] = transfers_df['curr_careunit'].isin(icu_keywords)
icu_flag = transfers_df.groupby(['subject_id', 'hadm_id'])['was_in_icu'].max().reset_index()
merged_df = merged_df.merge(icu_flag, on=['subject_id', 'hadm_id'], how='left')

# ---------------------------------------------------------------------
# 5. Transfers Event Sequence
# ---------------------------------------------------------------------
transfers_df['intime'] = pd.to_datetime(transfers_df['intime'])
event_sequence = (
    transfers_df.sort_values(['subject_id', 'hadm_id', 'intime'])
    .groupby(['subject_id', 'hadm_id'])[['curr_careunit', 'intime']]
    .apply(lambda df: list(zip(df['curr_careunit'], df['intime'].astype(str))))
    .reset_index(name='curr_careunit_sequence')
)

merged_df = merged_df.merge(event_sequence, on=['subject_id', 'hadm_id'], how='left')

# ---------------------------------------------------------------------
# 6. Save Output
# ---------------------------------------------------------------------
final_cols = [
    'subject_id', 'hadm_id', 'icd9_code', 'diagnosis_count', 'has_mental_health',
    'has_chronic_pain', 'co_occurrence', 'insurance', 'admission_type', 'gender',
    'approx_age', 'admittime', 'dischtime', 'length_of_stay', 'curr_careunit_sequence',
    'curr_service', 'seen_by_psych', 'on_psych_or_pain_meds', 'medication_count',
    'was_in_icu', 'in_hospital_mortality', 'expire_flag'
]

output_csv = "mimic-data/files/mimiciii/1.4/mimic_enriched_features.csv"
merged_df[final_cols].to_csv(output_csv, index=False)
print(f"Saved enriched dataset to {output_csv}.")

engine.dispose()
print("Database connection closed.")
