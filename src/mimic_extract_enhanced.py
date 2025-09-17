import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------
# 1. Database Connection
# ---------------------------------------------------------------------
HOST = os.getenv("MIMIC_HOST", "localhost")
DBNAME = os.getenv("MIMIC_DBNAME", "mimic")
USER = os.getenv("MIMIC_USER", "gchism")
PASSWORD = os.getenv("MIMIC_PASSWORD", "Chism1154")  # Set this in your environment
SCHEMA = os.getenv("MIMIC_SCHEMA", "mimiciii")
PORT = int(os.getenv("MIMIC_PORT", 5432))

DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text(f"SET search_path TO {SCHEMA};"))

print("Connected to the MIMIC-III database.")

# ---------------------------------------------------------------------
# 2. Define ICD-9 Code Sets + target medications
# ---------------------------------------------------------------------
MENTAL_HEALTH_CODES = {
    'depression': ['2962', '2963', '311'],
    'bipolar': ['2964', '2965', '2966', '2967'],
    'anxiety': ['3000', '3002', '3003'],
    'ptsd': ['30981'],
    'psychotic': ['295', '297', '298'],
    'personality': ['301'],
    'substance_use': ['303', '304', '305']
}

CHRONIC_PAIN_CODES = {
    'chronic_pain_general': ['3382', '3384'],
    'back_pain': ['724'],
    'arthritis': ['714', '715'],
    'fibromyalgia': ['7291'],
    'migraine': ['346'],
    'headache': ['7840'],
    'diabetic_neuropathy': ['2506', '3572']
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

# Multiclass label assignment

def assign_multiclass_label(row):
    if row['has_mental_health'] and row['has_chronic_pain']:
        return 2
    elif row['has_chronic_pain']:
        return 1
    elif row['has_mental_health']:
        return 0
    else:
        return None

diag_agg['multiclass_label'] = diag_agg.apply(assign_multiclass_label, axis=1)

# Merge all tabels
merged_df = diag_agg.merge(adm_df, on=['subject_id', 'hadm_id'], how='left') \
                    .merge(pat_df, on='subject_id', how='left') \
                    .merge(services_df, on=['subject_id', 'hadm_id'], how='left')

# Additional features
admit_year = pd.to_datetime(merged_df['admittime']).dt.year
merged_df['approx_age'] = admit_year - pd.to_datetime(merged_df['dob']).dt.year
merged_df['co_occurrence'] = merged_df['has_mental_health'] & merged_df['has_chronic_pain']
merged_df['seen_by_psych'] = merged_df['curr_service'].fillna('').str.contains("PSY", case=False)
merged_df['length_of_stay'] = (pd.to_datetime(merged_df['dischtime']) - pd.to_datetime(merged_df['admittime'])).dt.days
merged_df['in_hospital_mortality'] = pd.to_datetime(merged_df['dod']).between(
    pd.to_datetime(merged_df['admittime']), pd.to_datetime(merged_df['dischtime'])
)

# Insurance Group
merged_df['insurance_group'] = merged_df['insurance'].fillna('UNKNOWN').str.upper()

def simplify_insurance(ins):
    if 'MEDICARE' in ins:
        return 'Medicare'
    elif 'MEDICAID' in ins:
        return 'Medicaid'
    elif 'PRIVATE' in ins:
        return 'Private'
    elif 'GOVERNMENT' in ins:
        return 'Government'
    elif 'SELF PAY' in ins:
        return 'Self Pay'
    else:
        return 'Other'

merged_df['insurance_group'] = merged_df['insurance_group'].apply(simplify_insurance)


# Medication flags (match both subject_id and hadm_id)
PSYCH_MEDICATIONS = [
    'prozac', 'sertraline', 'zoloft', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram', 
    'venlafaxine', 'duloxetine', 'bupropion', 'amitriptyline', 'nortriptyline', 'trazodone',
    'olanzapine', 'risperidone', 'quetiapine', 'aripiprazole', 'haloperidol', 'lithium',
    'diazepam', 'lorazepam', 'alprazolam', 'clonazepam', 'buspirone'
]

PAIN_MEDICATIONS = [
    'morphine', 'oxycodone', 'hydrocodone', 'fentanyl', 'codeine', 'tramadol', 
    'gabapentin', 'pregabalin', 'lidocaine', 'methadone', 'buprenorphine',
    'ibuprofen', 'naproxen', 'celecoxib', 'diclofenac'
]

MEDICATION_REGEX = r'\\b(?:' + '|'.join([med.lower() for med in (PSYCH_MEDICATIONS + PAIN_MEDICATIONS)]) + r')\\b'

PAIN_MEDICATIONS = [
    'morphine', 'oxycodone', 'hydrocodone', 'fentanyl', 'codeine', 'tramadol', 
    'gabapentin', 'pregabalin', 'lidocaine', 'methadone', 'buprenorphine',
    'ibuprofen', 'naproxen', 'celecoxib', 'diclofenac'
]

filtered_rx = prescriptions_df[
    prescriptions_df['drug'].str.lower().str.contains(MEDICATION_REGEX, na=False)
][['subject_id', 'hadm_id']].drop_duplicates()

filtered_rx['on_psych_or_pain_meds'] = True
merged_df = merged_df.merge(filtered_rx, on=['subject_id', 'hadm_id'], how='left')
merged_df['on_psych_or_pain_meds'] = merged_df['on_psych_or_pain_meds'].fillna(False)
medication_count = prescriptions_df.groupby(['subject_id', 'hadm_id']).size().reset_index(name='medication_count')
merged_df = merged_df.merge(medication_count, on=['subject_id', 'hadm_id'], how='left')

merged_df['medication_count'] = merged_df['medication_count'].fillna(0)
merged_df['psych_or_pain_rx_count'] = merged_df['psych_or_pain_rx_count'].fillna(0)
merged_df['polypharmacy_flag'] = (merged_df['medication_count'] >= 5).astype(int)

# ICU flags (normalize careunit strings)
transfers_df['curr_careunit'] = transfers_df['curr_careunit'].str.upper()
transfers_df['was_in_icu'] = transfers_df['curr_careunit'].str.contains('ICU|CCU|MICU|SICU', na=False)
was_in_icu = transfers_df.groupby(['subject_id', 'hadm_id'])['was_in_icu'].max().reset_index()
transfer_count = transfers_df.groupby(['subject_id', 'hadm_id']).size().reset_index(name='transfer_count')

merged_df = merged_df.merge(was_in_icu, on=['subject_id', 'hadm_id'], how='left')
merged_df = merged_df.merge(transfer_count, on=['subject_id', 'hadm_id'], how='left')
merged_df['was_in_icu'] = merged_df['was_in_icu'].fillna(False)
merged_df['transfer_count'] = merged_df['transfer_count'].fillna(0)

# Transfer Event Sequence
transfers_df['intime'] = pd.to_datetime(transfers_df['intime'])
event_sequence = (
    transfers_df.sort_values(['subject_id', 'hadm_id', 'intime'])
    .groupby(['subject_id', 'hadm_id'])[['curr_careunit', 'intime']]
    .apply(lambda df: list(zip(df['curr_careunit'], df['intime'].astype(str))))
    .reset_index(name='curr_careunit_sequence')
)

merged_df = merged_df.merge(event_sequence, on=['subject_id', 'hadm_id'], how='left')

# ---------------------------------------------------------------------
# 5. Save Output
# ---------------------------------------------------------------------
final_cols = [
    'subject_id', 'hadm_id', 'icd9_code', 'diagnosis_count', 'has_mental_health',
    'has_chronic_pain', 'co_occurrence', 'multiclass_label', 'insurance', 'admission_type', 'gender',
    'approx_age', 'admittime', 'dischtime', 'length_of_stay', 'curr_careunit_sequence',
    'curr_service', 'seen_by_psych', 'on_psych_or_pain_meds', 'medication_count',
    'was_in_icu', 'in_hospital_mortality', 'expire_flag', 'transfer_count', 'insurance_group',
    'polypharmacy_flag', 'psych_or_pain_rx_count',
]

output_csv = "mimic_enriched_features.csv"
merged_df[final_cols].to_csv(output_csv, index=False)
print(f"Saved enriched dataset to {output_csv}.")

engine.dispose()
print("Database connection closed.")

counts = merged_df["multiclass_label"].value_counts(dropna=False).sort_index()
print("\n📊 Class Distribution Before Saving:")
for label in [0, 1, 2]:
    print(f"  Class {label}: {counts.get(label, 0)}")
