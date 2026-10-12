"""Explore the keyword-based pain-label definition used by synthea_extract.

This is a test-only script that prints patient-level label counts and the
most common matching diagnoses. It does not modify source data.
"""

from pathlib import Path
import re

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "analysis" / "data" / "rawData"
PATIENTS_PATH = DATA_DIR / "patients.csv"
CONDITIONS_PATH = DATA_DIR / "conditions.csv"

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash"
]
PAIN_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in PAIN_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def main():
    patients = pd.read_csv(PATIENTS_PATH, usecols=["Id"])
    conditions = pd.read_csv(CONDITIONS_PATH, usecols=["PATIENT", "CODE", "DESCRIPTION"])

    patient_ids = patients["Id"].dropna().astype(str).drop_duplicates()
    conditions["PATIENT"] = conditions["PATIENT"].astype(str)
    conditions["DESCRIPTION"] = conditions["DESCRIPTION"].astype("string")

    matching = conditions[
        conditions["DESCRIPTION"].str.contains(PAIN_PATTERN, na=False)
    ].copy()

    positive_ids = set(matching["PATIENT"].dropna().astype(str))
    labels = patient_ids.isin(positive_ids)

    total = len(patient_ids)
    positive_count = int(labels.sum())
    negative_count = total - positive_count

    print("Exploratory patient-level binary_label (description keywords)")
    print(f"Definition terms: {', '.join(PAIN_KEYWORDS)}")
    print(f"Patients: {total:,}")
    print(
        f"Positive (1): {positive_count:,} "
        f"({positive_count / total:.2%})" if total else "Positive (1): 0 (0.00%)"
    )
    print(
        f"Negative (0): {negative_count:,} "
        f"({negative_count / total:.2%})" if total else "Negative (0): 0 (0.00%)"
    )
    print(f"Matching condition rows: {len(matching):,}")
    print("\nMost common matching diagnoses:")

    top_diagnoses = (
        matching.groupby(["CODE", "DESCRIPTION"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    if top_diagnoses.empty:
        print("No condition descriptions matched.")
    else:
        print(top_diagnoses.to_string())


if __name__ == "__main__":
    main()
