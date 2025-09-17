# clinicalbert_tokenize_notes.py
# ---------------------------------------------------------------------
# Efficient Parallel Tokenization for ClinicalBERT
# ---------------------------------------------------------------------

import os
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--metric_prefix", type=str, default="iter1")
args = parser.parse_args()

# ---------------------------------------------------------------------
# 1. Load notes
# ---------------------------------------------------------------------
notes_path = "./note_sequences_per_patient.npy"
note_sequences = np.load(notes_path, allow_pickle=True).item()
# ---------------------------------------------------------------------
# Filter to shared validation subjects only (if available)
# ---------------------------------------------------------------------
val_ids_path = f"./shared_val_ids_{args.metric_prefix}.npy"
if os.path.exists(val_ids_path):
    shared_val_ids = set(np.load(val_ids_path))
    note_sequences = {sid: notes for sid, notes in note_sequences.items() if sid in shared_val_ids}
    print(f"🔍 Restricting to {len(note_sequences)} shared validation subjects")
else:
    print("⚠️ No val_ids filter applied — tokenizing all subjects")

# ---------------------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------------------
MAX_NOTES_PER_ADMISSION = 5
MAX_TOKENS_PER_NOTE = 256
CACHE_DIR = "./tokenized_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 3. Global Tokenizer for Worker Processes
# ---------------------------------------------------------------------
tokenizer = None
def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# ---------------------------------------------------------------------
# 4. Tokenization function for a single patient
# ---------------------------------------------------------------------
def tokenize_patient(subject_entry):
    global tokenizer
    subject_id, admission_notes = subject_entry
    admission_texts = []

    for note_list in admission_notes[:MAX_NOTES_PER_ADMISSION]:
        if not note_list:
            continue
        joined_note = " ".join(note_list)[:10_000]
        admission_texts.append(joined_note)

    if len(admission_texts) == 0:
        return subject_id, None  # skip empty

    while len(admission_texts) < MAX_NOTES_PER_ADMISSION:
        admission_texts.append("")  # pad with blanks

    encoded = tokenizer.batch_encode_plus(
        admission_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS_PER_NOTE,
        return_tensors="pt"
    )

    return subject_id, {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }

# ---------------------------------------------------------------------
# 5. Parallel tokenization
# ---------------------------------------------------------------------
tokenized_sequences = {}
skipped_subjects = []
print("\n🚀 Starting parallel batch tokenization...")

max_workers = min(8, os.cpu_count() or 1)
with ProcessPoolExecutor(max_workers=max_workers, initializer=init_tokenizer) as executor:
    futures = {executor.submit(tokenize_patient, entry): entry[0] for entry in note_sequences.items()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing patients (parallel)"):
        try:
            subject_id, tokenized = future.result()
            if tokenized is None:
                skipped_subjects.append(subject_id)
            else:
                tokenized_sequences[subject_id] = tokenized
                if len(tokenized_sequences) % 1000 == 0:
                    print(f"🧪 Tokenized {len(tokenized_sequences)} patients...")
        except Exception as e:
            print(f"⚠️ Error tokenizing subject {futures[future]}: {e}")
            skipped_subjects.append(futures[future])

# ---------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------
out_path = "./tokenized_clinicalbert_notes.pt"
torch.save(tokenized_sequences, out_path, pickle_protocol=4)

# ---------------------------------------------------------------------
# 7. Summary Report
# ---------------------------------------------------------------------
print("\n📋 Parallel Tokenization Summary")
print("--------------------------------------------------")
print(f"🧑‍⚕️ Patients tokenized: {len(tokenized_sequences)}")
print(f"📜 Max notes per admission: {MAX_NOTES_PER_ADMISSION}")
print(f"🔠 Max tokens per note: {MAX_TOKENS_PER_NOTE}")
print(f"📂 Saved to: {out_path}")
print("✅ Parallel tokenization complete!")

# ---------------------------------------------------------------------
# 8. Save completion marker
# ---------------------------------------------------------------------
with open("./tokenization_complete.txt", "w") as f:
    f.write(f"Tokenization completed successfully.\nPatients: {len(tokenized_sequences)}\n")
np.save(f"tokenized_subject_ids_{args.metric_prefix}.npy", list(tokenized_sequences.keys()))