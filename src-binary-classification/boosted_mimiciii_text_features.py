# boosted_mimiciii_text_features.py
# ---------------------------------------------------------------------
# Feature extraction script for MIMIC-III note-derived features
# Adds TF-IDF, sentiment, topic modeling, PCA/UMAP, clustering
# ---------------------------------------------------------------------

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from umap import UMAP
from textblob import TextBlob
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# 1. Load NOTEEVENTS Data
# ---------------------------------------------------------------------
note_path = os.getenv("MIMIC_NOTEEVENTS_PATH", "../src/NOTEEVENTS.csv")
notes_df = pd.read_csv(note_path, low_memory=False)

notes_df = notes_df[notes_df['TEXT'].notna() & notes_df['HADM_ID'].notna()]
keep_categories = ['Discharge summary', 'Nursing', 'Physician']
notes_df = notes_df[notes_df['CATEGORY'].isin(keep_categories)].copy()

# ---------------------------------------------------------------------
# 2. Clean and Preprocess Text
# ---------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"_+", " ", text)
    text = text.replace("[**", "").replace("**]", "")
    return text.strip()

drop_patterns = ["dictated by", "signed electronically"]
notes_df = notes_df[~notes_df['TEXT'].str.lower().str.contains('|'.join(drop_patterns), na=False)]
notes_df['TEXT'] = notes_df['TEXT'].map(clean_text)

# ---------------------------------------------------------------------
# 3. Aggregate Notes by SUBJECT_ID and HADM_ID
# ---------------------------------------------------------------------
notes_df['CHARTTIME'] = pd.to_datetime(notes_df['CHARTTIME'], errors='coerce')
notes_df = notes_df.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTTIME'])

agg_notes = (
    notes_df
    .groupby(['SUBJECT_ID', 'HADM_ID'])
    .agg(
        TEXT=('TEXT', lambda x: " ".join(x)),
        note_count=('TEXT', 'count'),
        avg_note_length=('TEXT', lambda x: np.mean([len(t.split()) for t in x]))
    )
    .reset_index()
)

# ---------------------------------------------------------------------
# 4. TF-IDF for Target Terms
# ---------------------------------------------------------------------
target_terms = [
    'pain', 'anxiety', 'depression', 'headache', 'fatigue', 'sleep',
    'sad', 'crying', 'hopeless', 'tired', 'insomnia', 'nausea', 'vomiting'
]
vectorizer = TfidfVectorizer(vocabulary=target_terms)
tfidf_matrix = vectorizer.fit_transform(agg_notes['TEXT'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{term}" for term in target_terms])

# ---------------------------------------------------------------------
# 5. Sentiment Analysis
# ---------------------------------------------------------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
agg_notes['sentiment'] = agg_notes['TEXT'].apply(get_sentiment)

# ---------------------------------------------------------------------
# 6. Topic Modeling (LDA)
# ---------------------------------------------------------------------
count_vectorizer = CountVectorizer(max_df=0.95, min_df=10, stop_words='english')
counts = count_vectorizer.fit_transform(agg_notes['TEXT'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(counts)
topics_df = pd.DataFrame(topics, columns=[f"topic_{i+1}" for i in range(topics.shape[1])])

# ---------------------------------------------------------------------
# 7. Dimensionality Reduction
# ---------------------------------------------------------------------
text_feature_matrix = pd.concat([tfidf_df, topics_df], axis=1)
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(text_feature_matrix)

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(scaled_matrix)
pca_df = pd.DataFrame(pca_components, columns=["pca_1", "pca_2"])

umap_model = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3, metric='cosine')
umap_components = umap_model.fit_transform(scaled_matrix)
umap_df = pd.DataFrame(umap_components, columns=["umap_1", "umap_2"])

# ---------------------------------------------------------------------
# 8. Clustering
# ---------------------------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_matrix)
cluster_df = pd.DataFrame({'note_cluster': kmeans_labels})

# ---------------------------------------------------------------------
# 9. Combine All Note Features
# ---------------------------------------------------------------------
note_features = pd.concat([
    agg_notes[['SUBJECT_ID', 'HADM_ID', 'note_count', 'avg_note_length', 'sentiment']],
    tfidf_df,
    topics_df,
    pca_df,
    umap_df,
    cluster_df
], axis=1)

# ---------------------------------------------------------------------
# 10. Merge with Boosted Structured Features
# ---------------------------------------------------------------------
features_path = "mimic_enriched_features.csv"
features_df = pd.read_csv(features_path)

final_df = features_df.merge(note_features, left_on=['subject_id', 'hadm_id'], right_on=['SUBJECT_ID', 'HADM_ID'], how='left')
final_df.drop(columns=['SUBJECT_ID', 'HADM_ID'], inplace=True)

# ---------------------------------------------------------------------
# 11. Save
# ---------------------------------------------------------------------
print("Saving final dataset...")
final_df.to_csv("mimic_enriched_features_w_notes.csv", index=False)
print("✅ Saved final dataset with structured + note features → mimic_enriched_features_w_notes.csv")

# ---------------------------------------------------------------------
# 12. Save tokenization sequences for ClinicalBERT
# ---------------------------------------------------------------------
note_sequences = {}
for (subj, hadm), group in notes_df.groupby(["SUBJECT_ID", "HADM_ID"]):
    clean_notes = group["TEXT"].tolist()
    note_sequences.setdefault(subj, []).append(clean_notes)

np.save("note_sequences_per_patient.npy", note_sequences)
print("✅ Saved note_sequences_per_patient.npy for ClinicalBERT.")

# ---------------------------------------------------------------------
# 13. Save completion marker
# ---------------------------------------------------------------------
with open("./boosted_features_complete.txt", "w") as f:
    f.write("Boosted structured + note features completed successfully.\n")
