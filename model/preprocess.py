"""
preprocess.py
-------------
Create all cached artefacts for the hybrid‑lyrics recommender.
Run this ONCE (or whenever you change the dataset)::

    python preprocess.py
"""
import os, joblib, numpy as np, pandas as pd
from dotenv import load_dotenv
import nltk, lyricsgenius
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# ── config ──────────────────────────────────────────
load_dotenv()
CSV_FILE  = "spotify_millsongdata.csv"
MODEL_ID  = "all-MiniLM-L6-v2"
CACHE_DIR = "cache"            # everything saved here
os.makedirs(CACHE_DIR, exist_ok=True)
# ----------------------------------------------------

nltk.download("vader_lexicon")
bert  = SentenceTransformer(MODEL_ID)
sia   = SentimentIntensityAnalyzer()

# 1. Load dataset
df = pd.read_csv(CSV_FILE).dropna(subset=["artist", "song", "text"])
df["title"] = df["song"].str.strip() + " - " + df["artist"].str.strip()
titles  = df["title"].tolist()
lyrics  = df["text"].tolist()

# 2. BERT embeddings
print("▶ Encoding lyrics with BERT …")
lyr_vec = bert.encode(lyrics, batch_size=64, show_progress_bar=True)
np.save(os.path.join(CACHE_DIR, "lyr_vec.npy"), lyr_vec)

# 3. Sentiment features
sent_feats = np.array([
    list(sia.polarity_scores(t).values()) for t in lyrics
], dtype="float32")
np.save(os.path.join(CACHE_DIR, "sent_feats.npy"), sent_feats)

# 4. Topics with BERTopic
print("▶ Training BERTopic …")
topic_model = BERTopic(verbose=False)
topics, _   = topic_model.fit_transform(lyrics)
topic_norm  = MinMaxScaler().fit_transform(np.array(topics).reshape(-1,1))
np.save(os.path.join(CACHE_DIR, "topic_vec.npy"), topic_norm)
topic_model.save(os.path.join(CACHE_DIR, "bertopic"))  # folder

# 5. Hybrid vector
hybrid = np.concatenate([lyr_vec, sent_feats, topic_norm], axis=1)
np.save(os.path.join(CACHE_DIR, "hybrid.npy"), hybrid)

# 6. UMAP reduction
print("▶ UMAP …")
umap = UMAP(n_neighbors=15, n_components=15, metric='cosine', random_state=42)
red  = umap.fit_transform(hybrid)
joblib.dump(umap, os.path.join(CACHE_DIR, "umap.pkl"))
np.save(os.path.join(CACHE_DIR, "red.npy"), red)

# 7. KNN index
knn = NearestNeighbors(metric='cosine', n_neighbors=5).fit(red)
joblib.dump(knn, os.path.join(CACHE_DIR, "knn.pkl"))

# 8. Optional clustering
kmeans = KMeans(n_clusters=10, random_state=42).fit(red)
df["cluster"] = kmeans.labels_
df.to_csv(os.path.join(CACHE_DIR, "dataset_with_clusters.csv"), index=False)
joblib.dump(kmeans, os.path.join(CACHE_DIR, "kmeans.pkl"))

# 9. Save titles list
pd.Series(titles).to_pickle(os.path.join(CACHE_DIR, "titles.pkl"))

print("✅  All artefacts saved to ./cache/")
