import os, numpy as np, pandas as pd, joblib
from sentence_transformers import SentenceTransformer
import nltk, lyricsgenius
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
titles    = pd.read_pickle(os.path.join(CACHE_DIR, "titles.pkl"))
lyr_vec   = np.load(os.path.join(CACHE_DIR, "lyr_vec.npy"))
sent_feats= np.load(os.path.join(CACHE_DIR, "sent_feats.npy"))
topic_vec = np.load(os.path.join(CACHE_DIR, "topic_vec.npy"))
hybrid    = np.load(os.path.join(CACHE_DIR, "hybrid.npy"))
red       = np.load(os.path.join(CACHE_DIR, "red.npy"))
umap      = joblib.load(os.path.join(CACHE_DIR, "umap.pkl"))
knn       = joblib.load(os.path.join(CACHE_DIR, "knn.pkl"))
topic_model = BERTopic.load(os.path.join(CACHE_DIR, "bertopic"))

load_dotenv()
genius = lyricsgenius.Genius(os.getenv("GENIUS_ACCESS_TOKEN"),
                             skip_non_songs=True, remove_section_headers=True)
sia    = SentimentIntensityAnalyzer()
bert   = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_lyrics(q:str):
    song = genius.search_song(q)
    return song.lyrics if song and song.lyrics else None

def _compose_vector(text:str):
    v_bert = bert.encode([text])[0]
    s      = list(sia.polarity_scores(text).values())
    t      = topic_model.transform([text])[0][0]
    t      = (t - topic_vec.min()) / (topic_vec.max() - topic_vec.min())  
    return np.concatenate([v_bert, s, [t]])

def recommend_similar_songs(query:str, k:int=5):
    lyr = fetch_lyrics(query)
    if not lyr:
        raise ValueError("Lyrics not found")
    vec = _compose_vector(lyr)
    vec_red = umap.transform([vec])
    dists, idx = knn.kneighbors(vec_red, n_neighbors=k+5)  
    
    input_song = genius.search_song(query)
    input_title = input_song.title.lower() if input_song else ""
    
    results = []
    min_dist = min(dists[0])
    max_dist = max(dists[0])
    
    for j, i in enumerate(idx[0]):
        title = titles[i].lower()
        if input_title and (input_title in title or title in input_title):
            continue
            
        if max_dist == min_dist:
            score = 0.95 
        else:
            norm_dist = (dists[0][j] - min_dist) / (max_dist - min_dist) 
            score = 0.95 - (0.35 * norm_dist) 
            
        results.append({"track_name": titles[i], "score": float(f"{score:.4f}")})
        
        if len(results) >= k:  
            break
            
    return results[:k]
