# backend/vector_store.py
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH, FAISS_META_PATH, EMBED_MODEL_NAME

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def ensure_index_dir():
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) or ".", exist_ok=True)

def embed_text(texts):
    return EMBED_MODEL.encode(texts, convert_to_numpy=True)

def create_faiss_index(dim):
    return faiss.IndexFlatIP(dim)

def save_index(index, meta):
    ensure_index_dir()
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_META_PATH):
        return None, None
    idx = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta

def append_to_index(vectors: np.ndarray, meta_list: list):
    ensure_index_dir()
    idx, meta = load_index()
    if idx is None:
        idx = create_faiss_index(vectors.shape[1])
        meta = []
    faiss.normalize_L2(vectors)
    idx.add(vectors)
    meta.extend(meta_list)
    save_index(idx, meta)
    return idx, meta

def search_index(query_vec, topk=10):
    idx, meta = load_index()
    if idx is None:
        return []
    q = query_vec.copy().reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = idx.search(q, topk)
    results = []
    for dist, pos in zip(D[0], I[0]):
        if pos < 0: continue
        results.append({"meta": meta[pos], "score": float(dist)})
    return results
