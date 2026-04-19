import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","in","is","it",
    "of","on","or","that","the","their","they","this","to","was","were","with","you","your"
}


def clean_query(q: str) -> str:
    q = (q or "").replace("\n", " ").replace("\r", " ").strip()
    q = re.sub(r"[^A-Za-z0-9\s']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def tokenize(text: str) -> List[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]


class CaseLawSearchEngine:
    def __init__(self):
        self.ready = False
        self.model_name = os.getenv("EMBED_MODEL", "nlpaueb/legal-bert-base-uncased")
        self.artifact_dir = Path(
            os.getenv("CASE_LAW_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "case_law_artifacts")
        )

        self.docs = []
        self.doc_map = {}
        self.tokens = []
        self.token_sets = []
        self.bm25 = None
        self.emb = None
        self.model = None

    def load(self):
        with open(self.artifact_dir / "docs.json", "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.doc_map = {
            d["case_id"]: d for d in self.docs if d.get("case_id")
        }

        with open(self.artifact_dir / "bm25.pkl", "rb") as f:
            payload = pickle.load(f)
            self.tokens = payload["tokens"]

        self.token_sets = [set(x) for x in self.tokens]
        self.bm25 = BM25Okapi(self.tokens)
        self.emb = np.load(self.artifact_dir / "embeddings.npy")
        self.model = SentenceTransformer(self.model_name)
        self.ready = True

    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.doc_map.get(case_id)

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_candidates: int = 80,
        alpha: float = 0.65,
        beta: float = 0.35,
        min_match_ratio: float = 0.30,
        min_semantic_cosine: float = 0.10,
    ) -> List[Dict[str, Any]]:

        if not self.ready:
            raise RuntimeError("Case law engine not loaded")

        q = clean_query(query)
        q_tokens = tokenize(q)
        if not q_tokens:
            return []

        q_set = set(q_tokens)
        required_hits = max(1, int(np.ceil(min_match_ratio * len(q_tokens))))

        bm25_scores = self.bm25.get_scores(q_tokens)

        candidates = []
        for i in range(len(self.docs)):
            b = float(bm25_scores[i])
            if b <= 0.0:
                continue
            overlap = len(q_set.intersection(self.token_sets[i]))
            if overlap < required_hits:
                continue
            candidates.append(i)

        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda i: float(bm25_scores[i]), reverse=True)[:bm25_candidates]

        q_emb = self.model.encode(q, convert_to_numpy=True, normalize_embeddings=True)
        bm25_arr = np.array([float(bm25_scores[i]) for i in candidates], dtype=float)
        cosine = self.emb[candidates] @ q_emb

        if bm25_arr.max() == bm25_arr.min():
            bm25_norm = np.ones_like(bm25_arr) if bm25_arr.max() > 0 else np.zeros_like(bm25_arr)
        else:
            bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())

        sem01 = (cosine + 1.0) / 2.0
        final = alpha * bm25_norm + beta * sem01

        out = []
        for j, i in enumerate(candidates):
            if cosine[j] < min_semantic_cosine:
                continue
            out.append({
                "doc": self.docs[i],
                "bm25": float(bm25_arr[j]),
                "semantic_cosine": float(cosine[j]),
                "score": float(final[j]),
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]