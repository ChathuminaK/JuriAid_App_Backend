import os
import re
import json
import pickle
from pathlib import Path
from datetime import date
from typing import List, Dict, Optional, DefaultDict
from collections import defaultdict
import hashlib
import numpy as np

# To this (lazy load = only loads when first request comes):
_embeddings = None
_meta = None
_sections = None

def get_artifacts():
    """✅ Lazy load search artifacts"""
    global _embeddings, _meta, _sections
    if _embeddings is None:
        print("⏳ Loading search artifacts...")
        base = os.path.dirname(os.path.dirname(__file__))
        _embeddings = np.load(os.path.join(base, "artifacts", "embeddings.npy"))
        with open(os.path.join(base, "artifacts", "meta.json")) as f:
            _meta = json.load(f)
        with open(os.path.join(base, "artifacts", "sections.json")) as f:
            _sections = json.load(f)
        print("✅ Artifacts loaded")
    return _embeddings, _meta, _sections

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # Much smaller model ~90MB instead of ~500MB
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

from app.kg_client import KGClient


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","in","is","it",
    "of","on","or","that","the","their","they","this","to","was","were","with","you","your"
}

def clean_query(q: str) -> str:
    q = (q or "").strip()
    q = q.replace("\\n", " ").replace("/n", " ").replace("\n", " ").replace("\r", " ")
    q = re.sub(r"[^A-Za-z0-9\s']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = _TOKEN_RE.findall(text.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    return toks

def today_str() -> str:
    return date.today().isoformat()

def temporal_ok(doc: Dict, as_of: str) -> bool:
    vf = doc.get("valid_from")
    vt = doc.get("valid_to")
    if vf and vf > as_of:
        return False
    if vt and vt < as_of:
        return False
    return True


class HybridSearchEngine:
    """
    Production-ready engine:
    - Can BUILD artifacts (offline script or dev mode)
    - Can LOAD artifacts fast (production)
    """
    def __init__(self):
        self.ready = False
        self.model = None

        self.sections: List[Dict] = []
        self.section_texts: List[str] = []
        self.section_tokens: List[List[str]] = []
        self.section_token_sets: List[set] = []

        self.bm25 = None
        self.doc_emb = None

        self.act_to_sections: DefaultDict[str, List[int]] = defaultdict(list)
        self.act_meta_tokens: DefaultDict[str, set] = defaultdict(set)

        # artifacts path
        self.artifact_dir = Path(os.getenv("ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # choose model (keep your research choice by default)
        self.model_name = os.getenv("EMBED_MODEL", "nlpaueb/legal-bert-base-uncased")

    # -----------------------------
    # Neo4j load
    # -----------------------------
    def _load_sections_from_neo4j(self, kg: KGClient) -> List[Dict]:
        cypher = """
        MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
        RETURN
          sv.version_id AS version_id,
          a.act_id AS act_id,
          a.law AS law,
          a.title AS act_title,
          a.jurisdiction AS jurisdiction,
          sv.section_no AS section_no,
          sv.title AS section_title,
          sv.text AS text,
          CASE WHEN sv.valid_from IS NULL THEN NULL ELSE toString(sv.valid_from) END AS valid_from,
          CASE WHEN sv.valid_to IS NULL THEN NULL ELSE toString(sv.valid_to) END AS valid_to,
          coalesce(sv.citations, []) AS citations,
          coalesce(sv.amended_by, []) AS amended_by,
          coalesce(sv.repealed_by, NULL) AS repealed_by,
          coalesce(sv.current_status, "active") AS current_status
        ORDER BY a.act_id, sv.section_no
        """
        out = []
        with kg.driver.session() as session:
            for r in session.run(cypher):
                out.append({
                    "version_id": r["version_id"],
                    "act_id": r["act_id"],
                    "law": r["law"],
                    "act_title": r["act_title"],
                    "jurisdiction": r["jurisdiction"],
                    "section_no": (r["section_no"] or "").strip(),
                    "section_title": r["section_title"],
                    "text": r["text"] or "",
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "citations": r["citations"] or [],
                    "amended_by": r["amended_by"] or [],
                    "repealed_by": r["repealed_by"],
                    "current_status": r["current_status"] or "active",
                })
        return out

    def _fingerprint_sections(self, sections: List[Dict]) -> str:
        """
        Stable-ish fingerprint so you can detect changes.
        Uses version_id + valid_from + valid_to + hash(text).
        """
        h = hashlib.sha256()
        for s in sections:
            h.update((s.get("version_id") or "").encode("utf-8"))
            h.update((s.get("valid_from") or "").encode("utf-8"))
            h.update((s.get("valid_to") or "").encode("utf-8"))
            txt = (s.get("section_title","") + " " + s.get("text","")).encode("utf-8")
            h.update(hashlib.md5(txt).digest())
        return h.hexdigest()

    # -----------------------------
    # Artifact paths
    # -----------------------------
    def _p_sections(self): return self.artifact_dir / "sections.json"
    def _p_bm25(self): return self.artifact_dir / "bm25.pkl"
    def _p_emb(self): return self.artifact_dir / "embeddings.npy"
    def _p_meta(self): return self.artifact_dir / "meta.json"

    def artifacts_exist(self) -> bool:
        return self._p_sections().exists() and self._p_bm25().exists() and self._p_emb().exists() and self._p_meta().exists()

    # -----------------------------
    # Build artifacts (slow)
    # -----------------------------
    def build_and_save_artifacts(self):
        kg = KGClient()
        if not kg.ping():
            raise RuntimeError("Neo4j is not reachable (Aura). Check env credentials/network.")
        sections = self._load_sections_from_neo4j(kg)
        kg.close()

        if not sections:
            raise RuntimeError("No sections loaded from Neo4j. Check your graph data.")

        section_texts = [(s.get("section_title", "") + " " + s.get("text", "")) for s in sections]
        section_tokens = [tokenize(t) for t in section_texts]
        bm25 = BM25Okapi(section_tokens)

        model = get_model()
        emb = model.encode(
            section_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # save sections.json
        with open(self._p_sections(), "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False)

        # save bm25.pkl (store token lists; rebuild BM25 quickly on load)
        with open(self._p_bm25(), "wb") as f:
            pickle.dump({"section_tokens": section_tokens}, f)

        # save embeddings
        np.save(self._p_emb(), emb)

        # meta
        meta = {
            "model_name": self.model_name,
            "count": len(sections),
            "fingerprint": self._fingerprint_sections(sections),
            "built_on": today_str()
        }
        with open(self._p_meta(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # Load artifacts (fast)
    # -----------------------------
    def load(self, allow_build: bool = False):
        """
        Production:
          allow_build=False and artifacts must exist.
        Dev:
          allow_build=True builds if missing.
        """
        if not self.artifacts_exist():
            if not allow_build:
                raise RuntimeError(
                    f"Search artifacts missing in {self.artifact_dir}. "
                    f"Run scripts/build_search_artifacts.py first."
                )
            self.build_and_save_artifacts()

        # load sections
        with open(self._p_sections(), "r", encoding="utf-8") as f:
            self.sections = json.load(f)

        self.section_texts = [(s.get("section_title", "") + " " + s.get("text", "")) for s in self.sections]

        # tokens
        with open(self._p_bm25(), "rb") as f:
            payload = pickle.load(f)
        self.section_tokens = payload["section_tokens"]
        self.section_token_sets = [set(t) for t in self.section_tokens]
        self.bm25 = BM25Okapi(self.section_tokens)

        # embeddings
        self.doc_emb = np.load(self._p_emb())

        # model for query embeddings
        # (this is much faster than embedding the whole corpus)
        self.model = get_model()

        # act expansion maps
        self.act_to_sections.clear()
        for i, s in enumerate(self.sections):
            self.act_to_sections[s.get("act_id")].append(i)

        self.act_meta_tokens.clear()
        for s in self.sections:
            act_id = s.get("act_id")
            meta = f"{s.get('act_id','')} {s.get('law','')} {s.get('act_title','')} {s.get('jurisdiction','')}"
            self.act_meta_tokens[act_id].update(tokenize(meta))

        self.ready = True

    # -----------------------------
    # Search (same as your logic)
    # -----------------------------
    def search(
        self,
        query: str,
        as_of_date: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        top_k: int = 10,
        bm25_candidates: int = 80,
        alpha: float = 0.65,
        beta: float = 0.35,
        min_match_ratio: float = 0.5,
        min_semantic_cosine: float = 0.20,
    ) -> List[Dict]:

        if not self.ready:
            raise RuntimeError("Search engine not loaded")

        as_of_date = as_of_date or today_str()

        q_clean = clean_query(query)
        q_tokens = tokenize(q_clean)
        if not q_tokens:
            return []

        q_set = set(q_tokens)
        bm25_scores = self.bm25.get_scores(q_tokens)
        q_emb = self.model.encode(q_clean, convert_to_numpy=True, normalize_embeddings=True)

        # ACT expansion
        matching_acts = []
        for act_id, meta_set in self.act_meta_tokens.items():
            if len(q_tokens) == 1:
                if q_tokens[0] in meta_set:
                    matching_acts.append(act_id)
            else:
                overlap = len(q_set.intersection(meta_set))
                if overlap / len(q_set) >= 0.6:
                    matching_acts.append(act_id)

        if matching_acts:
            idxs = []
            for act_id in matching_acts:
                for idx in self.act_to_sections.get(act_id, []):
                    doc = self.sections[idx]
                    if jurisdiction and doc.get("jurisdiction") != jurisdiction:
                        continue
                    if not temporal_ok(doc, as_of_date):
                        continue
                    idxs.append(idx)

            if not idxs:
                return []

            idxs = sorted(set(idxs))
            bm25_arr = np.array([float(bm25_scores[i]) for i in idxs], dtype=float)

            if bm25_arr.max() == bm25_arr.min():
                bm25_norm = np.ones_like(bm25_arr) if bm25_arr.max() > 0 else np.zeros_like(bm25_arr)
            else:
                bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())

            cosine = self.doc_emb[idxs] @ q_emb
            sem01 = (cosine + 1.0) / 2.0
            score = alpha * bm25_norm + beta * sem01

            results = []
            for j, idx in enumerate(idxs):
                if cosine[j] < min_semantic_cosine and bm25_arr[j] <= 0.0:
                    continue
                results.append({
                    "doc": self.sections[idx],
                    "bm25": float(bm25_arr[j]),
                    "bm25_norm": float(bm25_norm[j]),
                    "semantic_cosine": float(cosine[j]),
                    "score": float(score[j]),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k] if top_k else results

        # Strict BM25 gate + overlap
        required_hits = 1 if len(q_tokens) == 1 else max(1, int(np.ceil(min_match_ratio * len(q_tokens))))

        candidates = []
        for idx, doc in enumerate(self.sections):
            if jurisdiction and doc.get("jurisdiction") != jurisdiction:
                continue
            if not temporal_ok(doc, as_of_date):
                continue

            b = float(bm25_scores[idx])
            if b <= 0.0:
                continue

            overlap = len(q_set.intersection(self.section_token_sets[idx]))
            if overlap < required_hits:
                continue

            candidates.append(idx)

        if not candidates:
            return []

        candidates.sort(key=lambda i: float(bm25_scores[i]), reverse=True)
        candidates = candidates[: min(len(candidates), bm25_candidates)]

        bm25_arr = np.array([float(bm25_scores[i]) for i in candidates], dtype=float)
        if bm25_arr.max() == bm25_arr.min():
            bm25_norm = np.ones_like(bm25_arr) if bm25_arr.max() > 0 else np.zeros_like(bm25_arr)
        else:
            bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())

        cosine = self.doc_emb[candidates] @ q_emb
        sem01 = (cosine + 1.0) / 2.0
        score = alpha * bm25_norm + beta * sem01

        results = []
        for j, idx in enumerate(candidates):
            if cosine[j] < min_semantic_cosine:
                continue
            results.append({
                "doc": self.sections[idx],
                "bm25": float(bm25_arr[j]),
                "bm25_norm": float(bm25_norm[j]),
                "semantic_cosine": float(cosine[j]),
                "score": float(score[j]),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k] if top_k else results