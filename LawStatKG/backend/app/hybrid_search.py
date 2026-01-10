import os
import re
from datetime import date
from typing import List, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ENV + NEO4J CONNECTION

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Samidi123")



# 1) Cleaning + Tokenizer (STRICT)   fixes /n \n problem

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "they", "this", "to", "was", "were", "with", "you", "your"
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
    # STRICT: remove stopwords + tokens shorter than 3
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    return toks



# Temporal helpers

def today_str() -> str:
    return date.today().isoformat()

def temporal_ok(doc: Dict, as_of: str) -> bool:
    """
    doc['valid_from'] and doc['valid_to'] come from Neo4j as ISO strings or None.
    """
    vf = doc.get("valid_from")
    vt = doc.get("valid_to")
    if vf and vf > as_of:
        return False
    if vt and vt < as_of:
        return False
    return True



# Fetch sections directly from Neo4j KG

def load_sections_from_neo4j() -> List[Dict]:
    """
    Reads from KG:
      (a:Act)-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)

    IMPORTANT:
    - We return dates as strings using toString(date)
    - We also include Act metadata needed for act expansion
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    cypher = """
    MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
    RETURN
      sv.version_id            AS version_id,
      a.act_id                 AS act_id,
      a.law                    AS law,
      a.title                  AS act_title,
      a.jurisdiction           AS jurisdiction,

      sv.section_no            AS section_no,
      sv.title                 AS section_title,
      sv.text                  AS text,

      CASE WHEN sv.valid_from IS NULL THEN NULL ELSE toString(sv.valid_from) END AS valid_from,
      CASE WHEN sv.valid_to   IS NULL THEN NULL ELSE toString(sv.valid_to)   END AS valid_to,

      coalesce(sv.citations, [])      AS citations,
      coalesce(sv.amended_by, [])     AS amended_by,
      coalesce(sv.repealed_by, NULL)  AS repealed_by,
      coalesce(sv.current_status, "active") AS current_status
    ORDER BY a.act_id, sv.section_no
    """

    sections: List[Dict] = []
    with driver.session() as session:
        for record in session.run(cypher):
            sections.append({
                "version_id": record["version_id"],
                "act_id": record["act_id"],
                "law": record["law"],
                "act_title": record["act_title"],
                "jurisdiction": record["jurisdiction"],

                "section_no": record["section_no"],
                "section_title": record["section_title"],
                "text": record["text"] or "",

                "valid_from": record["valid_from"],
                "valid_to": record["valid_to"],

                "citations": record["citations"] or [],
                "amended_by": record["amended_by"] or [],
                "repealed_by": record["repealed_by"],
                "current_status": record["current_status"] or "active",
            })

    driver.close()
    return sections



# Build indexes from KG data (BM25 + maps + embeddings)

print("Loading sections directly from Neo4j Knowledge Graph...")
ALL_SECTIONS: List[Dict] = load_sections_from_neo4j()
print(f"Loaded {len(ALL_SECTIONS)} section versions from Neo4j")

print("Building BM25 index (sections)...")
SECTION_TEXTS = [(s.get("section_title", "") + " " + s.get("text", "")) for s in ALL_SECTIONS]
SECTION_TOKENS = [tokenize(t) for t in SECTION_TEXTS]
BM25 = BM25Okapi(SECTION_TOKENS)
SECTION_TOKEN_SETS = [set(t) for t in SECTION_TOKENS]
print("BM25 index ready")

# act_id -> section indexes
ACT_TO_SECTIONS: DefaultDict[str, List[int]] = defaultdict(list)
for i, s in enumerate(ALL_SECTIONS):
    ACT_TO_SECTIONS[s.get("act_id")].append(i)

# act metadata tokens (used for act expansion)
ACT_META_TOKENS: DefaultDict[str, set] = defaultdict(set)
for s in ALL_SECTIONS:
    act_id = s.get("act_id")
    meta = f"{s.get('act_id','')} {s.get('law','')} {s.get('act_title','')} {s.get('jurisdiction','')}"
    ACT_META_TOKENS[act_id].update(tokenize(meta))

print("Loading semantic model (LegalBERT)...")
MODEL = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
print("Model loaded")

print("Precomputing section embeddings (from KG text)...")
DOC_EMB = MODEL.encode(
    SECTION_TEXTS,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)
print("Embeddings ready")



# HYBRID STRICT SEARCH (BM25 gate + LegalBERT rerank)

def hybrid_strict_search(
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
    """
    STRICT rules:
    - If query tokens empty -> reject
    - If no BM25 candidates -> reject (out-of-dataset)
    - Act metadata match -> expand ALL sections from that Act, then hybrid rank
    - Otherwise: BM25 gate + token overlap + LegalBERT rerank
    """
    as_of_date = as_of_date or today_str()

    q_clean = clean_query(query)
    q_tokens = tokenize(q_clean)
    if not q_tokens:
        return []

    q_set = set(q_tokens)

    # BM25 scores
    bm25_scores = BM25.get_scores(q_tokens)

    # semantic query embedding
    q_emb = MODEL.encode(q_clean, convert_to_numpy=True, normalize_embeddings=True)

  
    # A) Act-level expansion when query matches Act metadata
   
    matching_acts = []
    for act_id, meta_set in ACT_META_TOKENS.items():
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
            for idx in ACT_TO_SECTIONS.get(act_id, []):
                doc = ALL_SECTIONS[idx]
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

        cosine = DOC_EMB[idxs] @ q_emb
        sem01 = (cosine + 1.0) / 2.0
        score = alpha * bm25_norm + beta * sem01

        results = []
        for j, idx in enumerate(idxs):
            # strict filter: avoid unrelated sections
            if cosine[j] < min_semantic_cosine and bm25_arr[j] <= 0.0:
                continue
            results.append({
                "doc": ALL_SECTIONS[idx],
                "bm25": float(bm25_arr[j]),
                "bm25_norm": float(bm25_norm[j]),
                "semantic_cosine": float(cosine[j]),
                "score": float(score[j]),
            })

        if not results:
            return []

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k] if top_k else results

   
    # B) Section-level strict BM25 gate + overlap

    required_hits = 1 if len(q_tokens) == 1 else max(1, int(np.ceil(min_match_ratio * len(q_tokens))))

    candidates = []
    for idx, doc in enumerate(ALL_SECTIONS):
        if jurisdiction and doc.get("jurisdiction") != jurisdiction:
            continue
        if not temporal_ok(doc, as_of_date):
            continue

        b = float(bm25_scores[idx])
        if b <= 0.0:
            continue

        overlap = len(q_set.intersection(SECTION_TOKEN_SETS[idx]))
        if overlap < required_hits:
            continue

        candidates.append(idx)

    # strict: out-of-dataset => reject
    if not candidates:
        return []

    candidates.sort(key=lambda i: float(bm25_scores[i]), reverse=True)
    candidates = candidates[: min(len(candidates), bm25_candidates)]

    bm25_arr = np.array([float(bm25_scores[i]) for i in candidates], dtype=float)
    if bm25_arr.max() == bm25_arr.min():
        bm25_norm = np.ones_like(bm25_arr) if bm25_arr.max() > 0 else np.zeros_like(bm25_arr)
    else:
        bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())

    cosine = DOC_EMB[candidates] @ q_emb
    sem01 = (cosine + 1.0) / 2.0
    score = alpha * bm25_norm + beta * sem01

    results = []
    for j, idx in enumerate(candidates):
        if cosine[j] < min_semantic_cosine:
            continue
        results.append({
            "doc": ALL_SECTIONS[idx],
            "bm25": float(bm25_arr[j]),
            "bm25_norm": float(bm25_norm[j]),
            "semantic_cosine": float(cosine[j]),
            "score": float(score[j]),
        })

    if not results:
        return []

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k] if top_k else results



# CLI helper

def parse_query(line: str) -> Tuple[str, Optional[str]]:
    line = line.strip()
    jur = None
    m = re.match(r"jurisdiction\s*:\s*([A-Za-z_]+)\s+(.*)$", line, flags=re.I)
    if m:
        jur = m.group(1)
        q = m.group(2).strip()
        return q, jur
    return line, None


#  Run interactively

if __name__ == "__main__":
    print("\n HYBRID STRICT Search THROUGH KG (Neo4j → BM25 → LegalBERT)")
    print("Type your query. Type 'exit' to quit.\n")

    while True:
        line = input("Enter query: ").strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break

        q, jur = parse_query(line)

        results = hybrid_strict_search(
            query=q,
            as_of_date=today_str(),
            jurisdiction=jur,
            top_k=0,
            bm25_candidates=80,
            alpha=0.65,
            beta=0.35,
            min_match_ratio=0.5,
            min_semantic_cosine=0.20,
        )

        q_show = clean_query(q)
        print(f"\nQuery       : {q_show}")
        print(f"As of date  : {today_str()}")
        print(f"Jurisdiction: {jur or 'ALL'}")

        if not results:
            print(" No results found! query not present in KG.\n")
            continue

        print(f"Top results : {len(results)}")

        for i, r in enumerate(results, start=1):
            doc = r["doc"]
            print("\n" + "=" * 80)
            print(
                f"[{i}] {doc.get('act_id')} ({doc.get('jurisdiction')}) "
                f"s.{doc.get('section_no')} - {doc.get('section_title')}"
            )
            print(f"    Hybrid score    : {r['score']:.4f}")
            print(f"    BM25 score      : {r['bm25']:.4f} (norm={r['bm25_norm']:.4f})")
            print(f"    Semantic cosine : {r['semantic_cosine']:.4f}")
            print(f"    Law             : {doc.get('law')}")
            print(f"    Act title       : {doc.get('act_title')}")
            print(f"    Version ID      : {doc.get('version_id')}")
            print(f"    Valid from      : {doc.get('valid_from')}")
            print(f"    Current status  : {doc.get('current_status')}")
            print(f"    Citations       : {doc.get('citations')}")
            print(f"    Amended by      : {doc.get('amended_by')}")
            print(f"    Repealed by     : {doc.get('repealed_by')}")

            text = doc.get("text", "")
            snippet = text[:300] + ("..." if len(text) > 300 else "")
            print(f"    Text            : {snippet}")

        print()