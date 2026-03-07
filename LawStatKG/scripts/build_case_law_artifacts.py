import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MODEL_NAME = os.getenv("EMBED_MODEL", "nlpaueb/legal-bert-base-uncased")

ART_DIR = PROJECT_ROOT / "backend" / "case_law_artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text: str):
    import re
    toks = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    stop = {"the","and","or","to","of","in","on","for","a","an","is","are","was","were","be","by","with","as","at","from"}
    return [t for t in toks if t not in stop and len(t) >= 3]


def fetch_case_law_docs() -> List[Dict[str, Any]]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cypher = """
    MATCH (src:CaseLawSource)
    OPTIONAL MATCH (src)-[:HAS_SECTION]->(sec:CaseLawSection)-[:HAS_CASE_LAW]->(c:CaseLaw)
    OPTIONAL MATCH (src)-[:HAS_TOPIC_CASE_LAW]->(tc:CaseLaw)

    WITH src,
         collect(DISTINCT {
            case_id: c.case_id,
            case_name: c.case_name,
            citation: c.citation,
            facts: c.facts,
            held: c.held,
            principle: c.principle,
            topic: c.topic,
            court: c.court,
            relevant_laws: c.relevant_laws,
            relevant_sections: c.relevant_sections,
            amending_law: c.amending_law,
            section_number: sec.section_number,
            section_title: sec.title,
            section_content: sec.content
         }) AS section_cases,
         collect(DISTINCT {
            case_id: tc.case_id,
            case_name: tc.case_name,
            citation: tc.citation,
            facts: tc.facts,
            held: tc.held,
            principle: tc.principle,
            topic: tc.topic,
            court: tc.court,
            relevant_laws: tc.relevant_laws,
            relevant_sections: tc.relevant_sections,
            amending_law: tc.amending_law,
            section_number: NULL,
            section_title: NULL,
            section_content: NULL
         }) AS topic_cases

    RETURN src.source_id AS source_id,
           src.title AS source_title,
           src.chapter AS chapter,
           section_cases,
           topic_cases
    """
    docs = []
    with driver.session() as session:
        rows = session.run(cypher)
        for r in rows:
            base = {
                "source_id": r["source_id"],
                "source_title": r["source_title"],
                "chapter": r["chapter"],
            }
            for item in (r["section_cases"] or []):
                if item.get("case_id"):
                    docs.append({**base, **item})
            for item in (r["topic_cases"] or []):
                if item.get("case_id"):
                    docs.append({**base, **item})
    driver.close()

    # remove duplicates
    seen = set()
    out = []
    for d in docs:
        cid = d["case_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out


def main():
    docs = fetch_case_law_docs()
    if not docs:
        raise RuntimeError("No case-law docs found in Neo4j.")

    texts = []
    for d in docs:
        blob = " ".join([
            d.get("source_title") or "",
            d.get("chapter") or "",
            d.get("section_number") or "",
            d.get("section_title") or "",
            d.get("section_content") or "",
            d.get("case_name") or "",
            d.get("citation") or "",
            d.get("facts") or "",
            " ".join(d.get("held") or []),
            " ".join(d.get("principle") or []),
            d.get("topic") or "",
            " ".join(d.get("relevant_laws") or []),
            " ".join(d.get("relevant_sections") or []),
        ]).strip()
        texts.append(blob)

    token_lists = [tokenize(t) for t in texts]
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    with open(ART_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    with open(ART_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"tokens": token_lists}, f)

    np.save(ART_DIR / "embeddings.npy", emb)

    with open(ART_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(docs), "model": MODEL_NAME}, f, ensure_ascii=False, indent=2)

    print(f"Case-law artifacts built successfully. Total docs: {len(docs)}")

if __name__ == "__main__":
    main()