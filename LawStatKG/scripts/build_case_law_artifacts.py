import os
import json
import pickle
import re
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


def clean_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            s = clean_text(str(x))
            if s:
                out.append(s)
        return out
    s = clean_text(str(value))
    return [s] if s else []


def tokenize(text: str):
    toks = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    stop = {
        "the", "and", "or", "to", "of", "in", "on", "for", "a", "an",
        "is", "are", "was", "were", "be", "by", "with", "as", "at", "from"
    }
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
                "source_id": clean_text(r["source_id"]),
                "source_title": clean_text(r["source_title"]),
                "chapter": clean_text(r["chapter"]),
            }

            for item in (r["section_cases"] or []):
                if item.get("case_id"):
                    docs.append({
                        **base,
                        "case_id": clean_text(item.get("case_id")),
                        "case_name": clean_text(item.get("case_name")),
                        "citation": clean_text(item.get("citation")),
                        "facts": clean_text(item.get("facts")),
                        "held": ensure_list(item.get("held")),
                        "principle": ensure_list(item.get("principle")),
                        "topic": clean_text(item.get("topic")),
                        "court": clean_text(item.get("court")),
                        "relevant_laws": ensure_list(item.get("relevant_laws")),
                        "relevant_sections": ensure_list(item.get("relevant_sections")),
                        "amending_law": clean_text(item.get("amending_law")),
                        "section_number": clean_text(item.get("section_number")),
                        "section_title": clean_text(item.get("section_title")),
                        "section_content": clean_text(item.get("section_content")),
                    })

            for item in (r["topic_cases"] or []):
                if item.get("case_id"):
                    docs.append({
                        **base,
                        "case_id": clean_text(item.get("case_id")),
                        "case_name": clean_text(item.get("case_name")),
                        "citation": clean_text(item.get("citation")),
                        "facts": clean_text(item.get("facts")),
                        "held": ensure_list(item.get("held")),
                        "principle": ensure_list(item.get("principle")),
                        "topic": clean_text(item.get("topic")),
                        "court": clean_text(item.get("court")),
                        "relevant_laws": ensure_list(item.get("relevant_laws")),
                        "relevant_sections": ensure_list(item.get("relevant_sections")),
                        "amending_law": clean_text(item.get("amending_law")),
                        "section_number": "",
                        "section_title": "",
                        "section_content": "",
                    })

    driver.close()

    # remove duplicates, but keep the richer version of each case_id
    seen = {}
    for d in docs:
        cid = d["case_id"]

        richness = (
            len(d.get("facts") or "") +
            len(" ".join(d.get("held") or [])) +
            len(" ".join(d.get("principle") or [])) +
            len(d.get("section_content") or "") +
            len(" ".join(d.get("relevant_laws") or [])) +
            len(" ".join(d.get("relevant_sections") or [])) +
            len(d.get("topic") or "")
        )

        if cid not in seen:
            seen[cid] = (richness, d)
        else:
            old_richness, _ = seen[cid]
            if richness > old_richness:
                seen[cid] = (richness, d)

    out = [v[1] for v in seen.values()]
    return out


def main():
    docs = fetch_case_law_docs()
    if not docs:
        raise RuntimeError("No case-law docs found in Neo4j.")

    texts = []
    cleaned_docs = []

    for d in docs:
        if not d.get("case_id") or not d.get("case_name"):
            continue

        topic_boost = " ".join([d.get("topic") or ""] * 3)
        case_name_boost = " ".join([d.get("case_name") or ""] * 2)
        section_title_boost = " ".join([d.get("section_title") or ""] * 2)
        principle_boost = " ".join((d.get("principle") or []) * 2)
        held_boost = " ".join((d.get("held") or []) * 2)
        relevant_laws_boost = " ".join((d.get("relevant_laws") or []) * 2)
        relevant_sections_boost = " ".join((d.get("relevant_sections") or []) * 2)

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
            d.get("court") or "",
            d.get("amending_law") or "",
            topic_boost,
            case_name_boost,
            section_title_boost,
            principle_boost,
            held_boost,
            relevant_laws_boost,
            relevant_sections_boost,
        ]).strip()

        blob = clean_text(blob)
        if not blob:
            continue

        d["_search_text"] = blob
        texts.append(blob)
        cleaned_docs.append(d)

    docs = cleaned_docs

    if not docs:
        raise RuntimeError("No usable case-law docs found after cleaning.")

    token_lists = [tokenize(t) for t in texts]

    # keep BM25 build for artifact compatibility and validation
    _ = BM25Okapi(token_lists)

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    with open(ART_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    with open(ART_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"tokens": token_lists}, f)

    np.save(ART_DIR / "embeddings.npy", emb)

    with open(ART_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(docs),
                "model": MODEL_NAME,
                "avg_tokens": round(sum(len(x) for x in token_lists) / max(1, len(token_lists)), 2),
                "embedding_dim": int(emb.shape[1]) if len(emb.shape) == 2 else None,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Case-law artifacts built successfully. Total docs: {len(docs)}")


if __name__ == "__main__":
    main()