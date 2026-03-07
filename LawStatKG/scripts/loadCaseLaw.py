import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def norm_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [str(v).strip()]


def make_case_id(prefix: str, case_name: str, citation: str) -> str:
    raw = f"{prefix}|{case_name}|{citation}".encode("utf-8")
    h = hashlib.md5(raw).hexdigest()[:12]
    return f"{prefix}_{h}"


def upsert_source(tx, source_id: str, title: str, chapter: str):
    tx.run(
        """
        MERGE (s:CaseLawSource {source_id:$source_id})
        SET s.title = $title,
            s.chapter = $chapter
        """,
        source_id=source_id,
        title=title,
        chapter=chapter,
    )


def upsert_section(tx, source_id: str, section: Dict[str, Any]):
    section_key = f"{source_id}::S{section['section_number']}"
    tx.run(
        """
        MERGE (sec:CaseLawSection {section_key:$section_key})
        SET sec.section_number = $section_number,
            sec.title = $title,
            sec.content = $content
        WITH sec
        MATCH (src:CaseLawSource {source_id:$source_id})
        MERGE (src)-[:HAS_SECTION]->(sec)
        """,
        section_key=section_key,
        section_number=section["section_number"],
        title=section.get("title"),
        content=section.get("content", ""),
        source_id=source_id,
    )


def upsert_topic(tx, topic: str):
    tx.run(
        """
        MERGE (t:CaseLawTopic {name:$topic})
        """,
        topic=topic,
    )


def upsert_case_law_under_section(tx, source_id: str, section_number: str, case_item: Dict[str, Any]):
    prefix = f"{source_id}_s{section_number}"
    case_id = make_case_id(prefix, case_item.get("case_name", ""), case_item.get("citation", ""))

    topic = case_item.get("topic", "General")
    held = norm_list(case_item.get("held"))
    principle = norm_list(case_item.get("principle"))
    relevant_laws = norm_list(case_item.get("relevant_laws"))
    relevant_sections = norm_list(case_item.get("relevant_sections"))
    if case_item.get("relevant_section"):
        relevant_sections.extend(norm_list(case_item.get("relevant_section")))

    tx.run(
        """
        MERGE (c:CaseLaw {case_id:$case_id})
        SET c.case_name = $case_name,
            c.citation = $citation,
            c.facts = $facts,
            c.held = $held,
            c.principle = $principle,
            c.topic = $topic,
            c.court = $court,
            c.relevant_laws = $relevant_laws,
            c.relevant_sections = $relevant_sections,
            c.amending_law = $amending_law
        WITH c
        MATCH (sec:CaseLawSection {section_key:$section_key})
        MERGE (sec)-[:HAS_CASE_LAW]->(c)
        """,
        case_id=case_id,
        case_name=case_item.get("case_name"),
        citation=case_item.get("citation"),
        facts=case_item.get("facts", ""),
        held=held,
        principle=principle,
        topic=topic,
        court=case_item.get("court", ""),
        relevant_laws=relevant_laws,
        relevant_sections=relevant_sections,
        amending_law=case_item.get("amending_law", ""),
        section_key=f"{source_id}::S{section_number}",
    )

    upsert_topic(tx, topic)
    tx.run(
        """
        MATCH (c:CaseLaw {case_id:$case_id})
        MATCH (t:CaseLawTopic {name:$topic})
        MERGE (c)-[:HAS_TOPIC]->(t)
        """,
        case_id=case_id,
        topic=topic,
    )


def upsert_topic_case_law(tx, source_id: str, topic_name: str, case_item: Dict[str, Any]):
    prefix = f"{source_id}_topic_{topic_name}"
    case_id = make_case_id(prefix, case_item.get("case_name", ""), case_item.get("citation", ""))

    held = norm_list(case_item.get("held"))
    principle = norm_list(case_item.get("principle"))
    relevant_laws = norm_list(case_item.get("relevant_laws"))
    relevant_sections = norm_list(case_item.get("relevant_sections"))
    if case_item.get("relevant_section"):
        relevant_sections.extend(norm_list(case_item.get("relevant_section")))

    tx.run(
        """
        MERGE (c:CaseLaw {case_id:$case_id})
        SET c.case_name = $case_name,
            c.citation = $citation,
            c.facts = $facts,
            c.held = $held,
            c.principle = $principle,
            c.topic = $topic,
            c.court = $court,
            c.relevant_laws = $relevant_laws,
            c.relevant_sections = $relevant_sections,
            c.amending_law = $amending_law
        WITH c
        MATCH (src:CaseLawSource {source_id:$source_id})
        MERGE (src)-[:HAS_TOPIC_CASE_LAW]->(c)
        """,
        case_id=case_id,
        case_name=case_item.get("case_name"),
        citation=case_item.get("citation"),
        facts=case_item.get("facts", ""),
        held=held,
        principle=principle,
        topic=topic_name,
        court=case_item.get("court", ""),
        relevant_laws=relevant_laws,
        relevant_sections=relevant_sections,
        amending_law=case_item.get("amending_law", ""),
        source_id=source_id,
    )

    upsert_topic(tx, topic_name)
    tx.run(
        """
        MATCH (c:CaseLaw {case_id:$case_id})
        MATCH (t:CaseLawTopic {name:$topic})
        MERGE (c)-[:HAS_TOPIC]->(t)
        """,
        case_id=case_id,
        topic=topic_name,
    )


def main():
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python scripts/load_case_law_dataset.py <json_path>")

    json_path = sys.argv[1]
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    root_key = next(iter(raw.keys()))
    data = raw[root_key]

    source_id = root_key
    title = "Civil Procedure Code Matrimonial Actions Case Law Corpus"
    chapter = data.get("chapter", "")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        session.execute_write(upsert_source, source_id, title, chapter)

        for sec in data.get("sections", []):
            session.execute_write(upsert_section, source_id, sec)
            for case_item in sec.get("case_laws", []):
                session.execute_write(
                    upsert_case_law_under_section,
                    source_id,
                    sec["section_number"],
                    case_item
                )

        for topic_name, topic_cases in data.get("case_laws_by_topic", {}).items():
            for case_item in topic_cases:
                session.execute_write(upsert_topic_case_law, source_id, topic_name, case_item)

    driver.close()
    print("Case law dataset loaded into Neo4j successfully.")

if __name__ == "__main__":
    main()