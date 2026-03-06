import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from neo4j import GraphDatabase
from dotenv import load_dotenv


# -----------------------------
# ENV + PATHS (no code changes elsewhere)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "backend" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NEO4J_URI = (os.getenv("NEO4J_URI") or "").strip()
NEO4J_USER = (os.getenv("NEO4J_USER") or "").strip()
NEO4J_PASSWORD = (os.getenv("NEO4J_PASSWORD") or "").strip()

DATA_PATH = PROJECT_ROOT / "data" / "new_legal_documents.json"


# -----------------------------
# HELPERS
# -----------------------------
def slugify(text: str) -> str:
    """
    Create a stable act_id compatible with your system.
    Example: "Prevention of Domestic Violence Act" -> "prevention_of_domestic_violence_act"
    """
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def flatten_sections(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Supports both:
      - content.sections[]
      - content.chapters[].sections[]
    Returns list of {number,title,text}
    """
    content = doc.get("content") or {}

    out = []

    # direct sections
    sections = content.get("sections")
    if isinstance(sections, list):
        out.extend(sections)

    # chapters -> sections
    chapters = content.get("chapters")
    if isinstance(chapters, list):
        for ch in chapters:
            for s in ch.get("sections", []):
                out.append(s)

    # normalize keys
    cleaned = []
    for s in out:
        number = str(s.get("number", "")).strip()
        title = str(s.get("title", "")).strip()
        text = str(s.get("text", "")).strip()

        if not number:
            continue

        cleaned.append({"number": number, "title": title, "text": text})

    return cleaned


# -----------------------------
# NEO4J WRITE QUERIES
# -----------------------------
def upsert_act(tx, act_id: str, law: str, title: str, chapter_no: str, jurisdiction: str):
    tx.run(
        """
        MERGE (a:Act {act_id:$act_id})
        ON CREATE SET
          a.law = $law,
          a.title = $title,
          a.chapter_no = $chapter_no,
          a.jurisdiction = $jurisdiction
        ON MATCH SET
          a.law = coalesce(a.law, $law),
          a.title = coalesce(a.title, $title),
          a.chapter_no = coalesce(a.chapter_no, $chapter_no),
          a.jurisdiction = coalesce(a.jurisdiction, $jurisdiction)
        """,
        act_id=act_id,
        law=law,
        title=title,
        chapter_no=chapter_no,
        jurisdiction=jurisdiction,
    )


def upsert_section_and_version(
    tx,
    act_id: str,
    section_no: str,
    section_title: str,
    text: str,
    version_id: str,
):
    section_key = f"{act_id}::S{section_no}"

    # Section node + relation
    tx.run(
        """
        MERGE (s:Section {key:$key})
        ON CREATE SET s.section_no=$section_no, s.act_id=$act_id
        WITH s
        MATCH (a:Act {act_id:$act_id})
        MERGE (a)-[:HAS_SECTION]->(s)
        """,
        key=section_key,
        section_no=section_no,
        act_id=act_id,
    )

    # Version node + relation
    tx.run(
        """
        MATCH (s:Section {key:$key})
        MERGE (sv:SectionVersion {version_id:$version_id})
        SET
          sv.section_no = $section_no,
          sv.title = $title,
          sv.text = $text,
          sv.valid_from = NULL,
          sv.valid_to = NULL,
          sv.current_status = "active",
          sv.citations = [],
          sv.amended_by = [],
          sv.repealed_by = NULL
        MERGE (s)-[:HAS_VERSION]->(sv)
        """,
        key=section_key,
        version_id=version_id,
        section_no=section_no,
        title=section_title,
        text=text,
    )


# -----------------------------
# MAIN LOADER
# -----------------------------
def load_new_dataset():
    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise RuntimeError(f"Missing env vars in {ENV_PATH}: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    documents = raw.get("legal_documents", [])
    if not documents:
        raise RuntimeError("No legal_documents found in dataset JSON.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for doc in documents:
            act_title = (doc.get("title") or "").strip()
            if not act_title:
                print("⚠️ skipped doc without title")
                continue

            act_id = slugify(act_title)

            chapter_no = (doc.get("act_number") or "").strip()
            jurisdiction = (doc.get("jurisdiction") or "Sri Lanka").strip()

            # Use your system field names
            session.execute_write(
                upsert_act,
                act_id,
                act_title,
                act_title,
                chapter_no,
                jurisdiction,
            )

            sections = flatten_sections(doc)
            print(f"Loading: {act_id} ({len(sections)} sections)")

            for s in sections:
                sec_no = s["number"]
                sec_title = s["title"]
                sec_text = s["text"]

                # stable version id pattern consistent with your KG
                version_id = f"{act_id}-s{sec_no}-v1"

                session.execute_write(
                    upsert_section_and_version,
                    act_id,
                    sec_no,
                    sec_title,
                    sec_text,
                    version_id,
                )

    driver.close()
    print("✅ New dataset loaded into Neo4j AuraDB (without changing existing system code).")


if __name__ == "__main__":
    load_new_dataset()