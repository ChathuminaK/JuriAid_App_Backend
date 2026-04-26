import os
import json
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv


# Load backend/.env reliably (project_root/backend/.env)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "backend" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

DATA_PATH = PROJECT_ROOT / "data" / "laws.json"


def load_laws():
    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise RuntimeError(
            f"Missing Neo4j env vars. Ensure {ENV_PATH} has NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"laws.json not found at: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        acts = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for act in acts:
            print(f"Loading act: {act.get('act_id')} - {act.get('law')}")
            session.execute_write(_create_act_with_sections, act)

    driver.close()
    print("Finished loading laws into Neo4j AuraDB.")


def _create_act_with_sections(tx, act: dict):
    tx.run(
        """
        MERGE (a:Act {act_id: $act_id})
        ON CREATE SET
            a.law = $law,
            a.title = $title,
            a.chapter_no = $chapter_no,
            a.jurisdiction = $jurisdiction,
            a.enactment_date = CASE WHEN $enactment_date IS NULL THEN NULL ELSE date($enactment_date) END,
            a.effective_date = CASE WHEN $effective_date IS NULL THEN NULL ELSE date($effective_date) END
        """,
        act_id=act["act_id"],
        law=act.get("law"),
        title=act.get("title"),
        chapter_no=act.get("chapter_no"),
        jurisdiction=act.get("jurisdiction"),
        enactment_date=act.get("enactment_date"),
        effective_date=act.get("effective_date"),
    )

    for sec in act.get("sections", []):
        section_key = f"{act['act_id']}::S{sec['section_no']}"

        tx.run(
            """
            MERGE (s:Section {key: $key})
            ON CREATE SET
                s.section_no = $section_no,
                s.act_id = $act_id
            WITH s
            MATCH (a:Act {act_id: $act_id})
            MERGE (a)-[:HAS_SECTION]->(s)
            """,
            key=section_key,
            section_no=sec["section_no"],
            act_id=act["act_id"],
        )

        tx.run(
            """
            MATCH (s:Section {key: $key})
            MERGE (sv:SectionVersion {version_id: $version_id})
            SET
                sv.section_no      = $section_no,
                sv.title           = $title,
                sv.text            = $text,
                sv.valid_from      = CASE WHEN $valid_from IS NULL THEN NULL ELSE date($valid_from) END,
                sv.valid_to        = CASE WHEN $valid_to   IS NULL THEN NULL ELSE date($valid_to)   END,
                sv.current_status  = $current_status,
                sv.citations       = $citations,
                sv.amended_by      = $amended_by,
                sv.repealed_by     = $repealed_by
            MERGE (s)-[:HAS_VERSION]->(sv)
            """,
            key=section_key,
            version_id=sec["version_id"],
            section_no=sec["section_no"],
            title=sec.get("title"),
            text=sec.get("text", ""),
            valid_from=sec.get("valid_from"),
            valid_to=sec.get("valid_to"),
            current_status=sec.get("current_status", "active"),
            citations=sec.get("citations", []),
            amended_by=sec.get("amended_by", []),
            repealed_by=sec.get("repealed_by"),
        )


if __name__ == "__main__":
    load_laws()
