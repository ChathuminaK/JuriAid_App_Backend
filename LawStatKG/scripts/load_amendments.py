import os
import json
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

AMENDMENTS_PATH = PROJECT_ROOT / "data" / "amendments.json"


def load_amendments():
    if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
        raise RuntimeError("Missing Neo4j env vars in backend/.env")

    if not AMENDMENTS_PATH.exists():
        raise FileNotFoundError(f"Missing: {AMENDMENTS_PATH}")

    with open(AMENDMENTS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    amendments = payload.get("amendments", [])
    if not amendments:
        print("No amendments found in amendments.json")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for am in amendments:
            session.execute_write(_upsert_and_link, am)
            print(f"Loaded + linked: {am.get('amend_id')}")

    driver.close()
    print("Done: Amendments loaded and linked.")


def _upsert_and_link(tx, am: dict):
    amend_id = (am.get("amend_id") or "").strip()
    act_id = (am.get("act_id") or "").strip()
    section_no = str(am.get("section_no") or "").strip()
    date_str = (am.get("date") or "").strip()

    if not amend_id or not act_id or not section_no or not date_str:
        # Skip broken rows safely
        return

    # 1) Create/Update Amendment node
    tx.run(
        """
        MERGE (am:Amendment {amend_id:$amend_id})
        SET am.date = date($date),
            am.am_title = $am_title,
            am.summary = $summary,
            am.section_no = $section_no,
            am.section_title = $section_title,
            am.act_id = $act_id,
            am.jurisdiction = $jurisdiction
        """,
        amend_id=amend_id,
        date=date_str,
        am_title=am.get("am_title"),
        summary=am.get("summary"),
        section_no=section_no,
        section_title=am.get("section_title"),
        act_id=act_id,
        jurisdiction=am.get("jurisdiction"),
    )

    # 2) Link to the closest SectionVersion by date (robust)
    # It finds the version in that section with valid_from closest to amendment date.
    tx.run(
        """
        MATCH (a:Act {act_id:$act_id})-[:HAS_SECTION]->(s:Section)
        WHERE trim(s.section_no) = trim($section_no) OR s.key ENDS WITH ("::S" + trim($section_no))

        MATCH (s)-[:HAS_VERSION]->(v:SectionVersion)
        WHERE v.valid_from IS NOT NULL

        WITH v, abs(duration.inDays(v.valid_from, date($date)).days) AS delta
        ORDER BY delta ASC
        LIMIT 1

        MATCH (am:Amendment {amend_id:$amend_id})
        MERGE (v)-[:CHANGED_BY]->(am)
        """,
        act_id=act_id,
        section_no=section_no,
        date=date_str,
        amend_id=amend_id,
    )


if __name__ == "__main__":
    load_amendments()
