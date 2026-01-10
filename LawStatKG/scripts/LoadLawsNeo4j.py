import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI","bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER","neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD","Samidi123")

DATA_PATH = os.path.join("data", "laws.json")


def load_laws():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        acts = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for act in acts:
            print(f" Loading act: {act['act_id']} - {act['law']}")
            session.execute_write(_create_act_with_sections, act)

    driver.close()
    print("Finished loading laws into Neo4j")


def _create_act_with_sections(tx, act: dict):
    # Create Act node
    tx.run(
        """
        MERGE (a:Act {act_id: $act_id})
        ON CREATE SET
            a.law = $law,
            a.title = $title,
            a.chapter_no = $chapter_no,
            a.jurisdiction = $jurisdiction,
            a.enactment_date = date($enactment_date),
            a.effective_date = date($effective_date)
        """,
        act_id=act["act_id"],
        law=act["law"],
        title=act["title"],
        chapter_no=act["chapter_no"],
        jurisdiction=act["jurisdiction"],
        enactment_date=act["enactment_date"],
        effective_date=act["effective_date"],
    )

    # Create Section & SectionVersion
    for sec in act["sections"]:
        section_key = f"{act['act_id']}::S{sec['section_no']}"
        
        citations = sec.get("citations", [])
        amended_by = sec.get("amended_by", [])
        repealed_by = sec.get("repealed_by")
        valid_to = sec.get("valid_to")       # can be None
        current_status = sec.get("current_status", "active")
            
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
       # One current SectionVersion (we’ll add amendments later)
        tx.run(
            """
            MATCH (s:Section {key: $key})
            MERGE (sv:SectionVersion {version_id: $version_id})
            SET
                sv.section_no      = $section_no,
                sv.title           = $title,
                sv.text            = $text,
                sv.valid_from      = date($valid_from),
                sv.valid_to        = CASE
                                        WHEN $valid_to IS NULL THEN NULL
                                        ELSE date($valid_to)
                                     END,
                sv.current_status  = $current_status,
                sv.citations       = $citations,
                sv.amended_by      = $amended_by,
                sv.repealed_by     = $repealed_by
            MERGE (s)-[:HAS_VERSION]->(sv)
            """,
            key=section_key,
            version_id=sec["version_id"],
            section_no=sec["section_no"],
            text=sec["text"],
            title=sec["title"],
            valid_from=sec["valid_from"],
            valid_to=sec["valid_to"],
            current_status=sec["current_status"],
            citations=sec.get("citations", []),
            amended_by=sec.get("amended_by", []),
            repealed_by=sec.get("repealed_by",[]),
        )


if __name__ == "__main__":
    load_laws()
