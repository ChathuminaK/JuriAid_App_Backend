import os
from pathlib import Path
from difflib import unified_diff
from neo4j import GraphDatabase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def build_links():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:

        rows = session.run("""
        MATCH (s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
        WITH s, sv
        ORDER BY sv.valid_from ASC
        WITH s, collect(sv) AS versions
        RETURN s.key AS section_key, versions
        """).data()

        for row in rows:
            versions = row["versions"]
            if len(versions) < 2:
                continue

            for i in range(len(versions)-1):
                before = versions[i]
                after = versions[i+1]

                diff = "\n".join(unified_diff(
                    (before.get("text") or "").splitlines(),
                    (after.get("text") or "").splitlines(),
                    lineterm=""
                ))

                added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
                removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))

                session.run("""
                MATCH (b:SectionVersion {version_id:$before_id})
                MATCH (a:SectionVersion {version_id:$after_id})
                MERGE (b)-[r:NEXT_VERSION]->(a)
                SET r.change_date = a.valid_from,
                    r.diff = $diff,
                    r.summary = $summary,
                    r.added = $added,
                    r.removed = $removed
                """,
                before_id=before["version_id"],
                after_id=after["version_id"],
                diff=diff[:12000],
                summary=f"Added: {added}, Removed: {removed}",
                added=added,
                removed=removed
                )

    driver.close()
    print("Timeline links built successfully.")


if __name__ == "__main__":
    build_links()
