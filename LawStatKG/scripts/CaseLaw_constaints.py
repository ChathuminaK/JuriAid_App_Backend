import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        queries = [
            "CREATE CONSTRAINT case_law_source_id_unique IF NOT EXISTS FOR (n:CaseLawSource) REQUIRE n.source_id IS UNIQUE",
            "CREATE CONSTRAINT case_law_section_key_unique IF NOT EXISTS FOR (n:CaseLawSection) REQUIRE n.section_key IS UNIQUE",
            "CREATE CONSTRAINT case_law_id_unique IF NOT EXISTS FOR (n:CaseLaw) REQUIRE n.case_id IS UNIQUE",
            "CREATE CONSTRAINT case_law_topic_name_unique IF NOT EXISTS FOR (n:CaseLawTopic) REQUIRE n.name IS UNIQUE"
        ]
        for q in queries:
            session.run(q)
    driver.close()
    print("Case law constraints created.")

if __name__ == "__main__":
    main()