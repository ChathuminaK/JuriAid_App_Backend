import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def init_constraints():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        statements = [
            """
            CREATE CONSTRAINT act_id_unique IF NOT EXISTS
            FOR (a:Act) REQUIRE a.act_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT section_key_unique IF NOT EXISTS
            FOR (s:Section) REQUIRE s.key IS UNIQUE
            """,
            """
            CREATE CONSTRAINT section_version_id_unique IF NOT EXISTS
            FOR (sv:SectionVersion) REQUIRE sv.version_id IS UNIQUE
            """
        ]
        for q in statements:
            session.run(q)
    driver.close()
    print("Neo4j constraints initialized")


if __name__ == "__main__":
    init_constraints()


