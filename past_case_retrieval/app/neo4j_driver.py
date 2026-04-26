from neo4j import GraphDatabase
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=300
        )

    def query(self, query, parameters=None):
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]


db = Neo4jConnection()