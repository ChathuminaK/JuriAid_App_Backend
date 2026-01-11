import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class KGClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

        # ✅ THIS LINE was missing in your code
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    # -----------------------------
    # 1) /statute/{act_id}?date=today or YYYY-MM-DD
    # -----------------------------
    def get_statute_as_of(self, act_id: str, as_of_date: str) -> Dict[str, Any]:
        cypher = """
        MATCH (a:Act {act_id:$act_id})-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
        WHERE (sv.valid_from IS NULL OR sv.valid_from <= date($as_of_date))
          AND (sv.valid_to IS NULL OR sv.valid_to >= date($as_of_date))
        RETURN
          a.act_id AS act_id,
          a.law AS law,
          a.title AS act_title,
          a.jurisdiction AS jurisdiction,
          a.chapter_no AS chapter_no,
          CASE WHEN a.enactment_date IS NULL THEN NULL ELSE toString(a.enactment_date) END AS enactment_date,
          CASE WHEN a.effective_date IS NULL THEN NULL ELSE toString(a.effective_date) END AS effective_date,
          collect({
              version_id: sv.version_id,
              section_no: sv.section_no,
              section_title: sv.title,
              text: sv.text,
              valid_from: CASE WHEN sv.valid_from IS NULL THEN NULL ELSE toString(sv.valid_from) END,
              valid_to: CASE WHEN sv.valid_to IS NULL THEN NULL ELSE toString(sv.valid_to) END,
              current_status: coalesce(sv.current_status, "active"),
              citations: coalesce(sv.citations, []),
              amended_by: coalesce(sv.amended_by, []),
              repealed_by: coalesce(sv.repealed_by, NULL)
          }) AS sections
        """

        with self.driver.session() as session:
            rec = session.run(cypher, act_id=act_id, as_of_date=as_of_date).single()

        if not rec:
            return {"error": "Act not found", "act_id": act_id, "as_of_date": as_of_date}

        return dict(rec)

    # -----------------------------
    # 2) /graph/{act_id}  (basic KG view)
    # -----------------------------
    def get_act_graph(self, act_id: str, limit_sections: int = 50) -> Dict[str, Any]:
        cypher = """
        MATCH (a:Act {act_id:$act_id})-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
        RETURN
          a.act_id AS act_id,
          a.law AS law,
          a.title AS act_title,
          a.jurisdiction AS jurisdiction,
          collect({
            section_key: s.key,
            section_no: s.section_no,
            version_id: sv.version_id,
            section_title: sv.title
          })[0..$limit_sections] AS section_nodes
        """

        with self.driver.session() as session:
            rec = session.run(cypher, act_id=act_id, limit_sections=limit_sections).single()

        if not rec:
            return {"error": "Act not found", "act_id": act_id}

        return dict(rec)