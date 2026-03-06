import os
from typing import Any, Dict, List
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

# Load environment variables
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)


class KGClient:
    def __init__(self):
        self.uri = (os.getenv("NEO4J_URI") or "").strip()
        self.user = (os.getenv("NEO4J_USER") or "").strip()
        self.password = (os.getenv("NEO4J_PASSWORD") or "").strip()

        if not self.uri or not self.user or not self.password:
            raise RuntimeError(
                "Missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env"
            )

        # Aura uses neo4j+s://
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=20,
            max_connection_pool_size=50,
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    # --------------------------------------------------
    # STATUTE AS-OF DATE
    # --------------------------------------------------
    def get_statute_as_of(self, act_id: str, as_of_date: str) -> Dict[str, Any]:
        act_id = act_id.strip()
        as_of_date = as_of_date.strip()

        cypher = """
        MATCH (a:Act {act_id:$act_id})-[:HAS_SECTION]->(s:Section)-[:HAS_VERSION]->(sv:SectionVersion)
        WHERE (sv.valid_from IS NULL OR sv.valid_from <= date($as_of_date))
          AND (sv.valid_to IS NULL OR sv.valid_to >= date($as_of_date))
        RETURN
          a.act_id AS act_id,
          a.title AS act_title,
          a.jurisdiction AS jurisdiction,
          collect({
              version_id: sv.version_id,
              section_no: sv.section_no,
              section_title: sv.title,
              text: sv.text,
              valid_from: CASE WHEN sv.valid_from IS NULL THEN NULL ELSE toString(sv.valid_from) END,
              valid_to: CASE WHEN sv.valid_to IS NULL THEN NULL ELSE toString(sv.valid_to) END,
              current_status: coalesce(sv.current_status, "active")
          }) AS sections
        """

        try:
            with self.driver.session() as session:
                rec = session.run(
                    cypher,
                    act_id=act_id,
                    as_of_date=as_of_date
                ).single()

            if not rec:
                return {"error": "Act not found", "act_id": act_id}

            return dict(rec)

        except Neo4jError as e:
            return {"error": "Neo4j query failed", "detail": str(e)}

    # --------------------------------------------------
    # SECTION TIMELINE
    # --------------------------------------------------
    def get_section_timeline(self, act_id: str, section_no: str) -> Dict[str, Any]:
        act_id = act_id.strip()
        section_no = section_no.strip()

        cypher = """
        MATCH (a:Act {act_id:$act_id})-[:HAS_SECTION]->(s:Section)
        WHERE trim(s.section_no) = trim($section_no)

        MATCH (s)-[:HAS_VERSION]->(sv:SectionVersion)
        OPTIONAL MATCH (sv)-[:CHANGED_BY]->(am:Amendment)
        OPTIONAL MATCH (prev:SectionVersion)-[r:NEXT_VERSION]->(sv)

        RETURN
          a.act_id AS act_id,
          a.title AS act_title,
          a.jurisdiction AS jurisdiction,
          s.section_no AS section_no,
          collect({
            version_id: sv.version_id,
            valid_from: CASE WHEN sv.valid_from IS NULL THEN NULL ELSE toString(sv.valid_from) END,
            valid_to:   CASE WHEN sv.valid_to   IS NULL THEN NULL ELSE toString(sv.valid_to) END,
            section_title: sv.title,
            text: sv.text,
            amendment: CASE WHEN am IS NULL THEN NULL ELSE {
              amend_id: am.amend_id,
              date: toString(am.date),
              am_title: am.am_title,
              summary: am.summary,
              section_no: am.section_no,
              section_title: am.section_title,
              act_id: am.act_id,
              jurisdiction: am.jurisdiction
            } END,
            change_from_prev: CASE WHEN r IS NULL THEN NULL ELSE {
              summary: r.summary,
              added: r.added,
              removed: r.removed
            } END
          }) AS timeline
        """

        try:
            with self.driver.session() as session:
                rec = session.run(
                    cypher,
                    act_id=act_id,
                    section_no=section_no
                ).single()

            if not rec:
                return {"error": "Section not found"}

            data = dict(rec)
            data["timeline"] = sorted(
                data.get("timeline", []),
                key=lambda x: (x.get("valid_from") or "")
            )
            return data

        except Neo4jError as e:
            return {"error": "Neo4j query failed", "detail": str(e)}

    # --------------------------------------------------
    # BEFORE / AFTER BY VERSION
    # --------------------------------------------------
    def get_change_detail(self, after_version_id: str) -> Dict[str, Any]:
        after_version_id = after_version_id.strip()

        cypher = """
        MATCH (before:SectionVersion)-[r:NEXT_VERSION]->(after:SectionVersion {version_id:$after_id})
        OPTIONAL MATCH (after)-[:CHANGED_BY]->(am:Amendment)
        RETURN
          before {
            version_id: before.version_id,
            valid_from: toString(before.valid_from),
            text: before.text
          } AS before_version,
          after {
            version_id: after.version_id,
            valid_from: toString(after.valid_from),
            text: after.text
          } AS after_version,
          {
            summary: r.summary,
            diff: r.diff,
            added: r.added,
            removed: r.removed
          } AS change,
          CASE WHEN am IS NULL THEN NULL ELSE {
            amend_id: am.amend_id,
            date: toString(am.date),
            am_title: am.am_title,
            summary: am.summary
          } END AS amendment
        """

        try:
            with self.driver.session() as session:
                rec = session.run(
                    cypher,
                    after_id=after_version_id
                ).single()

            if not rec:
                return {"error": "Change not found"}

            return dict(rec)

        except Neo4jError as e:
            return {"error": "Neo4j query failed", "detail": str(e)}

    # --------------------------------------------------
    # AMENDMENT BY ID
    # --------------------------------------------------
    def get_amendment_detail(self, amend_id: str) -> Dict[str, Any]:
        amend_id = amend_id.strip()

        cypher = """
        MATCH (am:Amendment {amend_id:$amend_id})
        RETURN am {
            amend_id: am.amend_id,
            date: toString(am.date),
            am_title: am.am_title,
            summary: am.summary,
            section_no: am.section_no,
            section_title: am.section_title,
            act_id: am.act_id,
            jurisdiction: am.jurisdiction
        } AS amendment
        """

        try:
            with self.driver.session() as session:
                rec = session.run(
                    cypher,
                    amend_id=amend_id
                ).single()

            if not rec:
                return {"error": "Amendment not found"}

            return dict(rec)

        except Neo4jError as e:
            return {"error": "Neo4j query failed", "detail": str(e)}

    # --------------------------------------------------
    # AMENDMENTS BY DATE
    # --------------------------------------------------
    def get_amendments_by_date(self, as_of_date: str) -> Dict[str, Any]:
        as_of_date = as_of_date.strip()

        cypher = """
        MATCH (am:Amendment)
        WHERE am.date IS NOT NULL AND am.date <= date($as_of_date)
        RETURN am {
            amend_id: am.amend_id,
            date: toString(am.date),
            am_title: am.am_title,
            summary: am.summary,
            section_no: am.section_no,
            section_title: am.section_title,
            act_id: am.act_id,
            jurisdiction: am.jurisdiction
        } AS amendment
        ORDER BY am.date
        """

        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher,
                    as_of_date=as_of_date
                )
                amendments = [r["amendment"] for r in result]

            return {
                "as_of_date": as_of_date,
                "count": len(amendments),
                "amendments": amendments
            }

        except Neo4jError as e:
            return {"error": "Neo4j query failed", "detail": str(e)}
