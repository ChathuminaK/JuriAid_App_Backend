# backend/knowledge_graph.py
from neo4j import GraphDatabase
import os
import re
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASS

# Create driver (singleton)
_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def close_driver():
    global _driver
    if _driver:
        _driver.close()

def create_case_node(case_id: str, title: str = "", year: str = ""):
    """
    MERGE a Case node so re-ingesting same case is safe.
    """
    with _driver.session() as session:
        session.run(
            "MERGE (c:Case {id:$id}) "
            "SET c.title = $title, c.year = $year",
            id=case_id, title=title or "", year=year or ""
        )

def create_citation(from_case: str, to_case: str):
    """
    Create a CITES relationship (MERGE safe).
    """
    with _driver.session() as session:
        session.run(
            """
            MATCH (a:Case {id:$from})
            MERGE (b:Case {id:$to})
            MERGE (a)-[:CITES]->(b)
            """,
            parameters={
                "from": from_case,   
                "to": to_case
            }
        )


def extract_candidate_case_ids(text: str, max_candidates: int = 50):
    """
    Heuristic extraction of case-like tokens for citation edges.
    You should adapt this if your jurisdiction has standard citation patterns.
    We keep it conservative and short.
    """
    # Basic: look for repeated uppercase tokens or tokens with 'v' (v. / vs / vs.)
    tokens = set()
    # pattern: WORD v WORD  (e.g., 'Smith v Jones')
    vs_matches = re.findall(r'\b([A-Z][A-Za-z]{2,})\s+v(?:\.|s)?\s+([A-Z][A-Za-z]{2,})\b', text)
    for a, b in vs_matches:
        tokens.add(f"{a}_v_{b}")
    # also find uppercase words that look like case ids or filenames
    words = re.findall(r'\b[A-Z][A-Za-z0-9_\-]{2,}\b', text)
    for w in words:
        # ignore very long tokens
        if 3 <= len(w) <= 30:
            tokens.add(w)
        if len(tokens) >= max_candidates:
            break
    return list(tokens)[:max_candidates]

def get_direct_citation(query_case_id: str, candidate_case_id: str) -> bool:
    """
    Return True if query -> candidate exists (query cites candidate)
    """
    with _driver.session() as session:
        res = session.run(
            "MATCH (q:Case {id:$q})-[r:CITES]->(c:Case {id:$c}) RETURN count(r) AS ct",
            q=query_case_id, c=candidate_case_id
        ).single()
        return (res and res["ct"] and res["ct"] > 0)

def get_reverse_citation(query_case_id: str, candidate_case_id: str) -> bool:
    """
    Return True if candidate -> query exists
    """
    with _driver.session() as session:
        res = session.run(
            "MATCH (c:Case {id:$c})-[r:CITES]->(q:Case {id:$q}) RETURN count(r) AS ct",
            q=query_case_id, c=candidate_case_id
        ).single()
        return (res and res["ct"] and res["ct"] > 0)

def get_shared_neighbors_count(query_case_id: str, candidate_case_id: str) -> int:
    """
    Number of common nodes they both cite (two-hop overlap)
    """
    with _driver.session() as session:
        res = session.run(
            """
            MATCH (q:Case {id:$q})-[:CITES]->(x)<-[:CITES]-(c:Case {id:$c})
            RETURN count(distinct x) AS ct
            """, q=query_case_id, c=candidate_case_id
        ).single()
        return int(res["ct"]) if res else 0


