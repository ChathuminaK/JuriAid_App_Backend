from app.neo4j_driver import db
from app.config import COURT_LEVELS

def citation_similarity(query_id, candidate_id):
    query = """
    MATCH (q:Case {case_id:$qid})-[:CITES*1..2]->(ref)<-[:CITES*1..2]-(c:Case {case_id:$cid})
    RETURN count(ref) as score
    """
    result = db.query(query, {"qid": query_id, "cid": candidate_id})
    count = result[0]["score"] if result else 0
    return min(count / 10, 1)

def legal_issue_similarity(query_id, candidate_id):
    query = """
    MATCH (q:Case {case_id:$qid})-[:INVOLVES_ISSUE]->(i)<-[:INVOLVES_ISSUE]-(c:Case {case_id:$cid})
    RETURN count(i) as shared
    """
    result = db.query(query, {"qid": query_id, "cid": candidate_id})
    count = result[0]["shared"] if result else 0
    return min(count / 5, 1)

def temporal_similarity(q_year, c_year):
    return 1 / (1 + abs(q_year - c_year))

def court_similarity(q_court, c_court):
    q_level = COURT_LEVELS.get(q_court, 0)
    c_level = COURT_LEVELS.get(c_court, 0)
    return 1 / (1 + abs(q_level - c_level))