from app.config import WEIGHTS
from app.neo4j_driver import db


# -------------------------------------------------
# Get shared legal issues between cases
# -------------------------------------------------
def get_shared_legal_issues(query_issues, candidate_id):

    result = db.query("""
    MATCH (c:Case {case_id:$cid})-[:INVOLVES_ISSUE]->(i)
    RETURN collect(i.name) AS issues
    """, {"cid": candidate_id})

    if not result:
        return []

    candidate_issues = result[0]["issues"]
    return list(set(query_issues) & set(candidate_issues))


# -------------------------------------------------
#Generate clear explanation text
# -------------------------------------------------
def generate_reason(breakdown, shared_issues):

    reasons = []

    if breakdown["facts"] > 0.85:
        reasons.append("Very similar factual background")

    if shared_issues:
        reasons.append("Shared legal issues: " + ", ".join(shared_issues))

    if breakdown["arguments"] > 0.75:
        reasons.append("Similar legal reasoning")

    if breakdown.get("decisions", 0) > 0.75:
        reasons.append("Similar judicial decision")

    if not reasons:
        reasons.append("General legal similarity")

    return ". ".join(reasons) + "."


# -------------------------------------------------
#Hybrid Search Function (UPDATED WITH DECISIONS)
# -------------------------------------------------
def hybrid_search(embeddings, query_issues):

    # 🔎 Step 1: Vector search based on facts
    vector_query = """
    CALL db.index.vector.queryNodes('facts_embedding_index', 5, $facts_embedding)
    YIELD node, score
    RETURN node.case_id AS case_id,
           node.case_name AS case_name,
           node.summary AS summary,
           score AS facts_score
    """

    results = db.query(vector_query, {
        "facts_embedding": embeddings["facts"]
    })

    final_results = []

    for r in results:

        cid = r["case_id"]

        # ---- Issue similarity ----
        issue_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
       RETURN
CASE
    WHEN c.decisions_embedding IS NOT NULL
    THEN gds.similarity.cosine(c.decisions_embedding, $embedding)
    ELSE 0
END AS score
        """, {
            "id": cid,
            "embedding": embeddings["issues"]
        })

        issue_score = issue_score_result[0]["score"] if issue_score_result else 0

        # ---- Argument similarity ----
        arg_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.arguments_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["arguments"]
        })

        arg_score = arg_score_result[0]["score"] if arg_score_result else 0

        # ---- Decision similarity (NEW) ----
        decision_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.decisions_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["decisions"]
        })

        decision_score = decision_score_result[0]["score"] if decision_score_result else 0

        # ---- Score breakdown ----
        breakdown = {
            "facts": r["facts_score"],
            "issues": issue_score,
            "arguments": arg_score,
            "decisions": decision_score
        }

        # ---- Final weighted score ----
        final_score = (
            WEIGHTS["facts"] * breakdown["facts"] +
            WEIGHTS["issues"] * breakdown["issues"] +
            WEIGHTS["arguments"] * breakdown["arguments"] +
            WEIGHTS["decisions"] * breakdown["decisions"]
        )

        # ---- Shared issues ----
        shared_issues = get_shared_legal_issues(query_issues, cid)

        # ---- Explanation ----
        reason = generate_reason(breakdown, shared_issues)

        final_results.append({
            "case_id": cid,
            "case_name": r["case_name"],
            "final_score": round(final_score, 4),
            "summary": r["summary"][:1000] if r["summary"] else "",
            "reason": reason,
            "shared_issues": shared_issues,
            "breakdown": breakdown
        })

    # ---- Filter & Sort ----
    filtered = [
        r for r in final_results
        if r["final_score"] >= 0.50
    ]

    filtered.sort(key=lambda x: x["final_score"], reverse=True)

    return filtered[:3]