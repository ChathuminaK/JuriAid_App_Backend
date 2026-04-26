from app.config import WEIGHTS
from app.neo4j_driver import db


def get_shared_legal_issues(query_issues, candidate_id):

    result = db.query("""
    MATCH (c:Case {case_id:$cid})-[:INVOLVES_ISSUE]->(i)
    RETURN collect(i.name) AS issues
    """, {"cid": candidate_id})

    if not result:
        return []

    candidate_issues = result[0]["issues"]
    return list(set(query_issues) & set(candidate_issues))


def generate_reason(breakdown, shared_issues):

    reasons = []

    if breakdown["facts"] > 0.85:
        reasons.append("Very similar factual background")

    if shared_issues:
        reasons.append("Shared legal issues: " + ", ".join(shared_issues))

    if breakdown["arguments"] > 0.75:
        reasons.append("Similar legal reasoning")

    if breakdown["decisions"] > 0.75:
        reasons.append("Similar judicial decision")

    if not reasons:
        reasons.append("General legal similarity")

    return ". ".join(reasons) + "."


def hybrid_search(embeddings, query_issues, limit=3):

    vector_query = """
    CALL db.index.vector.queryNodes('facts_embedding_index', 20, $facts_embedding)
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

        # ISSUE SIMILARITY
        issue_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.issues_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["issues"]
        })

        issue_score = issue_score_result[0]["score"] if issue_score_result else 0


        # ARGUMENT SIMILARITY
        arg_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.arguments_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["arguments"]
        })

        arg_score = arg_score_result[0]["score"] if arg_score_result else 0


        # DECISION SIMILARITY
        decision_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.decisions_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["decisions"]
        })

        decision_score = decision_score_result[0]["score"] if decision_score_result else 0


        breakdown = {
            "facts": r["facts_score"],
            "issues": issue_score,
            "arguments": arg_score,
            "decisions": decision_score
        }

        final_score = (
            WEIGHTS["facts"] * breakdown["facts"] +
            WEIGHTS["issues"] * breakdown["issues"] +
            WEIGHTS["arguments"] * breakdown["arguments"] +
            WEIGHTS["decisions"] * breakdown["decisions"]
        )

        shared_issues = get_shared_legal_issues(query_issues, cid)

        reason = generate_reason(breakdown, shared_issues)

        final_results.append({
            "case_id": cid,
            "case_name": r["case_name"],
            "final_score": round(final_score, 4),
            "judgment_preview": r["summary"][:500] if r["summary"] else "",
            "reason": reason,
            "shared_issues": shared_issues,
            "breakdown": breakdown,
            "view_case_details": f"/case/{cid}",
            "view_full_case_file": f"/case-file/{cid}"
        })


    filtered = [
        r for r in final_results
        if r["final_score"] >= 0.50
    ]

    filtered.sort(key=lambda x: x["final_score"], reverse=True)

    return filtered[:limit]
