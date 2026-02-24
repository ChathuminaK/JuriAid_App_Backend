from app.config import WEIGHTS
from app.neo4j_driver import db


# -------------------------------------------------
# 1️⃣ Get shared legal issues between cases
# -------------------------------------------------
def get_shared_legal_issues(query_issues, candidate_id):

    result = db.query("""
    MATCH (c:Case {case_id:$cid})-[:INVOLVES_ISSUE]->(i)
    RETURN collect(i.name) AS issues
    """, {"cid": candidate_id})

    if not result:
        return []

    candidate_issues = result[0]["issues"]

    # Find common issues
    shared = list(set(query_issues) & set(candidate_issues))

    return shared


# -------------------------------------------------
# 2️⃣ Generate clear explanation text
# -------------------------------------------------
def generate_reason(breakdown, shared_issues):

    reasons = []

    if breakdown["facts"] > 0.85:
        reasons.append("Very similar factual background")

    if shared_issues:
        reasons.append("Shared legal issues: " + ", ".join(shared_issues))

    if breakdown["arguments"] > 0.75:
        reasons.append("Similar legal reasoning")

    if not reasons:
        reasons.append("General legal similarity")

    return ". ".join(reasons) + "."


# -------------------------------------------------
# 3️⃣ Hybrid Search Function
# -------------------------------------------------
def hybrid_search(embeddings, query_issues):

    # 🔎 Step 1: Get top 5 similar by facts embedding
    vector_query = """
    CALL db.index.vector.queryNodes('facts_embedding_index', 5, $facts_embedding)
    YIELD node, score
    RETURN node.case_id AS case_id,
           node.summary AS summary,
           score AS facts_score
    """

    results = db.query(vector_query, {
        "facts_embedding": embeddings["facts"]
    })

    final_results = []

    # 🔎 Step 2: Calculate hybrid similarity for each candidate
    for r in results:

        cid = r["case_id"]
        summary = r.get("summary", "")

        # --- Issue similarity ---
        issue_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.issues_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["issues"]
        })

        issue_score = issue_score_result[0]["score"] if issue_score_result else 0


        # --- Argument similarity ---
        arg_score_result = db.query("""
        MATCH (c:Case {case_id:$id})
        RETURN gds.similarity.cosine(c.arguments_embedding, $embedding) AS score
        """, {
            "id": cid,
            "embedding": embeddings["arguments"]
        })

        arg_score = arg_score_result[0]["score"] if arg_score_result else 0


        # --- Score breakdown ---
        breakdown = {
            "facts": r["facts_score"],
            "issues": issue_score,
            "arguments": arg_score
        }


        # --- Final weighted score ---
        final_score = (
            WEIGHTS["facts"] * breakdown["facts"] +
            WEIGHTS["issues"] * breakdown["issues"] +
            WEIGHTS["arguments"] * breakdown["arguments"]
        )


        # --- Get shared legal issues ---
        shared_issues = get_shared_legal_issues(query_issues, cid)


        # --- Generate explanation ---
        reason = generate_reason(breakdown, shared_issues)


        # --- Append result ---
        final_results.append({
            "case_id": cid,
            "final_score": round(final_score, 4),
            "summary": summary[:1000] if summary else "Summary not available",
            "reason": reason,
            "shared_issues": shared_issues,
            "breakdown": breakdown
        })


    # -------------------------------------------------
    # 4️⃣ FILTER + SORT
    # -------------------------------------------------

    SIMILARITY_THRESHOLD = 0.50

    # Remove weak matches
    filtered_results = [
        r for r in final_results
        if r["final_score"] >= SIMILARITY_THRESHOLD
    ]

    # Sort highest first
    filtered_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Return only top 3 strongest matches
    return filtered_results[:3]