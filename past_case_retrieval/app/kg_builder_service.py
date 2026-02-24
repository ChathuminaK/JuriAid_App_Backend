from app.neo4j_driver import db


def store_case(case_id, roles, embeddings, issues, summary=None):
    """
    Store case embeddings and legal issues.
    """

    query = """
    MERGE (c:Case {case_id:$case_id})
    SET c.facts_embedding=$facts_embedding,
        c.issues_embedding=$issues_embedding,
        c.arguments_embedding=$arguments_embedding,
        c.summary=$summary
    """

    db.query(query, {
        "case_id": case_id,
        "facts_embedding": embeddings["facts"],
        "issues_embedding": embeddings["issues"],
        "arguments_embedding": embeddings["arguments"],
        "summary": summary
    })

    # Create LegalIssue nodes
    for issue in issues:
        db.query("""
        MERGE (i:LegalIssue {name:$name})
        WITH i
        MATCH (c:Case {case_id:$case_id})
        MERGE (c)-[:INVOLVES_ISSUE]->(i)
        """, {
            "name": issue.strip(),
            "case_id": case_id
        })