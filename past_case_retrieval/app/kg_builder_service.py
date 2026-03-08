from app.neo4j_driver import db


def store_case(case_id, case_name, roles, embeddings, issues,
               summary=None, complaint=None, defense=None,
               file_id=None):

    query = """
    MERGE (c:Case {case_id:$case_id})
    SET c.case_name=$case_name,
        c.facts_embedding=$facts_embedding,
        c.issues_embedding=$issues_embedding,
        c.arguments_embedding=$arguments_embedding,
        c.decisions_embedding=$decisions_embedding,
        c.summary=$summary,
        c.complaint=$complaint,
        c.defense=$defense,
        c.file_id=$file_id
    """

    db.query(query, {
        "case_id": case_id,
        "case_name": case_name,
        "facts_embedding": embeddings["facts"],
        "issues_embedding": embeddings["issues"],
        "arguments_embedding": embeddings["arguments"],
        "decisions_embedding": embeddings["decisions"],
        "summary": summary,
        "complaint": complaint,
        "defense": defense,
        "file_id": file_id
    })

    # Store legal issues
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