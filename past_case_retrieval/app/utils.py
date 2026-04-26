import hashlib
from app.neo4j_driver import db


# ----------------------------------------
# Generate SHA256 hash from file bytes
# ----------------------------------------
def generate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# ----------------------------------------
# Check if case already exists
# ----------------------------------------
def case_exists(case_id: str) -> bool:
    result = db.query("""
    MATCH (c:Case {case_id:$id})
    RETURN c.case_id AS case_id
    """, {"id": case_id})

    return len(result) > 0