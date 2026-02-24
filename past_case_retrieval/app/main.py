from fastapi import FastAPI, UploadFile, File
import uuid, os

from app.pdf_service import extract_text_from_pdf_bytes
from app.role_classifier import classify_text
from app.embedding_service import generate_embedding
from app.kg_builder_service import store_case
from app.hybrid_engine import hybrid_search
from app.neo4j_driver import db
from app.legal_validator import is_legal_document

app = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------
# HEALTH CHECK
# --------------------------------
@app.get("/")
def root():
    return {"status": "Backend running"}

@app.get("/health")
def health():
    try:
        db.query("RETURN 1")
        return {"database": "connected"}
    except Exception as e:
        return {"database": "error", "message": str(e)}


# --------------------------------
# ADMIN: STORE CASES IN KG
# --------------------------------
@app.post("/admin/upload-case")
async def build_kg(file: UploadFile = File(...)):

    case_id = str(uuid.uuid4())

    file_bytes = await file.read()

    text = extract_text_from_pdf_bytes(file_bytes)

    roles = classify_text(text)

    embeddings = {
        r: generate_embedding(" ".join(roles[r]))
        for r in ["facts", "issues", "arguments"]
    }

    issues = list(set(roles["issues"]))[:5]

    store_case(
        case_id=case_id,
        roles=roles,
        embeddings=embeddings,
        issues=issues,
        summary=text
    )

    return {
        "message": "Case stored in Knowledge Graph",
        "case_id": case_id
    }


# --------------------------------
# USER: SEARCH SIMILAR CASES
# --------------------------------
@app.post("/search")
async def search(file: UploadFile = File(...)):

    case_id = str(uuid.uuid4())

    file_bytes = await file.read()

    text = extract_text_from_pdf_bytes(file_bytes)

    if not is_legal_document(text):
        return {
            "message": "Uploaded file is not a legal case document",
            "similar_cases": []
        }

    roles = classify_text(text)

    embeddings = {
        r: generate_embedding(" ".join(roles[r]))
        for r in ["facts", "issues", "arguments"]
    }

    # ✅ DEFINE issues BEFORE calling hybrid_search
    issues = list(set(roles["issues"]))[:5]

    # ✅ PASS issues properly
    results = hybrid_search(embeddings, issues)

    # Store case after search
    store_case(
        case_id=case_id,
        roles=roles,
        embeddings=embeddings,
        issues=issues,
        summary=text
    )

    return {
        "new_case_id": case_id,
        "similar_cases": results
    }