from fastapi import FastAPI, UploadFile, File
import uuid, os
import re

from app.pdf_service import extract_text_from_pdf_bytes
from app.legalbert_classifier import classify_text
from app.embedding_service import generate_embedding
from app.kg_builder_service import store_case
from app.hybrid_engine import hybrid_search
from app.neo4j_driver import db
from app.legal_validator import is_legal_document
from app.metadata_service import extract_case_name
from app.complaint_defense_extractor import extract_complaint_defense


# -------------------------------------------------
# ✅ CLEAN TEXT FUNCTION (FIX ADDED HERE)
# -------------------------------------------------
def clean_text(text: str):

    # Remove page numbers like ( 353 )
    text = re.sub(r"\(\s*\d+\s*\)", "", text)

    # Remove strange OCR symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Fix broken hyphenated words
    text = re.sub(r"-\s+", "", text)

    # Replace multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Replace multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


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

    # ✅ ADD VALIDATION HERE
    if not is_legal_document(text):
        return {
            "message": "Uploaded file is not a valid legal case document",
            "case_stored": False
        }

    case_name = extract_case_name(text)

    complaint_text, defense_text = extract_complaint_defense(text)

    roles = classify_text(text)

    embeddings = {
        r: generate_embedding(" ".join(roles[r]))
        for r in ["facts", "issues", "arguments", "decisions"]
    }

    issues = list(set(roles["issues"]))[:5]

    store_case(
        case_id=case_id,
        case_name=case_name,
        roles=roles,
        embeddings=embeddings,
        issues=issues,
        summary=text,
        complaint=complaint_text,
        defense=defense_text
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
    case_name = extract_case_name(text)

    if not is_legal_document(text):
        return {
            "message": "Uploaded file is not a legal case document",
            "similar_cases": []
        }

    complaint_text, defense_text = extract_complaint_defense(text)

    roles = classify_text(text)

    embeddings = {
        r: generate_embedding(" ".join(roles[r]))
        for r in ["facts", "issues", "arguments", "decisions"]
    }

    issues = list(set(roles["issues"]))[:5]

    results = hybrid_search(embeddings, issues)

    store_case(
        case_id=case_id,
        case_name=case_name,
        roles=roles,
        embeddings=embeddings,
        issues=issues,
        summary=text,
        complaint=complaint_text,
        defense=defense_text
    )

    return {
        "new_case_id": case_id,
        "similar_cases": results
    }


# --------------------------------
# GET STORED CASE
# --------------------------------
@app.get("/case/{case_id}")
def get_case(case_id: str):

    result = db.query("""
    MATCH (c:Case {case_id:$id})
    RETURN c.case_id AS case_id,
           c.case_name AS case_name,
           c.summary AS full_text,
           c.complaint AS complaint,
           c.defense AS defense
    """, {"id": case_id})

    if not result:
        return {"message": "Case not found"}

    record = result[0]

    cleaned_complaint = clean_text(record["complaint"]) if record["complaint"] else ""
    cleaned_defense = clean_text(record["defense"]) if record["defense"] else ""

    return {
        "case_id": record["case_id"],
        "case_name": record["case_name"],
        "complaint": cleaned_complaint if cleaned_complaint else "Complaint not available",
        "defense": cleaned_defense if cleaned_defense else "Defense not available"
    }