from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uuid
import os
import re
import io

from app.pdf_service import extract_text_from_pdf_bytes
from app.legalbert_classifier import classify_text
from app.embedding_service import generate_embedding
from app.kg_builder_service import store_case
from app.hybrid_engine import hybrid_search
from app.neo4j_driver import db
from app.legal_validator import is_legal_document
from app.metadata_service import extract_case_name, extract_case_number
from app.complaint_defense_extractor import extract_complaint_defense
from app.utils import generate_file_hash, case_exists

from app.legal_issue_extractor import extract_legal_issues

# MongoDB storage
from app.mongodb_service import upload_case_file, get_case_file


# -------------------------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------------------------
def clean_text(text: str):

    text = re.sub(r"\(\s*\d+\s*\)", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ Don't load heavy models at startup — let them lazy load on first request
    print("✅ Past Case Retrieval service started (models will load on first request)")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan)


# --------------------------------
# HEALTH CHECK
# --------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "past-case-retrieval"}


# --------------------------------
# ADMIN: STORE CASE IN KG
# --------------------------------
@app.post("/admin/upload-case")
async def build_kg(file: UploadFile = File(...)):

    file_bytes = await file.read()

    case_id = generate_file_hash(file_bytes)

    # prevent duplicates
    if case_exists(case_id):
        return {
            "message": "Case already exists in Knowledge Graph",
            "case_id": case_id,
            "case_stored": False
        }

    text = extract_text_from_pdf_bytes(file_bytes)

    if not is_legal_document(text):
        return {
            "message": "Uploaded file is not a valid legal document",
            "case_stored": False
        }

    case_number = extract_case_number(text)
    case_name = extract_case_name(text)

    display_name = f"{case_number} - {case_name}"

    complaint_text, defense_text = extract_complaint_defense(text)

    roles = classify_text(text)

    embeddings = {
        r: generate_embedding(" ".join(roles[r]))
        for r in ["facts", "issues", "arguments", "decisions"]
    }

    issues = list(set(roles["issues"]))[:5]

    # store PDF in MongoDB
    file_id = upload_case_file(case_id, file_bytes)

    # store metadata in Neo4j
    store_case(
        case_id=case_id,
        case_name=display_name,
        roles=roles,
        embeddings=embeddings,
        issues=issues,
        summary=text,
        complaint=complaint_text,
        defense=defense_text,
        file_id=file_id
    )

    return {
        "message": "Case stored in Knowledge Graph",
        "case_id": case_id,
        "case_stored": True
    }


# --------------------------------
# USER: SEARCH SIMILAR CASES
# --------------------------------
@app.post("/search")
async def search(file: UploadFile = File(...)):

    file_bytes = await file.read()

    case_id = generate_file_hash(file_bytes)

    text = extract_text_from_pdf_bytes(file_bytes)

    case_number = extract_case_number(text)
    case_name = extract_case_name(text)

    display_name = f"{case_number} - {case_name}"

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

    
    issues = roles["issues"][:5]

    results = hybrid_search(embeddings, issues)

    # store new case if not exist
    if not case_exists(case_id):

        file_id = upload_case_file(case_id, file_bytes)

        store_case(
            case_id=case_id,
            case_name=display_name,
            roles=roles,
            embeddings=embeddings,
            issues=issues,
            summary=text,
            complaint=complaint_text,
            defense=defense_text,
            file_id=file_id
        )

    return {
        "new_case_id": case_id,
        "similar_cases": results
    }


# --------------------------------
# GET CASE DETAILS
# --------------------------------
@app.get("/case/{case_id}")
def get_case(case_id: str):

    result = db.query("""
    MATCH (c:Case {case_id:$id})
    RETURN c.case_id AS case_id,
           c.case_name AS case_name,
           c.summary AS judgment,
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
        "complaint": cleaned_complaint,
        "defense": cleaned_defense,

        # FULL JUDGMENT
        "judgment": record["judgment"]
    }


# --------------------------------
# VIEW FULL PDF FILE
# --------------------------------
@app.get("/case-file/{case_id}")
def view_case(case_id: str):

    result = db.query("""
    MATCH (c:Case {case_id:$id})
    RETURN c.file_id AS file_id
    """, {"id": case_id})

    if not result:
        return {"message": "Case not found"}

    file_id = result[0]["file_id"]

    file_bytes = get_case_file(file_id)

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/pdf"
    )