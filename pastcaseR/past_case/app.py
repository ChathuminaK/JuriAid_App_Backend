# backend/app.py
import os
import io
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from extractor import extract_text
from processor import split_into_sentences, clean_text
from classifier import predict_roles
from vector_store import embed_text, append_to_index, load_index, save_index, search_index
from knowledge_graph import create_case_node, create_citation, extract_candidate_case_ids
from hybrid import hybrid_rank
from config import DATA_FOLDER, TMP_UPLOAD_FOLDER, ROLE_WEIGHTS
from typing import List

os.makedirs(TMP_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = FastAPI(title="Hybrid Retrieval Backend")

@app.get("/health")
def health():
    return {"status": "ok"}

def _make_meta_entry(case_id: str, role: str, snippet: str):
    return {
        "case_id": case_id,
        "role": role,
        "snippet": snippet
    }

@app.post("/ingest_folder")
def ingest_folder(folder: str = DATA_FOLDER):
    """
    Ingests all PDFs in the given folder.
    Each PDF is processed into role-aggregated vectors (case_id|role -> vector),
    appended to FAISS, and case nodes + citations are created in Neo4j.
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not files:
        return {"ingested": 0, "message": f"No pdfs found in {folder}"}

    vectors = []
    metas = []
    for fname in files:
        case_id = os.path.splitext(fname)[0]
        path = os.path.join(folder, fname)
        try:
            text = extract_text(path, prefer_ocr=True)
            text = clean_text(text)
            print(f"[DEBUG] {fname} extracted text length:", len(text))

            if not text or len(text) < 20:
                # skip empty PDFs
                continue
            sentences = split_into_sentences(text)
            roles = predict_roles(sentences)
            # create node
            create_case_node(case_id, title=case_id)
            # create citations heuristically
            cited_ids = extract_candidate_case_ids(text)
            for c in cited_ids:
                create_citation(case_id, c)
            # group sentences by role
            role_map = {}
            for s, r in zip(sentences, roles):
                role_map.setdefault(r, []).append(s)
            # embed aggregated text per role
            for r, s_list in role_map.items():
                combined = " ".join(s_list).strip()
                if len(combined) < 5:
                    continue
                vec = embed_text([combined])[0].astype("float32")
                vectors.append(vec)
                metas.append(_make_meta_entry(case_id, r, combined[:800]))
        except Exception as e:
            # log & continue
            print(f"Error processing {fname}: {e}")
            continue

    if len(vectors) == 0:
        return {"ingested": 0, "message": "No vectors created (empty texts?)"}
    vectors_np = np.vstack(vectors).astype("float32")
    index, meta = append_to_index(vectors_np, metas)
    return {"ingested": len(vectors), "cases": len(files)}

@app.post("/upload_and_search")
async def upload_and_search(file: UploadFile = File(...), topk: int = 5):
    """
    Upload a new case PDF, build a role-aware query representation,
    enrich the Knowledge Graph with a temporary query node,
    and perform true hybrid retrieval (Vector + KG).
    """

    # ---------- 1. Read & validate ----------
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    tmp_name = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(TMP_UPLOAD_FOLDER, tmp_name)

    with open(tmp_path, "wb") as f:
        f.write(contents)

    temp_query_case_id = f"__QUERY__{uuid.uuid4().hex[:10]}"

    try:
        # ---------- 2. Text extraction ----------
        text = extract_text(tmp_path, prefer_ocr=True)
        text = clean_text(text)

        sentences = split_into_sentences(text)
        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="No usable text extracted from PDF"
            )

        # ---------- 3. Role classification ----------
        roles = predict_roles(sentences)

        role_map = {}
        for sent, role in zip(sentences, roles):
            role_map.setdefault(role, []).append(sent)

        # ---------- 4. Role-wise embeddings ----------
        role_vectors = {}
        for role, sents in role_map.items():
            combined = " ".join(sents).strip()
            if len(combined) < 10:
                continue
            role_vectors[role] = embed_text([combined])[0].astype("float32")

        if not role_vectors:
            raise HTTPException(
                status_code=400,
                detail="Failed to generate embeddings for uploaded case"
            )

        # ---------- 5. Weighted query aggregation ----------
        agg_vec = None
        total_weight = 0.0

        for role, vec in role_vectors.items():
            weight = ROLE_WEIGHTS.get(role, 0.05)
            agg_vec = vec * weight if agg_vec is None else agg_vec + vec * weight
            total_weight += weight

        query_vec = (agg_vec / max(total_weight, 1e-9)).astype("float32")

        # ---------- 6. Create TEMP query node in KG ----------
        create_case_node(
            temp_query_case_id,
            title=file.filename
        )

        # ---------- 7. Extract & attach citations ----------
        cited_case_ids = extract_candidate_case_ids(text)

        for cited_id in cited_case_ids:
            create_citation(temp_query_case_id, cited_id)

        # ---------- 8. Hybrid retrieval ----------
        results = hybrid_rank(
            query_vec=query_vec,
            query_case_id=temp_query_case_id,
            topk=topk
        )

        # ---------- 9. Response ----------
        return JSONResponse({
            "query_case_id": temp_query_case_id,
            "roles_detected": list(role_vectors.keys()),
            "results": results
        })

    finally:
        # ---------- 10. Cleanup ----------
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/rebuild_index")
def rebuild_index(folder: str = DATA_FOLDER):
    """
    Rebuild the FAISS index from scratch using all PDFs in folder.
    """
    # clear existing index files if present
    from vector_store import FAISS_INDEX_PATH, FAISS_META_PATH
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(FAISS_META_PATH):
            os.remove(FAISS_META_PATH)
    except Exception as e:
        print("Error removing index files:", e)
    return ingest_folder(folder)
