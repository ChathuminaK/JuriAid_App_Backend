from datetime import datetime
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse

from .orchestrator.core import (
    UPLOADS_DIR,
    OUTPUTS_DIR,
    process_single_file,        # orchestrates full pipeline
    analyze_case_content,       # text-only analysis
    generate_legal_summary      # builds summary
)

app = FastAPI(
    title="JuriAid Orchestrator API",
    version="1.0.0",
    description="Upload Sri Lankan legal case files (PDF/TXT) and get structured analysis."
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.get("/")
def health():
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "uploads_dir": UPLOADS_DIR,
        "outputs_dir": OUTPUTS_DIR
    }

@app.post("/api/upload-case")
async def upload_case(file: UploadFile = File(..., description="form-data key: file")):
    # Normalize common invalid states to a consistent message
    if not file or not getattr(file, "filename", "").strip():
        raise HTTPException(status_code=422, detail="Missing file upload (form-data key 'file').")
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file content.")

    ts_name = f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}"
    dest = os.path.join(UPLOADS_DIR, ts_name)
    with open(dest, "wb") as f:
        f.write(data)

    result = process_single_file(dest)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
    return JSONResponse(result)

@app.post("/api/upload-case-with-prompt")
async def upload_case_with_prompt(
    file: UploadFile = File(..., description="PDF or TXT"),
    prompt: str = Form(..., description="User instruction / extra context")
):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    ts_name = f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}"
    dest = os.path.join(UPLOADS_DIR, ts_name)
    with open(dest, "wb") as f:
        f.write(data)
    result = process_single_file(dest)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error"))
    result["prompt_used"] = prompt
    return JSONResponse(result)

@app.post("/api/analyze-text")
async def analyze_text(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    analysis = analyze_case_content(text)
    summary = generate_legal_summary(text, analysis)
    return {
        "success": True,
        "case_type": analysis["case_type"],
        "length": analysis["length"],
        "preview": text[:500] + ("..." if len(text) > 500 else ""),
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(status_code=500, content={"success": False, "detail": str(exc)})