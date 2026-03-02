"""
JuriAid Orchestrator API
========================
The brain of the JuriAid legal AI system.

Endpoints:
  GET  /health         → Health check
  POST /api/analyze    → Full case analysis pipeline
  POST /api/cases/save → Save case for future reference
"""

import logging
import sys
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from auth_middleware import verify_token
from orchestrator.schemas import AnalysisResponse, CaseSaveResponse, HealthResponse
from orchestrator.pipeline import run_analysis_pipeline
from orchestrator.service_clients import upload_case_to_kg

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- App ----------
settings = get_settings()

app = FastAPI(
    title="JuriAid Orchestrator",
    description="Agentic AI Framework - Central coordinator for JuriAid legal analysis",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------

async def _validate_pdf(file: UploadFile) -> bytes:
    """Validate uploaded file is a proper PDF. Returns bytes."""

    # Check content type
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files accepted",
            )

    # Read bytes
    pdf_bytes = await file.read()

    # Check empty
    if not pdf_bytes or len(pdf_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    # Check size
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Basic PDF header check
    if not pdf_bytes[:5] == b"%PDF-":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file",
        )

    return pdf_bytes


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - no auth required."""
    return HealthResponse()


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_case(
    file: UploadFile = File(..., description="Sri Lankan legal case PDF"),
    prompt: str = Form(default="Analyze this case", description="User prompt for analysis guidance"),
    user: dict = Depends(verify_token),
):
    """
    Main analysis pipeline.

    - Accepts PDF + prompt (no save_for_reference field)
    - LLM dynamically detects if user wants to save based on prompt
    - Runs: PDF extract → parallel(past cases + laws) → LLM summary → questions → synthesis
    - If save intent detected → auto-saves to KG
    """
    logger.info(f"Analyze request from user {user.get('sub')} | file={file.filename}")

    try:
        pdf_bytes = await _validate_pdf(file)

        result = await run_analysis_pipeline(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "upload.pdf",
            user_prompt=prompt,
            user_id=user.get("sub", 0),
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {str(e)}",
        )


@app.post("/api/cases/save", response_model=CaseSaveResponse)
async def save_case(
    file: UploadFile = File(..., description="Legal case PDF to save for future reference"),
    user: dict = Depends(verify_token),
):
    """
    Save case PDF to Knowledge Graph without running full analysis.
    Internally calls POST :8002/admin/upload-case.
    """
    logger.info(f"Save case request from user {user.get('sub')} | file={file.filename}")

    try:
        pdf_bytes = await _validate_pdf(file)

        result = await upload_case_to_kg(pdf_bytes, file.filename or "upload.pdf")

        case_id = result.get("case_id", "")
        if case_id:
            return CaseSaveResponse(saved=True, case_id=case_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Case save returned no case_id",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Case save failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unreachable",
        )