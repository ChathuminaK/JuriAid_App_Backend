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
from orchestrator.pdf_extractor import extract_text_from_pdf
from orchestrator.case_validator import validate_divorce_case

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
    description="Agentic AI Framework - Central coordinator for JuriAid legal analysis system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- File Validation ----------

async def _validate_pdf(file: UploadFile) -> bytes:
    """Validate uploaded file is a proper PDF within size limits."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files accepted",
            )

    pdf_bytes = await file.read()

    if not pdf_bytes or len(pdf_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

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
    file: UploadFile = File(..., description="Sri Lankan divorce case PDF. Max 10MB."),
    prompt: str = Form(
        default="Analyze this case",
        description="User prompt for analysis. Include 'save' to store case for future reference.",
    ),
    user: dict = Depends(verify_token),
):
    """
    POST /api/analyze - Main analysis pipeline.

    Validates the PDF is a Sri Lankan divorce/matrimonial case before processing.
    The LLM dynamically detects user intent from the prompt.
    """
    logger.info(f"Analyze request from user {user.get('sub')} | file={file.filename}")

    try:
        # Step 1: Validate PDF file (size, format)
        pdf_bytes = await _validate_pdf(file)

        # Step 2: Quick text extraction for validation
        case_text = extract_text_from_pdf(pdf_bytes)

        # Step 3: Validate this is a Sri Lankan divorce case
        is_valid, validation_details = validate_divorce_case(case_text)
        if not is_valid:
            logger.warning(
                f"Case validation failed for '{file.filename}': "
                f"{validation_details.get('reason')}"
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid case type",
                    "message": validation_details.get("reason", ""),
                    "matched_keywords": validation_details.get("matched_keywords", 0),
                    "hint": "Please upload a Sri Lankan divorce plaint, answer, or judgment PDF.",
                },
            )

        # Step 4: Run full analysis pipeline
        result = await run_analysis_pipeline(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "upload.pdf",
            user_prompt=prompt,
            user_id=user.get("sub", 0),
            pre_extracted_text=case_text,
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