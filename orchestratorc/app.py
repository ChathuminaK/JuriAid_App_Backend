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
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from auth_middleware import verify_token
from orchestrator.pipeline import CaseAnalysisPipeline
from orchestrator.schemas import CaseAnalysisResponse, CaseSaveResponse
from orchestrator.service_clients import past_case_client, ServiceError

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 JuriAid Orchestrator starting...")
    logger.info(f"   PastCase:    {settings.PAST_CASE_SERVICE_URL}")
    logger.info(f"   LawStatKG:   {settings.LAWSTATKG_SERVICE_URL}")
    logger.info(f"   QuestionGen: {settings.QUESTIONGEN_SERVICE_URL}")
    logger.info("✅ Orchestrator ready!")
    yield
    logger.info("👋 Shutting down")


app = FastAPI(
    title="JuriAid Orchestrator",
    description="Agentic AI Framework for Sri Lankan Legal Case Analysis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = CaseAnalysisPipeline()


# ═══════════════════════════════════════════════════════════
#  GET /health
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "JuriAid Orchestrator", "version": "2.0.0"}


# ═══════════════════════════════════════════════════════════
#  POST /api/analyze - Full Case Analysis
# ═══════════════════════════════════════════════════════════

@app.post("/api/analyze", response_model=CaseAnalysisResponse)
async def analyze_case(
    file: UploadFile = File(..., description="Legal case PDF"),
    prompt: str = Form(default="Analyze this case"),
    save_for_reference: bool = Form(default=False),
    user: dict = Depends(verify_token),
):
    """
    Upload a Sri Lankan legal case PDF → get full AI analysis.

    Pipeline: PDF extract → PastCase search → LawStatKG search →
              Gemini summary → QuestionGen → Agent synthesis → Response
    """

    # Validate
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files accepted")

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    logger.info(f"📥 Analyze | user={user.get('sub')} | file={file.filename} | {size_mb:.2f}MB")

    try:
        return await pipeline.analyze(
            file_bytes=file_bytes,
            filename=file.filename,
            user_prompt=prompt,
            save_for_reference=save_for_reference,
            user_id=user.get("sub"),
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Analysis failed: {e}")


# ═══════════════════════════════════════════════════════════
#  POST /api/cases/save - Save Case for Future Reference
# ═══════════════════════════════════════════════════════════

@app.post("/api/cases/save", response_model=CaseSaveResponse)
async def save_case(
    file: UploadFile = File(..., description="Legal case PDF"),
    user: dict = Depends(verify_token),
):
    """
    Save a case PDF to the Knowledge Graph for future reference.
    Calls PastCase /search which auto-indexes the case in Neo4j.
    The case will appear in similarity searches for future analyses.
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files accepted")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

    logger.info(f"💾 Save case | user={user.get('sub')} | file={file.filename}")

    try:
        # Call /search which auto-indexes to Neo4j
        result = await past_case_client.search_similar(file_bytes, file.filename)

        return CaseSaveResponse(
            saved=True,
            case_id=result.get("new_case_id"),
            message="Case saved and indexed for future reference",
        )
    except ServiceError as e:
        raise HTTPException(e.status_code, e.detail)
    except Exception as e:
        logger.error(f"❌ Save failed: {e}", exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Save failed: {e}")