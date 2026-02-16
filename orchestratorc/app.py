"""
JuriAid Orchestrator – FastAPI application
============================================
Primary entry point for the Agentic AI backend.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from orchestrator.core import (
    OUTPUTS_DIR,
    UPLOADS_DIR,
    analyze_case_content,
    generate_legal_summary,
    process_single_file,
    read_input_file,
)
from orchestrator.agent_gemini import GeminiPlannerAgent
from orchestrator.agent_langchain import run_agent, get_chat_history
from auth_middleware import verify_user_token

# ---------------------------------------------------------------------------
# Initialise agents
# ---------------------------------------------------------------------------
planner_agent = GeminiPlannerAgent(
    kb_path=os.path.join(OUTPUTS_DIR, "knowledge_base.json")
)

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="JuriAid Orchestrator API",
    version="2.0.0",
    description="Agentic AI legal-case analysis orchestrator for Sri Lankan law.",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str


class LawSearchRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = "Sri Lanka"
    as_of_date: Optional[str] = None
    top_k: int = 5


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    """Return service status and a quick liveness probe for each dependency."""

    async def _ping(url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(url)
                return "ok" if r.status_code < 500 else "degraded"
        except Exception:
            return "unreachable"

    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "agent": "langgraph-gemini-1.5-flash",
        "services": {
            "auth": await _ping(f"{settings.AUTH_SERVICE_URL}/"),
            "lawstatkg": await _ping(f"{settings.LAWSTATKG_SERVICE_URL}/docs"),
            "past_case": await _ping(f"{settings.PAST_CASE_SERVICE_URL}/health"),
            "questiongen": await _ping(f"{settings.QUESTIONGEN_SERVICE_URL}/docs"),
        },
    }


# =========================================================================
# PRIMARY  –  /api/chat   (LangGraph agent)
# =========================================================================
@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    user: dict = Depends(verify_user_token),
):
    """Multi-turn conversational endpoint powered by the LangGraph agent.

    Accepts an optional file (PDF/TXT). The agent decides which
    micro-services to call, then returns an executive summary.
    """
    case_text: str | None = None

    if file and file.filename:
        if not file.filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(400, "Only PDF or TXT files supported.")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file.")
        ts_name = f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}"
        dest = os.path.join(UPLOADS_DIR, ts_name)
        with open(dest, "wb") as fh:
            fh.write(data)
        case_text = read_input_file(dest)

    result = await run_agent(query=query, session_id=session_id, case_text=case_text)
    result["user_id"] = user.get("user_id")
    result["email"] = user.get("email")
    return JSONResponse(result)


# Also accept a pure-JSON body (no file)
@app.post("/api/chat/json")
async def chat_json(
    body: ChatRequest,
    user: dict = Depends(verify_user_token),
):
    """JSON-only variant of /api/chat (no file upload)."""
    result = await run_agent(query=body.query, session_id=body.session_id)
    result["user_id"] = user.get("user_id")
    return JSONResponse(result)


# =========================================================================
# Chat history
# =========================================================================
@app.get("/api/chat/history/{session_id}")
async def chat_history(session_id: str, user: dict = Depends(verify_user_token)):
    """Return the full conversation history for a session from Redis."""
    return {"session_id": session_id, "messages": get_chat_history(session_id)}


# =========================================================================
# Direct proxy  –  Law search  (no agent reasoning)
# =========================================================================
@app.post("/api/search/laws")
async def search_laws(body: LawSearchRequest, user: dict = Depends(verify_user_token)):
    """Proxy to LawStatKG /Lawsearch – no agent reasoning, direct pass-through."""
    payload = {
        "query": body.query,
        "jurisdiction": body.jurisdiction,
        "as_of_date": body.as_of_date or datetime.now().strftime("%Y-%m-%d"),
        "top_k": body.top_k,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{settings.LAWSTATKG_SERVICE_URL}/Lawsearch", json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(502, f"LawStatKG service error: {exc}")


# =========================================================================
# Direct proxy  –  Past-case search  (no agent reasoning)
# =========================================================================
@app.post("/api/search/cases")
async def search_cases(
    file: UploadFile = File(...),
    topk: int = Form(5),
    user: dict = Depends(verify_user_token),
):
    """Proxy to Past-Case Retrieval – upload & search."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                f"{settings.PAST_CASE_SERVICE_URL}/upload_and_search",
                files={"file": (file.filename, data, file.content_type or "application/pdf")},
                params={"topk": topk},
            )
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        raise HTTPException(502, f"Past-Case service error: {exc}")


# =========================================================================
# Legacy endpoints  (kept for backwards compatibility)
# =========================================================================
@app.post("/api/upload-case")
async def upload_case(
    file: UploadFile = File(...),
    user: dict = Depends(verify_user_token),
):
    if not file or not getattr(file, "filename", "").strip():
        raise HTTPException(422, "Missing file upload.")
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF or TXT files supported.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file content.")

    ts_name = f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}"
    dest = os.path.join(UPLOADS_DIR, ts_name)
    with open(dest, "wb") as fh:
        fh.write(data)

    result = process_single_file(dest)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "Processing failed"))
    result["user_id"] = user["user_id"]
    result["subscription_tier"] = user.get("subscription_tier")
    result["email"] = user.get("email")
    return JSONResponse(result)


@app.post("/api/upload-case-with-prompt")
async def upload_case_with_prompt(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    user: dict = Depends(verify_user_token),
):
    """Upload + prompt → runs the **LangGraph agent** (fixed: prompt is now used)."""
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF or TXT files supported.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    ts_name = f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}"
    dest = os.path.join(UPLOADS_DIR, ts_name)
    with open(dest, "wb") as fh:
        fh.write(data)

    case_text = read_input_file(dest)
    session_id = f"upload_{datetime.now():%Y%m%d%H%M%S}_{user.get('user_id', 'anon')}"
    result = await run_agent(query=prompt, session_id=session_id, case_text=case_text)
    result["user_id"] = user["user_id"]
    return JSONResponse(result)


@app.post("/api/agent/plan-run")
async def agent_plan_run(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    user: dict = Depends(verify_user_token),
):
    """Legacy Gemini planner agent (kept as fallback)."""
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF or TXT files supported.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")

    tmp = os.path.join(UPLOADS_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{file.filename}")
    with open(tmp, "wb") as fh:
        fh.write(data)

    case_text = read_input_file(tmp)
    try:
        out = planner_agent.run(case_text, prompt)
        out["user_id"] = user["user_id"]
        out["email"] = user.get("email")
        return JSONResponse(out)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/analyze-text")
async def analyze_text(request: Request):
    """Public: quick keyword-based analysis (no auth)."""
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty.")
    analysis = analyze_case_content(text)
    summary = generate_legal_summary(text, analysis)
    return {
        "success": True,
        "case_type": analysis["case_type"],
        "length": analysis["length"],
        "preview": text[:500] + ("..." if len(text) > 500 else ""),
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(
        status_code=500, content={"success": False, "detail": str(exc)}
    )