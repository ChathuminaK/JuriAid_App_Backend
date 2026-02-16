"""
JuriAid Orchestrator – Tool definitions
========================================
Every tool is an **async** function decorated with ``@tool`` so that
LangChain / LangGraph can discover and invoke them automatically.

Each tool calls the corresponding microservice via HTTP.  If the
microservice is unreachable the tool falls back to static / local data
and logs a warning — the agent will still get *some* result it can
reason over.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool

from config import settings

logger = logging.getLogger("juriaid.tools")

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


async def _post_json(url: str, payload: dict) -> dict | list | None:
    """POST JSON and return parsed body, or *None* on failure."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("POST %s failed: %s", url, exc)
        return None


async def _get_json(url: str, params: dict | None = None) -> dict | list | None:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("GET %s failed: %s", url, exc)
        return None


# =========================================================================
# Tool 1 – Search Law Statutes  (LawStatKG service)
# =========================================================================

_FALLBACK_STATUTES: List[Dict[str, Any]] = [
    {
        "id": "1",
        "code": "Divorce Act (Chapter 51) - Section 2",
        "title": "Grounds for Divorce - Cruelty",
        "description": "Defines cruelty as grounds for divorce under Sri Lankan matrimonial law",
        "applicability": "High",
        "keyPoints": [
            "Physical violence causing bodily harm",
            "Mental cruelty causing reasonable apprehension of harm",
            "Persistent pattern of conduct making cohabitation impossible",
        ],
    },
    {
        "id": "2",
        "code": "Divorce Act - Section 3",
        "title": "Malicious Abandonment",
        "description": "Addresses abandonment as grounds for divorce without reasonable cause",
        "applicability": "High",
        "keyPoints": [
            "Intentional separation for one year or more",
            "Abandonment must be without reasonable cause",
            "Failure to provide maintenance during abandonment",
        ],
    },
    {
        "id": "3",
        "code": "Maintenance Act (Chapter 37)",
        "title": "Maintenance for Wife and Children",
        "description": "Governs financial support obligations for dependents",
        "applicability": "High",
        "keyPoints": [
            "Court considers earning capacity of both parties",
            "Standard of living during marriage",
            "Needs of children including education and healthcare",
        ],
    },
]


@tool
async def search_law_statutes(query: str) -> str:
    """Search Sri Lankan law statutes via the LawStatKG hybrid-search service.

    Use when you need to find relevant Acts, Sections or legal provisions
    for a given legal question or case facts.  Returns a JSON list of
    scored statute results.

    Args:
        query: Natural-language legal query, e.g. "grounds for divorce cruelty"
    """
    payload = {
        "query": query,
        "jurisdiction": "Sri Lanka",
        "as_of_date": date.today().isoformat(),
        "top_k": 5,
    }
    data = await _post_json(
        f"{settings.LAWSTATKG_SERVICE_URL}/Lawsearch", payload
    )
    if data is not None:
        logger.info("LawStatKG returned %s results", len(data) if isinstance(data, list) else "?")
        return json.dumps(data, indent=2, default=str)

    # ── fallback ──
    logger.warning("LawStatKG unavailable – returning fallback statutes")
    return json.dumps(
        {"source": "fallback_static", "results": _FALLBACK_STATUTES}, indent=2
    )


# =========================================================================
# Tool 2 – Get statute by Act ID  (LawStatKG service)
# =========================================================================

@tool
async def get_statute_by_act(act_id: str, as_of_date: str = "") -> str:
    """Retrieve the full text of a specific Act from the knowledge graph.

    Use when the agent already knows the act_id and needs the complete
    statute content (e.g. all sections of Divorce Act Chapter 51).

    Args:
        act_id: The Act identifier, e.g. "divorce_act_chapter_51"
        as_of_date: Optional ISO date (YYYY-MM-DD). Defaults to today.
    """
    dt = as_of_date or date.today().isoformat()
    data = await _get_json(
        f"{settings.LAWSTATKG_SERVICE_URL}/statute/{act_id}",
        params={"date": dt},
    )
    if data is not None:
        return json.dumps(data, indent=2, default=str)

    return json.dumps({"error": "LawStatKG service unavailable", "act_id": act_id})


# =========================================================================
# Tool 3 – Search Past Cases  (Past-Case Retrieval service)
# =========================================================================

_FALLBACK_PRECEDENTS: List[Dict[str, Any]] = [
    {
        "case_id": "Fernando_v_Fernando_2018",
        "title": "Fernando v Fernando (2018)",
        "citation": "2 SLR 145",
        "court": "Supreme Court of Sri Lanka",
        "relevance": "High",
        "summary": "Landmark case interpreting the definition of cruelty under the Divorce Act.",
        "keyHolding": "Mental cruelty includes conduct that causes reasonable apprehension of danger.",
    },
    {
        "case_id": "Perera_v_Perera_2016",
        "title": "Perera v Perera (2016)",
        "citation": "1 SLR 298",
        "court": "Court of Appeal, Sri Lanka",
        "relevance": "High",
        "summary": "Addressed requirements for proving malicious abandonment.",
        "keyHolding": "Abandonment must be willful and without just cause.",
    },
    {
        "case_id": "Silva_v_Silva_2020",
        "title": "Silva v Silva (2020)",
        "citation": "3 SLR 67",
        "court": "High Court of Colombo",
        "relevance": "Medium",
        "summary": "Established principles for determining maintenance amounts in divorce.",
        "keyHolding": "Maintenance considers standard of living, earning capacity, and children's best interests.",
    },
]


@tool
async def search_past_cases(case_text: str) -> str:
    """Search for legally relevant past court cases using hybrid semantic + citation retrieval.

    Upload the current case text and receive ranked similar precedents.

    Args:
        case_text: The full (or summarised) text of the current case.
    """
    try:
        # Write case text to a temp file, then multipart-upload it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(case_text)
            tmp_path = tmp.name

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            with open(tmp_path, "rb") as fh:
                files = {"file": ("case.txt", fh, "text/plain")}
                resp = await client.post(
                    f"{settings.PAST_CASE_SERVICE_URL}/upload_and_search",
                    files=files,
                    params={"topk": 5},
                )
                resp.raise_for_status()
                data = resp.json()

        os.unlink(tmp_path)
        logger.info("Past-Case service returned %s results", len(data.get("results", [])))
        return json.dumps(data, indent=2, default=str)

    except Exception as exc:
        logger.warning("Past-Case service unavailable: %s – using fallback", exc)
        return json.dumps(
            {"source": "fallback_static", "results": _FALLBACK_PRECEDENTS}, indent=2
        )


# =========================================================================
# Tool 4 – Generate Legal Questions  (QuestionGen service)
# =========================================================================

_FALLBACK_QUESTIONS: List[str] = [
    "What is the exact date of marriage and where was it registered?",
    "What are the specific grounds for seeking divorce (e.g., cruelty, abandonment, adultery)?",
    "Are there any minor children from this marriage? Please provide their names, ages, and living arrangements.",
    "What custody arrangement are you seeking and why?",
    "Describe any incidents of physical violence or mental cruelty with dates and details.",
    "When did the separation occur and who left the matrimonial home?",
    "Has the spouse provided any financial support since separation?",
    "What is your current monthly income, employment status, and financial obligations?",
    "What is your spouse's estimated monthly income and employment details?",
    "Do you have documentation such as medical reports, police complaints, or witness statements?",
    "What property was acquired during the marriage?",
    "Have you attempted mediation, counseling, or reconciliation?",
    "Are there any ongoing court cases involving you and your spouse?",
    "What is your preferred visitation arrangement if custody is not granted to you?",
]


@tool
async def generate_legal_questions(
    case_text: str,
    law_context: str = "",
    past_cases_context: str = "",
) -> str:
    """Generate context-aware legal questions to assist case preparation.

    Calls the QuestionGen micro-service which uses Agentic RAG
    (Mistral via Ollama) to produce numbered legal questions.

    Args:
        case_text: The current case summary or facts.
        law_context: Stringified statutes / laws relevant to the case.
        past_cases_context: Stringified past-case search results.
    """
    payload = {
        "case_text": case_text[:3000],
        "law": law_context[:2000] if law_context else "Not provided",
        "cases": past_cases_context[:2000] if past_cases_context else "Not provided",
    }
    data = await _post_json(
        f"{settings.QUESTIONGEN_SERVICE_URL}/generate-questions", payload
    )
    if data is not None and "questions" in data:
        logger.info("QuestionGen returned questions")
        return data["questions"]

    logger.warning("QuestionGen unavailable – returning fallback questions")
    return "\n".join(f"{i}. {q}" for i, q in enumerate(_FALLBACK_QUESTIONS, 1))


# =========================================================================
# Tool 5 – Summarise Case  (local Gemini call – no microservice needed)
# =========================================================================

@tool
async def summarize_case(case_text: str) -> str:
    """Produce a structured JSON summary of a legal case using Gemini AI.

    The summary includes title, type, parties, damages claimed, and
    recommended actions.

    Args:
        case_text: The raw case text to summarise.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("models/gemini-1.5-flash")

        prompt = f"""You are a Sri Lankan legal analyst. Analyse this case and provide a structured summary.

Case Text:
{case_text[:3000]}

Return ONLY valid JSON (no markdown):
{{
  "title": "Brief case title",
  "type": "Case type (e.g. Matrimonial Law, Contract Law)",
  "description": "2-3 sentence comprehensive case description",
  "parties": {{"plaintiff": "...", "defendant": "..."}},
  "dateFilied": "Filing date if mentioned, otherwise Not specified",
  "damagesClaimed": "Relief sought or damages claimed",
  "recommendedActions": [
    {{"id":"1","action":"Action title","priority":"High","description":"Details"}}
  ]
}}"""
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return raw[start : end + 1]

    except Exception as exc:
        logger.warning("Gemini summarisation failed: %s", exc)

    # ── fallback ──
    fallback = {
        "title": "Legal Case Analysis",
        "type": "General Legal Matter",
        "description": case_text[:500] + ("..." if len(case_text) > 500 else ""),
        "parties": {"plaintiff": "Petitioner", "defendant": "Respondent"},
        "dateFilied": "Not specified",
        "damagesClaimed": "To be determined",
        "recommendedActions": [
            {"id": "1", "action": "Review Case Documents", "priority": "High",
             "description": "Collect and review all relevant case documents and evidence"},
        ],
    }
    return json.dumps(fallback, indent=2)


# =========================================================================
# Tool 6 – Update Knowledge Base  (local JSON file)
# =========================================================================

@tool
async def update_knowledge_base(case_text: str, case_type: str = "General") -> str:
    """Save the current case to the local knowledge base for future reference.

    Args:
        case_text: The full case text.
        case_type: Detected case type (e.g. Family Law).
    """
    kb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "knowledge_base.json"
    )
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)

    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {"entries": []}

    keywords = ["divorce", "custody", "maintenance", "contract", "breach", "agreement"]
    tags = [kw for kw in keywords if kw in case_text.lower()]

    entry = {
        "id": len(kb["entries"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "case_type": case_type,
        "length": len(case_text),
        "snippet": case_text[:400],
        "tags": tags,
    }
    kb["entries"].append(entry)

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

    return json.dumps({"status": "updated", "total_entries": len(kb["entries"])})


# =========================================================================
# Convenience list used by the agent to bind tools to the LLM
# =========================================================================

ALL_TOOLS = [
    search_law_statutes,
    get_statute_by_act,
    search_past_cases,
    generate_legal_questions,
    summarize_case,
    update_knowledge_base,
]