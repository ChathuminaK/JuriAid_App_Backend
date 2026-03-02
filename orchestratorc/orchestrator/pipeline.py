"""
JuriAid Case Analysis Pipeline - v2.1 (Resilient)
"""

import asyncio
import time
import uuid
import logging
from typing import Optional

from orchestrator.pdf_extractor import extract_text_from_pdf
from orchestrator.service_clients import (
    search_similar_cases,
    get_applicable_laws,
    generate_questions,
    upload_case_to_kg,
)
from orchestrator.agent import (
    detect_user_intent,
    generate_case_summary,
    synthesize_analysis,
)
from orchestrator.memory_manager import (
    save_conversation,
    get_conversation_history,
)
from orchestrator.schemas import (
    AnalysisResponse,
    AnalysisMetadata,
    SimilarCase,
    RelevantLaw,
    GeneratedQuestion,
)

logger = logging.getLogger(__name__)


def _format_cases_text(cases_data: dict) -> str:
    """Format similar cases into readable text for LLM context."""
    cases = cases_data.get("similar_cases", [])
    if not cases:
        return "No similar past cases found."

    parts = []
    for i, c in enumerate(cases[:5], 1):
        name = _clean_case_name(c.get("case_name", ""), c.get("case_id", ""))
        complaint = c.get("complaint", "N/A") or "N/A"
        defense = c.get("defense", "N/A") or "N/A"
        score = c.get("score", 0)
        parts.append(
            f"{i}. {name} (similarity: {score})\n"
            f"   Complaint: {complaint}\n"
            f"   Defense: {defense}"
        )

    return "\n".join(parts)


def _format_laws_text(laws_data: dict) -> str:
    """Format applicable laws into readable text for LLM context."""
    laws = laws_data.get("applicable_laws", [])
    if not laws:
        return "No applicable laws retrieved."

    parts = []
    for law in laws[:10]:
        title = law.get("title", "Unknown")
        section = law.get("section", "")
        content = law.get("content", "")
        score = law.get("relevance_score", 0)
        parts.append(f"- {title} {section} (relevance: {score}): {content}")

    return "\n".join(parts)


def _clean_case_name(name: str, case_id: str = "") -> str:
    """
    Clean case names that come as 'Page 1' or empty from PastCase service.
    Falls back to a readable case ID reference.
    """
    name = (name or "").strip()

    # Check if name is useless (Page X, empty, etc.)
    if not name or name.lower().startswith("page") or len(name) < 5:
        if case_id:
            short_id = str(case_id)[:8]
            return f"Case Ref: {short_id}"
        return "Unnamed Case"

    return name


def _parse_similar_cases(cases_data: dict) -> list[SimilarCase]:
    """Parse raw cases data into schema objects with cleaned names."""
    result = []
    for c in cases_data.get("similar_cases", []):
        case_id = str(c.get("case_id", ""))
        raw_name = str(c.get("case_name", ""))

        result.append(SimilarCase(
            case_id=case_id,
            case_name=_clean_case_name(raw_name, case_id),
            score=float(c.get("score", 0)),
            complaint=str(c.get("complaint", "") or ""),
            defense=str(c.get("defense", "") or ""),
        ))
    return result


def _parse_relevant_laws(laws_data: dict) -> list[RelevantLaw]:
    """Parse raw laws data into schema objects."""
    result = []
    for law in laws_data.get("applicable_laws", []):
        result.append(RelevantLaw(
            act_id=str(law.get("act_id", "")),
            title=str(law.get("title", "")),
            section=str(law.get("section", "")),
            relevance_score=float(law.get("relevance_score", 0)),
            content=str(law.get("content", "")),
        ))
    return result


def _parse_questions(questions_data: dict) -> list[GeneratedQuestion]:
    """Parse raw questions string into structured schema objects."""
    raw = questions_data.get("questions", "")
    if not raw:
        return []

    questions = []
    q_id = 0
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Remove leading number/bullet patterns like "1.", "1)", "- ", "* "
        clean = line.lstrip("0123456789.-)*• ").strip()
        if clean and len(clean) > 10:
            q_id += 1
            questions.append(GeneratedQuestion(question_id=q_id, question=clean))

    return questions


async def run_analysis_pipeline(
    pdf_bytes: bytes,
    filename: str,
    user_prompt: str,
    user_id: int,
    session_id: Optional[str] = None,
) -> AnalysisResponse:
    """
    Main orchestration pipeline - the Agentic AI Framework coordinator.

    Steps:
      1. Extract text from PDF (local - PyMuPDF)
      2. Detect user intent via LLM (save? focus?)
      3. Parallel: search past cases + get applicable laws
      4. Generate case summary via LLM (with cases + laws context)
      5. Generate questions (sequential - needs case text + laws + cases)
      6. Final synthesis via LLM (refine summary with questions)
      7. If save intent → upload to KG (non-blocking)
      8. Save to conversation memory

    Graceful degradation: if any service fails, pipeline continues with partial results.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    session_id = session_id or analysis_id

    logger.info(f"[{analysis_id}] Starting analysis pipeline for '{filename}'")

    # --- Step 1: Extract PDF text ---
    logger.info(f"[{analysis_id}] Step 1: Extracting PDF text")
    case_text = extract_text_from_pdf(pdf_bytes)

    if not case_text:
        logger.warning(f"[{analysis_id}] No text extracted from PDF")
        case_text = "(No text could be extracted from the uploaded PDF document)"

    file_size_mb = round(len(pdf_bytes) / (1024 * 1024), 2)

    # --- Step 2: Detect user intent via LLM ---
    logger.info(f"[{analysis_id}] Step 2: Detecting user intent")
    intent = await detect_user_intent(user_prompt)
    should_save = intent.get("should_save_case", False)
    logger.info(
        f"[{analysis_id}] Intent: save={should_save}, "
        f"focus={intent.get('analysis_focus')}, "
        f"topics={intent.get('key_topics')}"
    )

    # --- Step 3: Get conversation history for context ---
    conversation_history = get_conversation_history(session_id)

    # Save user message to memory
    save_conversation(session_id, "user", f"[File: {filename}] {user_prompt}")

    # --- Step 4: Parallel calls to specialist agents ---
    logger.info(f"[{analysis_id}] Step 3-4: Parallel calls → PastCase + LawStatKG")

    cases_task = search_similar_cases(pdf_bytes, filename)
    laws_task = get_applicable_laws(pdf_bytes, filename)

    results = await asyncio.gather(cases_task, laws_task, return_exceptions=True)

    # Handle exceptions from gather
    cases_data = results[0] if not isinstance(results[0], Exception) else {"similar_cases": []}
    laws_data = results[1] if not isinstance(results[1], Exception) else {"applicable_laws": []}

    if isinstance(results[0], Exception):
        logger.error(f"[{analysis_id}] Past case search exception: {results[0]}")
    if isinstance(results[1], Exception):
        logger.error(f"[{analysis_id}] Law search exception: {results[1]}")

    # Format for LLM context
    cases_text = _format_cases_text(cases_data)
    laws_text = _format_laws_text(laws_data)

    # --- Step 5: Generate case summary via LLM (Reasoning Engine) ---
    logger.info(f"[{analysis_id}] Step 5: Generating case summary via Gemini 2.5")
    case_summary = await generate_case_summary(
        case_text=case_text,
        user_prompt=user_prompt,
        similar_cases_text=cases_text,
        laws_text=laws_text,
        conversation_history=conversation_history,
    )

    # --- Step 6: Generate questions via QuestionGen agent ---
    logger.info(f"[{analysis_id}] Step 6: Generating legal questions (may take several minutes)")
    questions_data = await generate_questions(case_text, laws_text, cases_text)

    # --- Step 7: Final synthesis via LLM ---
    logger.info(f"[{analysis_id}] Step 7: Final synthesis")
    raw_questions = questions_data.get("questions", "")
    final_summary = await synthesize_analysis(case_summary, raw_questions, user_prompt)

    # --- Step 8: Dynamic save decision based on user intent ---
    saved_for_reference = False
    if should_save:
        logger.info(f"[{analysis_id}] Step 8: User intent → saving case to KG")
        try:
            save_result = await upload_case_to_kg(pdf_bytes, filename)
            saved_for_reference = bool(save_result.get("case_id"))
            if saved_for_reference:
                logger.info(f"[{analysis_id}] Case saved: {save_result.get('case_id')}")
        except Exception as e:
            logger.error(f"[{analysis_id}] Case save failed (non-blocking): {e}")

    # --- Save assistant response to memory ---
    save_conversation(session_id, "assistant", final_summary[:2000])

    # --- Build response ---
    processing_time = round(time.time() - start_time, 2)
    logger.info(f"[{analysis_id}] Pipeline completed in {processing_time}s")

    return AnalysisResponse(
        analysis_id=analysis_id,
        status="completed",
        case_summary=final_summary,
        similar_cases=_parse_similar_cases(cases_data),
        relevant_laws=_parse_relevant_laws(laws_data),
        generated_questions=_parse_questions(questions_data),
        metadata=AnalysisMetadata(
            filename=filename,
            file_size_mb=file_size_mb,
            text_length=len(case_text),
            user_id=user_id,
            user_prompt=user_prompt,
            saved_for_reference=saved_for_reference,
        ),
        processing_time_seconds=processing_time,
    )