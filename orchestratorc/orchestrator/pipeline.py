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
    """Format similar cases into text for LLM context."""
    cases = cases_data.get("similar_cases", [])
    if not cases:
        return "No similar past cases found."

    parts = []
    for i, c in enumerate(cases[:5], 1):
        name = c.get("case_name", "Unknown Case")
        score = c.get("score", 0)
        reason = c.get("reason", "")
        summary = c.get("summary", "")

        # Use first 300 chars of summary if no complaint/defense
        complaint = c.get("complaint", "")
        if not complaint and summary:
            complaint = summary[:300]

        parts.append(
            f"{i}. {name} (similarity: {score:.2f})"
            + (f" - {reason}" if reason else "")
            + (f"\n   Details: {complaint}" if complaint else "")
        )

    return "\n".join(parts)


def _format_laws_text(laws_data: dict) -> str:
    """Format applicable laws into text for LLM context."""
    laws = laws_data.get("applicable_laws", [])
    if not laws:
        return "No applicable laws retrieved."

    personal_law = laws_data.get("personal_law", "Unknown")
    parts = [f"Personal Law System: {personal_law}\n"]

    for law in laws[:10]:
        title = law.get("title", "Unknown")
        section = law.get("section", "")
        section_title = law.get("section_title", "")
        content = law.get("content", "")
        relevance = law.get("relevance_score", 0)

        line = f"- {title}"
        if section:
            line += f" Section {section}"
        if section_title:
            line += f" ({section_title})"
        line += f" [relevance: {relevance:.3f}]"
        if content:
            line += f": {content}"
        parts.append(line)

    return "\n".join(parts)


def _parse_similar_cases(cases_data: dict) -> list[SimilarCase]:
    """Parse raw cases data into schema objects."""
    result = []
    for c in cases_data.get("similar_cases", []):
        result.append(
            SimilarCase(
                case_id=str(c.get("case_id", "")),
                case_name=str(c.get("case_name", "")),
                score=float(c.get("score", 0)),
                summary=str(c.get("summary", "")),
                reason=str(c.get("reason", "")),
                complaint=str(c.get("complaint", "")),
                defense=str(c.get("defense", "")),
            )
        )
    return result


def _parse_relevant_laws(laws_data: dict) -> list[RelevantLaw]:
    """Parse raw laws data into schema objects."""
    result = []
    for law in laws_data.get("applicable_laws", []):
        try:
            result.append(
                RelevantLaw(
                    act_id=str(law.get("act_id", "")),
                    title=str(law.get("title", "")),
                    section=str(law.get("section", "")),
                    relevance_score=float(law.get("relevance_score", 0)),
                    content=str(law.get("content", "")),
                )
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping malformed law entry: {e}")
    logger.info(f"Parsed {len(result)} relevant laws for response")
    return result


def _parse_questions(questions_data: dict) -> list[GeneratedQuestion]:
    """Parse raw questions string into schema objects."""
    raw = questions_data.get("questions", "")
    if not raw:
        return []

    questions = []
    q_id = 0
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
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
    pre_extracted_text: Optional[str] = None,
) -> AnalysisResponse:
    """
    Main orchestration pipeline.
    Accepts optional pre_extracted_text to avoid double PDF extraction.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    session_id = session_id or analysis_id

    logger.info(f"[{analysis_id}] Starting analysis pipeline for '{filename}'")

    # --- Step 1: Extract PDF text (or use pre-extracted) ---
    if pre_extracted_text:
        case_text = pre_extracted_text
        logger.info(f"[{analysis_id}] Step 1: Using pre-extracted text ({len(case_text)} chars)")
    else:
        logger.info(f"[{analysis_id}] Step 1: Extracting PDF text")
        case_text = extract_text_from_pdf(pdf_bytes)

    if not case_text:
        logger.warning(f"[{analysis_id}] No text extracted from PDF")
        case_text = "(No text could be extracted from the uploaded PDF document)"

    file_size_mb = round(len(pdf_bytes) / (1024 * 1024), 2)

    # --- Step 2: Detect user intent ---
    logger.info(f"[{analysis_id}] Step 2: Detecting user intent")
    intent = await detect_user_intent(user_prompt)
    should_save = intent.get("should_save_case", False)
    logger.info(
        f"[{analysis_id}] Intent: save={should_save}, focus={intent.get('analysis_focus')}"
    )

    # --- Step 3: Get conversation history ---
    conversation_history = get_conversation_history(session_id)
    save_conversation(session_id, "user", f"[File: {filename}] {user_prompt}")

    # --- Step 4: Parallel calls ---
    logger.info(f"[{analysis_id}] Step 4: Parallel service calls (past cases + laws)")

    cases_task = search_similar_cases(pdf_bytes, filename)
    laws_task = get_applicable_laws(pdf_bytes, filename)

    results = await asyncio.gather(cases_task, laws_task, return_exceptions=True)

    if isinstance(results[0], Exception):
        logger.error(f"[{analysis_id}] Past case search exception: {results[0]}")
        cases_data = {"similar_cases": []}
    else:
        cases_data = results[0]

    if isinstance(results[1], Exception):
        logger.error(f"[{analysis_id}] Law search exception: {results[1]}")
        laws_data = {"applicable_laws": []}
    else:
        laws_data = results[1]

    cases_count = len(cases_data.get("similar_cases", []))
    laws_count = len(laws_data.get("applicable_laws", []))
    logger.info(f"[{analysis_id}] Service results: {cases_count} cases, {laws_count} laws")

    cases_text = _format_cases_text(cases_data)
    laws_text = _format_laws_text(laws_data)

    # --- Step 5: Generate case summary via LLM ---
    logger.info(f"[{analysis_id}] Step 5: Generating case summary via LLM")
    case_summary = await generate_case_summary(
        case_text=case_text,
        user_prompt=user_prompt,
        similar_cases_text=cases_text,
        laws_text=laws_text,
        conversation_history=conversation_history,
    )

    # --- Step 6: Generate questions ---
    logger.info(f"[{analysis_id}] Step 6: Generating legal questions (may take several minutes)")
    questions_data = await generate_questions(case_text, laws_text, cases_text)

    # --- Step 7: Final synthesis ---
    logger.info(f"[{analysis_id}] Step 7: Final synthesis")
    raw_questions = questions_data.get("questions", "")
    final_summary = await synthesize_analysis(case_summary, raw_questions, user_prompt)

    # --- Step 8: Dynamic save ---
    saved_for_reference = False
    if should_save:
        logger.info(f"[{analysis_id}] Step 8: Saving case to KG")
        try:
            save_result = await upload_case_to_kg(pdf_bytes, filename)
            saved_for_reference = bool(save_result.get("case_id"))
        except Exception as e:
            logger.error(f"[{analysis_id}] Case save failed (non-blocking): {e}")

    # --- Save to memory ---
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