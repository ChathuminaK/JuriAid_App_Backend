"""
JuriAid Case Analysis Pipeline - v2.3 (Multi-Agent Orchestration)

Multi-Agent Architecture:
  - OrchestratorAgent: Central coordinator (this pipeline)
  - IntentDetectionAgent: Analyzes user intent via Gemini LLM
  - CaseRetrievalAgent: Finds similar past cases (Port 8002)
  - LawRetrievalAgent: Retrieves applicable statutes (Port 8003)
  - SummaryAgent: Generates case summary via Gemini LLM
  - QuestionGenAgent: Generates investigation questions (Port 8004)
  - SynthesisAgent: Final report synthesis via Gemini LLM
  - MemoryAgent: Manages short-term + long-term conversation memory
  - ValidationAgent: Validates uploaded case is Sri Lankan divorce
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
    get_case_judgment,
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

logger = logging.getLogger("orchestrator_agent")


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
        preview = c.get("judgment_preview", "")

        parts.append(
            f"{i}. {name} (similarity: {score:.2f})"
            + (f" - {reason}" if reason else "")
            + (f"\n   Preview: {preview[:300]}" if preview else "")
        )

    return "\n".join(parts)


def _format_laws_text(laws_data: dict) -> str:
    """Format applicable case laws into text for LLM context."""
    laws = laws_data.get("relevant_case_laws", [])
    if not laws:
        return "No applicable case laws retrieved."

    parts = []
    for law in laws[:10]:
        case_name = law.get("case_name", "Unknown")
        section_number = law.get("section_number", "")
        section_title = law.get("section_title", "")
        principles = law.get("principle", [])
        confidence = law.get("confidence_score", 0)

        line = f"- {case_name}"
        if section_number:
            line += f" Section {section_number}"
        if section_title:
            line += f" ({section_title})"
        line += f" [confidence: {confidence:.3f}]"
        if principles:
            line += f": {'; '.join(str(p) for p in principles[:2])}"
        parts.append(line)

    return "\n".join(parts)


def _format_laws_for_questions(laws_data: dict) -> str:
    """Format case_name + principle for QuestionGen input (law field)."""
    laws = laws_data.get("relevant_case_laws", [])
    if not laws:
        return "No applicable laws found."

    parts = []
    for law in laws[:10]:
        case_name = law.get("case_name", "Unknown")
        principles = law.get("principle", [])
        principle_text = "; ".join(str(p) for p in principles) if principles else ""

        line = f"- {case_name}"
        if principle_text:
            line += f": {principle_text}"
        parts.append(line)

    return "\n".join(parts)


async def _format_cases_for_questions(cases_data: dict) -> str:
    """Fetch judgment text for each similar case for QuestionGen input (cases field)."""
    cases = cases_data.get("similar_cases", [])
    if not cases:
        return "No similar past cases found."

    # Fetch full judgment text for each case in parallel
    case_ids = [c.get("case_id", "") for c in cases[:5] if c.get("case_id")]
    tasks = [get_case_judgment(cid) for cid in case_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    judgment_map = {}
    for cid, result in zip(case_ids, results):
        if not isinstance(result, Exception) and result:
            judgment_map[cid] = result

    parts = []
    for i, c in enumerate(cases[:5], 1):
        case_id = c.get("case_id", "")
        case_name = c.get("case_name", "Unknown")

        detail = judgment_map.get(case_id, {})
        judgment = detail.get("judgment", "")
        if judgment:
            parts.append(f"{i}. {case_name}:\n{judgment[:1000]}")
        else:
            preview = c.get("judgment_preview", "")
            if preview:
                parts.append(f"{i}. {case_name}:\n{preview[:500]}")

    if not parts:
        return "No similar past cases found."

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
                reason=str(c.get("reason", "")),
                judgment_preview=str(c.get("judgment_preview", "")),
                shared_issues=c.get("shared_issues", []),
                breakdown=c.get("breakdown", {}),
                view_case_details=str(c.get("view_case_details", "")),
                view_full_case_file=str(c.get("view_full_case_file", "")),
            )
        )
    return result


def _parse_relevant_laws(laws_data: dict) -> list[RelevantLaw]:
    """Parse raw case laws data into schema objects."""
    result = []
    for law in laws_data.get("relevant_case_laws", []):
        try:
            result.append(
                RelevantLaw(
                    case_id=str(law.get("case_id", "")),
                    case_name=str(law.get("case_name", "")),
                    citation=str(law.get("citation", "")),
                    topic=str(law.get("topic", "")),
                    section_number=str(law.get("section_number", "") or ""),
                    section_title=str(law.get("section_title", "") or ""),
                    principle=law.get("principle", []) or [],
                    held=law.get("held", []) or [],
                    facts=str(law.get("facts", "") or ""),
                    referenced_laws=law.get("relevant_laws", []) or [],
                    relevant_sections=law.get("relevant_sections", []) or [],
                    court=str(law.get("court", "") or ""),
                    amending_law=str(law.get("amending_law", "") or ""),
                    confidence_score=float(law.get("confidence_score", 0)),
                    support_score=float(law.get("support_score", 0)),
                    query_hits=int(law.get("query_hits", 0)),
                    detail_url=str(law.get("detail_url", "")),
                )
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"[OrchestratorAgent] Skipping malformed law entry: {e}")
    logger.info(f"[OrchestratorAgent] Parsed {len(result)} relevant laws for response")
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
    Main multi-agent orchestration pipeline.

    Agent Flow:
      OrchestratorAgent → IntentDetectionAgent → MemoryAgent →
      [CaseRetrievalAgent ∥ LawRetrievalAgent] → SummaryAgent →
      QuestionGenAgent → SynthesisAgent → MemoryAgent
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    session_id = session_id or analysis_id

    logger.info(f"[OrchestratorAgent] ═══════════════════════════════════════════")
    logger.info(f"[OrchestratorAgent] Starting multi-agent pipeline | id={analysis_id}")
    logger.info(f"[OrchestratorAgent] File: {filename} | User: {user_id}")
    logger.info(f"[OrchestratorAgent] Framework: LangChain + Gemini 2.5 Flash LLM")
    logger.info(f"[OrchestratorAgent] Architecture: 9 specialized agents coordinated")
    logger.info(f"[OrchestratorAgent] ═══════════════════════════════════════════")

    # --- Step 1: PDF Text Extraction ---
    if pre_extracted_text:
        case_text = pre_extracted_text
        logger.info(
            f"[OrchestratorAgent] Step 1/7: PDF text pre-extracted "
            f"({len(case_text)} chars)"
        )
    else:
        logger.info(f"[OrchestratorAgent] Step 1/7: Extracting PDF text")
        case_text = extract_text_from_pdf(pdf_bytes)

    if not case_text:
        logger.warning(f"[OrchestratorAgent] No text extracted from PDF")
        case_text = "(No text could be extracted from the uploaded PDF document)"

    file_size_mb = round(len(pdf_bytes) / (1024 * 1024), 2)

    # --- Step 2: Intent Detection Agent ---
    logger.info(f"[IntentDetectionAgent] Step 2/7: Analyzing user intent via LangChain → Gemini LLM")
    intent = await detect_user_intent(user_prompt)
    should_save = intent.get("should_save_case", False)
    focus = intent.get("analysis_focus", "general")
    logger.info(
        f"[IntentDetectionAgent] LangChain agent result: save={should_save}, focus='{focus}'"
    )

    # --- Step 3: Memory Agent — Load conversation history ---
    logger.info(f"[MemoryAgent] Step 3/7: Loading conversation context for session {session_id[:8]}...")
    conversation_history = get_conversation_history(session_id)
    if conversation_history:
        logger.info(f"[MemoryAgent] Found existing conversation history — multi-turn context active")
    else:
        logger.info(f"[MemoryAgent] New session — no prior conversation history")
    save_conversation(session_id, "user", f"[File: {filename}] {user_prompt}")

    # --- Step 4: Parallel Agent Delegation ---
    logger.info(f"[OrchestratorAgent] Step 4/7: Delegating to specialist agents in parallel")
    logger.info(f"[CaseRetrievalAgent] → Searching similar past cases (Neo4j + FAISS)")
    logger.info(f"[LawRetrievalAgent] → Retrieving applicable statutes (LegalKG + Embeddings)")

    cases_task = search_similar_cases(pdf_bytes, filename)
    laws_task = get_applicable_laws(pdf_bytes, filename)

    results = await asyncio.gather(cases_task, laws_task, return_exceptions=True)

    if isinstance(results[0], Exception):
        logger.error(f"[CaseRetrievalAgent] ✗ Failed: {results[0]}")
        cases_data = {"similar_cases": []}
    else:
        cases_data = results[0]

    if isinstance(results[1], Exception):
        logger.error(f"[LawRetrievalAgent] ✗ Failed: {results[1]}")
        laws_data = {"relevant_case_laws": []}
    else:
        laws_data = results[1]

    cases_count = len(cases_data.get("similar_cases", []))
    laws_count = len(laws_data.get("relevant_case_laws", []))
    logger.info(f"[CaseRetrievalAgent] ✓ Found {cases_count} similar past cases")
    logger.info(f"[LawRetrievalAgent] ✓ Found {laws_count} applicable legal provisions")

    # Format for LLM context
    cases_text = _format_cases_text(cases_data)
    laws_text = _format_laws_text(laws_data)

    # --- Step 5: Summary Agent ---
    logger.info(f"[SummaryAgent] Step 5/7: Generating case analysis via LangChain → Gemini LLM agent")
    logger.info(f"[SummaryAgent] LangChain context injection: {len(case_text)} chars case text, {cases_count} cases, {laws_count} laws")
    case_summary = await generate_case_summary(
        case_text=case_text,
        user_prompt=user_prompt,
        similar_cases_text=cases_text,
        laws_text=laws_text,
        conversation_history=conversation_history,
    )
    logger.info(f"[SummaryAgent] ✓ Generated case summary ({len(case_summary)} chars)")

    # --- Step 6: Question Generation Agent ---
    logger.info(f"[QuestionGenAgent] Step 6/7: Generating investigation questions via Legal-BERT + LoRA")

    # Format law: case_name + principle from relevant case laws
    q_laws_text = _format_laws_for_questions(laws_data)

    # Format cases: fetch judgment text from /case/{case_id} for each similar case
    q_cases_text = await _format_cases_for_questions(cases_data)

    logger.info(f"[QuestionGenAgent] Input: case_summary ({len(case_summary[:5000])} chars), "
                f"laws ({len(q_laws_text)} chars), cases ({len(q_cases_text)} chars)")

    questions_data = await generate_questions(case_summary, q_laws_text, q_cases_text)
    q_count = len(questions_data.get("questions", "").strip().split("\n"))
    logger.info(f"[QuestionGenAgent] ✓ Generated ~{q_count} investigation questions")

    # --- Step 7: Synthesis Agent ---
    logger.info(f"[SynthesisAgent] Step 7/7: Final report synthesis via LangChain → Gemini LLM agent")
    raw_questions = questions_data.get("questions", "")
    final_summary = await synthesize_analysis(case_summary, raw_questions, user_prompt)
    logger.info(f"[SynthesisAgent] ✓ Final analysis synthesized ({len(final_summary)} chars)")

    # --- Optional: Save case to Knowledge Graph ---
    saved_for_reference = False
    if should_save:
        logger.info(f"[OrchestratorAgent] Saving case to Knowledge Graph (user requested)")
        try:
            save_result = await upload_case_to_kg(pdf_bytes, filename)
            saved_for_reference = bool(save_result.get("case_id"))
            if saved_for_reference:
                logger.info(f"[OrchestratorAgent] ✓ Case saved to KG: {save_result.get('case_id', '')[:8]}...")
        except Exception as e:
            logger.error(f"[OrchestratorAgent] ✗ Case save failed (non-blocking): {e}")

    # --- Memory Agent: Save assistant response ---
    save_conversation(session_id, "assistant", final_summary[:2000])
    logger.info(f"[MemoryAgent] ✓ Conversation saved to short-term + long-term memory")

    # --- Build final response ---
    processing_time = round(time.time() - start_time, 2)

    logger.info(f"[OrchestratorAgent] ═══════════════════════════════════════════")
    logger.info(f"[OrchestratorAgent] Multi-Agent Pipeline COMPLETED in {processing_time}s")
    logger.info(f"[OrchestratorAgent]   Agents used: IntentDetection, CaseRetrieval, LawRetrieval,")
    logger.info(f"[OrchestratorAgent]                Summary, QuestionGen, Synthesis, Memory")
    logger.info(f"[OrchestratorAgent]   Cases: {cases_count} | Laws: {laws_count} | Questions: {q_count}")
    logger.info(f"[OrchestratorAgent]   Saved: {saved_for_reference} | Session: {session_id[:8]}...")
    logger.info(f"[OrchestratorAgent]   Framework: LangChain | LLM: Gemini 2.5 Flash")
    logger.info(f"[OrchestratorAgent] ═══════════════════════════════════════════")

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