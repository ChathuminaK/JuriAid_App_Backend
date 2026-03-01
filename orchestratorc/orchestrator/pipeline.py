"""
JuriAid Case Analysis Pipeline
================================
The central orchestration engine that coordinates all microservices and agents.

Main Flow (POST /api/analyze):
  Step 1: Extract text from PDF (local)
  Step 2: Send PDF → PastCase /search → similar cases      ┐
  Step 3: Send PDF → LawStatKG /case/laws → applicable laws ├─ Parallel
  Step 4: LangChain Agent summarize case text → summary     ┘
  Step 5: Send summary+laws+cases → QuestionGen → questions
  Step 6: Agent synthesize final analysis
  Step 7: (Optional) Save case via PastCase /admin/upload-case

Save Flow (POST /api/cases/save):
  Step 1: Send PDF → PastCase /search (auto-indexes in Neo4j)
"""

import asyncio
import uuid
import time
import re
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

from orchestrator.pdf_extractor import extract_text_from_pdf
from orchestrator.agent import summarize_case, synthesize_analysis
from orchestrator.service_clients import (
    past_case_client,
    lawstatkg_client,
    questiongen_client,
    ServiceError,
)
from orchestrator.schemas import (
    CaseAnalysisResponse,
    SimilarCase,
    RelevantLaw,
    GeneratedQuestion,
)

logger = logging.getLogger(__name__)


class CaseAnalysisPipeline:
    """
    Central orchestrator pipeline.
    Implements the Agentic AI Framework's task decomposition and delegation.
    """

    # ═══════════════════════════════════════════════════════
    #  Main Analysis Flow
    # ═══════════════════════════════════════════════════════

    async def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        user_prompt: str = "Analyze this case",
        save_for_reference: bool = False,
        user_id: Optional[int] = None,
    ) -> CaseAnalysisResponse:
        """Run the full case analysis pipeline."""

        analysis_id = str(uuid.uuid4())
        start = time.time()
        logger.info(f"🚀 Pipeline START | id={analysis_id} | file={filename}")

        # ── Step 1: Extract text from PDF ─────────────────
        logger.info("📄 Step 1: Extracting text from PDF...")
        case_text = extract_text_from_pdf(file_bytes)

        # ── Steps 2+3+4: Parallel execution ───────────────
        # These 3 tasks are independent - run them concurrently
        logger.info("🔄 Steps 2-4: Parallel retrieval + summarization...")

        results = await asyncio.gather(
            self._search_past_cases(file_bytes, filename),     # Step 2
            self._search_laws(file_bytes, filename),           # Step 3
            self._summarize(case_text, user_prompt),           # Step 4
            return_exceptions=True,
        )

        cases_raw = self._safe_result(results[0], "PastCase")
        laws_raw = self._safe_result(results[1], "LawStatKG")
        summary = self._safe_result(results[2], "Summary")

        # Fallback summary if Gemini failed
        if not summary or not isinstance(summary, str):
            summary = f"[Auto-extract] {case_text[:2000]}..."

        # ── Step 5: Generate questions ────────────────────
        logger.info("❓ Step 5: Generating legal questions...")
        laws_text = self._format_laws_text(laws_raw)
        cases_text = self._format_cases_text(cases_raw)

        questions_raw = await self._generate_questions(summary, laws_text, cases_text)

        # ── Step 6: Agent synthesis ───────────────────────
        logger.info("🧠 Step 6: Agent synthesizing final analysis...")
        questions_text = self._format_questions_text(questions_raw)

        try:
            enriched_summary = await synthesize_analysis(
                summary=summary,
                similar_cases_text=cases_text,
                laws_text=laws_text,
                questions_text=questions_text,
            )
        except Exception as e:
            logger.warning(f"⚠️ Synthesis failed, using base summary: {e}")
            enriched_summary = summary

        # ── Step 7: Optional save for future reference ────
        if save_for_reference:
            logger.info("💾 Step 7: Saving case for future reference...")
            await self._save_case(file_bytes, filename)

        # ── Build response ────────────────────────────────
        elapsed = round(time.time() - start, 2)
        logger.info(f"✅ Pipeline DONE in {elapsed}s | id={analysis_id}")

        return CaseAnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            case_summary=enriched_summary,
            similar_cases=self._parse_cases(cases_raw),
            relevant_laws=self._parse_laws(laws_raw),
            generated_questions=self._parse_questions(questions_raw),
            metadata={
                "filename": filename,
                "file_size_mb": round(len(file_bytes) / (1024 * 1024), 2),
                "text_length": len(case_text),
                "user_id": user_id,
                "user_prompt": user_prompt,
                "saved_for_reference": save_for_reference,
            },
            created_at=datetime.now(timezone.utc),
            processing_time_seconds=elapsed,
        )

    # ═══════════════════════════════════════════════════════
    #  Microservice Calls (with graceful error handling)
    # ═══════════════════════════════════════════════════════

    async def _search_past_cases(self, file_bytes: bytes, filename: str) -> Any:
        """Step 2: POST PDF → PastCase /search"""
        try:
            return await past_case_client.search_similar(file_bytes, filename)
        except ServiceError as e:
            logger.warning(f"⚠️ PastCase failed: {e.detail}")
            return None

    async def _search_laws(self, file_bytes: bytes, filename: str) -> Any:
        """Step 3: POST PDF → LawStatKG /case/laws"""
        try:
            return await lawstatkg_client.get_case_laws(file_bytes, filename)
        except ServiceError as e:
            logger.warning(f"⚠️ LawStatKG failed: {e.detail}")
            return None

    async def _summarize(self, case_text: str, user_prompt: str) -> str:
        """Step 4: LangChain Agent → Gemini summary"""
        try:
            return await summarize_case(case_text, user_prompt)
        except Exception as e:
            logger.warning(f"⚠️ Summary failed: {e}")
            return None

    async def _generate_questions(self, summary: str, laws: str, cases: str) -> Any:
        """Step 5: POST summary+laws+cases → QuestionGen /generate-questions"""
        try:
            return await questiongen_client.generate(summary, laws, cases)
        except ServiceError as e:
            logger.warning(f"⚠️ QuestionGen failed: {e.detail}")
            return None

    async def _save_case(self, file_bytes: bytes, filename: str) -> None:
        """Step 7: POST PDF → PastCase /admin/upload-case"""
        try:
            await past_case_client.save_case(file_bytes, filename)
            logger.info("💾 Case saved to Knowledge Graph")
        except ServiceError as e:
            logger.warning(f"⚠️ Save failed: {e.detail}")

    # ═══════════════════════════════════════════════════════
    #  Result Handling
    # ═══════════════════════════════════════════════════════

    def _safe_result(self, result: Any, name: str) -> Any:
        """Handle asyncio.gather results safely."""
        if isinstance(result, Exception):
            logger.warning(f"⚠️ {name} error: {result}")
            return None
        return result

    # ═══════════════════════════════════════════════════════
    #  Format data for QuestionGen input
    # ═══════════════════════════════════════════════════════

    def _format_laws_text(self, data: Any) -> str:
        """Format LawStatKG response → readable string for QuestionGen."""
        if not data:
            return "No relevant laws retrieved."

        items = []
        if isinstance(data, dict):
            items = (
                data.get("applicable_laws")
                or data.get("results")
                or data.get("laws")
                or []
            )
        elif isinstance(data, list):
            items = data

        lines = []
        for item in items:
            if isinstance(item, dict):
                title = item.get("title") or item.get("law_name") or item.get("act_id", "")
                section = item.get("section") or item.get("section_no", "")
                content = item.get("content") or item.get("description", "")
                line = f"{title} {section}: {content}".strip(" :")
                if line:
                    lines.append(line)
            elif isinstance(item, str):
                lines.append(item)

        return "\n".join(lines) if lines else str(data)[:3000]

    def _format_cases_text(self, data: Any) -> str:
        """Format PastCase response → readable string for QuestionGen."""
        if not data:
            return "No similar past cases retrieved."

        items = data.get("similar_cases", []) if isinstance(data, dict) else []

        lines = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("case_name") or item.get("case_id", "Unknown")
                score = item.get("score", 0)
                complaint = item.get("complaint", "")
                defense = item.get("defense", "")
                lines.append(f"{name} (score: {score}): Complaint: {complaint}. Defense: {defense}")

        return "\n".join(lines) if lines else "No similar cases found."

    def _format_questions_text(self, data: Any) -> str:
        """Format QuestionGen response → readable string."""
        if not data:
            return "No questions generated."
        if isinstance(data, dict):
            return data.get("questions", str(data))
        return str(data)

    # ═══════════════════════════════════════════════════════
    #  Parse responses → schema models
    # ═══════════════════════════════════════════════════════

    def _parse_cases(self, data: Any) -> List[SimilarCase]:
        """Parse PastCase response → List[SimilarCase]."""
        if not data or not isinstance(data, dict):
            return []

        cases = []
        for item in data.get("similar_cases", []):
            if isinstance(item, dict):
                cases.append(SimilarCase(
                    case_id=str(item.get("case_id", "")),
                    case_name=item.get("case_name"),
                    score=item.get("score"),
                    complaint=item.get("complaint"),
                    defense=item.get("defense"),
                ))
        return cases

    def _parse_laws(self, data: Any) -> List[RelevantLaw]:
        """Parse LawStatKG response → List[RelevantLaw]."""
        if not data:
            return []

        items = []
        if isinstance(data, dict):
            items = (
                data.get("applicable_laws")
                or data.get("results")
                or data.get("laws")
                or []
            )
        elif isinstance(data, list):
            items = data

        laws = []
        for item in items:
            if isinstance(item, dict):
                laws.append(RelevantLaw(
                    act_id=item.get("act_id"),
                    title=item.get("title") or item.get("law_name"),
                    section=item.get("section") or item.get("section_no"),
                    relevance_score=item.get("relevance_score") or item.get("score"),
                    content=item.get("content") or item.get("description"),
                ))
        return laws

    def _parse_questions(self, data: Any) -> List[GeneratedQuestion]:
        """Parse QuestionGen response → List[GeneratedQuestion]."""
        if not data:
            return []

        text = data.get("questions", "") if isinstance(data, dict) else str(data)
        if not text:
            return []

        questions = []
        qid = 0
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\d+[\.\)\:\-]\s*(.+)", line)
            if match:
                qid += 1
                questions.append(GeneratedQuestion(
                    question_id=qid,
                    question=match.group(1).strip(),
                ))
        return questions