"""
JuriAid Case Analysis Pipeline - v2.1 (Resilient)
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

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)


class CaseAnalysisPipeline:

    async def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        user_prompt: str = "Analyze this case",
        save_for_reference: bool = False,
        user_id: Optional[int] = None,
    ) -> CaseAnalysisResponse:

        analysis_id = str(uuid.uuid4())
        start = time.time()
        logger.info(f"🚀 Pipeline START | id={analysis_id} | file={filename}")

        # Step 1: Extract text
        logger.info("📄 Step 1: Extracting text from PDF...")
        case_text = extract_text_from_pdf(file_bytes)

        # Steps 2+3+4: Parallel
        logger.info("🔄 Steps 2-4: Parallel retrieval + summarization...")
        results = await asyncio.gather(
            self._search_past_cases(file_bytes, filename),
            self._search_laws(file_bytes, filename),
            self._summarize(case_text, user_prompt),
            return_exceptions=True,
        )

        cases_raw = self._safe_result(results[0], "PastCase")
        laws_raw = self._safe_result(results[1], "LawStatKG")
        summary = self._safe_result(results[2], "Summary")

        # Log what we got
        logger.info(f"📊 PastCase raw: {type(cases_raw).__name__} | keys={list(cases_raw.keys()) if isinstance(cases_raw, dict) else 'N/A'}")
        logger.info(f"📊 LawStatKG raw: {type(laws_raw).__name__} | keys={list(laws_raw.keys()) if isinstance(laws_raw, dict) else 'N/A'}")
        logger.info(f"📊 Summary: {'OK (' + str(len(summary)) + ' chars)' if isinstance(summary, str) and summary else 'FAILED'}")

        # Fallback summary
        if not summary or not isinstance(summary, str):
            logger.warning("⚠️ Gemini summary failed, using extracted text")
            summary = case_text[:3000]

        # Step 5: Generate questions
        logger.info("❓ Step 5: Generating legal questions...")
        laws_text = self._format_laws_text(laws_raw)
        cases_text = self._format_cases_text(cases_raw)

        qgen_input = summary if len(summary) > 500 else case_text[:5000]
        questions_raw = await self._generate_questions(qgen_input, laws_text, cases_text)

        # Step 6: Synthesis
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
            logger.warning(f"⚠️ Synthesis failed: {e}")
            enriched_summary = summary

        # Step 7: Optional save
        if save_for_reference:
            logger.info("💾 Step 7: Saving case...")
            await self._save_case(file_bytes, filename)

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

    # ── Microservice calls ────────────────────────────────

    async def _search_past_cases(self, file_bytes: bytes, filename: str) -> Any:
        try:
            return await past_case_client.search_similar(file_bytes, filename)
        except ServiceError as e:
            logger.warning(f"⚠️ PastCase failed: {e.detail}")
            return None

    async def _search_laws(self, file_bytes: bytes, filename: str) -> Any:
        try:
            return await lawstatkg_client.get_case_laws(file_bytes, filename)
        except ServiceError as e:
            logger.warning(f"⚠️ LawStatKG failed: {e.detail}")
            return None

    async def _summarize(self, case_text: str, user_prompt: str) -> str:
        try:
            return await summarize_case(case_text, user_prompt)
        except Exception as e:
            logger.warning(f"⚠️ Summary failed: {e}")
            return None

    async def _generate_questions(self, summary: str, laws: str, cases: str) -> Any:
        try:
            return await questiongen_client.generate(summary, laws, cases)
        except ServiceError as e:
            logger.warning(f"⚠️ QuestionGen failed: {e.detail}")
            return None

    async def _save_case(self, file_bytes: bytes, filename: str) -> None:
        try:
            await past_case_client.save_case(file_bytes, filename)
            logger.info("💾 Case saved to Knowledge Graph")
        except ServiceError as e:
            logger.warning(f"⚠️ Save failed: {e.detail}")

    def _safe_result(self, result: Any, name: str) -> Any:
        if isinstance(result, Exception):
            logger.warning(f"⚠️ {name} error: {result}")
            return None
        return result

    # ══════════════════════════════════════════════════════
    #  FORMAT for QuestionGen input
    # ══════════════════════════════════════════════════════

    def _format_laws_text(self, data: Any) -> str:
        """Format LawStatKG ACTUAL response → string for QuestionGen.

        ACTUAL LawStatKG response shape:
        {
            "personal_law": "General",
            "relevant_laws": [
                {
                    "act_id": "general_marriages_cap131",
                    "act_title": "An Ordinance to consolidate...",
                    "section_no": "34",
                    "section_title": "Solemnization of marriage by minister",
                    "confidence_score": 0.484,
                    "evidence_from_case": "...",
                    ...
                }
            ]
        }
        """
        if not data or not isinstance(data, dict):
            return "No relevant laws retrieved."

        items = (
            data.get("relevant_laws")
            or data.get("applicable_laws")
            or data.get("results")
            or data.get("laws")
            or data.get("sections")
            or []
        )

        if not items:
            return "No relevant laws retrieved."

        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue

            title = item.get("act_title") or item.get("title") or item.get("law_name") or item.get("act_id") or ""
            section = item.get("section_title") or item.get("section") or ""
            section_no = item.get("section_no") or ""
            content = item.get("evidence_from_case") or item.get("content") or item.get("description") or ""
            score = item.get("confidence_score") or item.get("relevance_score") or item.get("score") or ""

            parts = []
            if title:
                parts.append(title)
            if section_no:
                parts.append(f"Section {section_no}")
            if section:
                parts.append(f"({section})")
            if content:
                parts.append(f"- {content}")
            if score:
                parts.append(f"[confidence: {score}]")

            line = " ".join(parts)
            if line.strip():
                lines.append(f"- {line}")

        return "\n".join(lines) if lines else "No relevant laws retrieved."

    def _format_cases_text(self, data: Any) -> str:
        """Format PastCase ACTUAL response → string for QuestionGen.

        ACTUAL PastCase response shape:
        {
            "new_case_id": "...",
            "similar_cases": [
                {
                    "case_id": "...",
                    "case_name": "Page 1",   ← this is the bug from PastCase
                    "score": null,
                    "complaint": null,
                    "defense": null
                }
            ]
        }
        """
        if not data or not isinstance(data, dict):
            return "No similar past cases retrieved."

        items = (
            data.get("similar_cases")
            or data.get("results")
            or data.get("cases")
            or []
        )

        if not items:
            return "No similar past cases found."

        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue

            name = (
                item.get("case_name")
                or item.get("title")
                or item.get("name")
                or ""
            )
            # Fix "Page 1" or empty names
            if not name or name.strip().lower().startswith("page"):
                name = f"Case {item.get('case_id', 'Unknown')}"

            score = item.get("score") or item.get("similarity_score") or ""
            complaint = item.get("complaint") or item.get("complaint_summary") or ""
            defense = item.get("defense") or item.get("defense_summary") or ""

            parts = [f"Case: {name}"]
            if score:
                parts.append(f"Score: {score}")
            if complaint:
                parts.append(f"Complaint: {complaint}")
            if defense:
                parts.append(f"Defense: {defense}")

            lines.append(" | ".join(parts))

        return "\n".join(lines) if lines else "No similar cases found."

    def _format_questions_text(self, data: Any) -> str:
        if not data:
            return "No questions generated."
        if isinstance(data, dict):
            return data.get("questions") or data.get("result") or str(data)
        return str(data)

    # ══════════════════════════════════════════════════════
    #  PARSE to schema models
    # ══════════════════════════════════════════════════════

    def _parse_cases(self, data: Any) -> List[SimilarCase]:
        if not data or not isinstance(data, dict):
            return []

        items = data.get("similar_cases") or data.get("results") or data.get("cases") or []

        cases = []
        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("case_name") or item.get("title") or item.get("name") or ""
            if not name or name.strip().lower().startswith("page"):
                name = f"Case {item.get('case_id', 'Unknown')}"

            cases.append(SimilarCase(
                case_id=str(item.get("case_id", "")),
                case_name=name,
                score=item.get("score") or item.get("similarity_score"),
                complaint=item.get("complaint") or item.get("complaint_summary"),
                defense=item.get("defense") or item.get("defense_summary"),
            ))
        return cases

    def _parse_laws(self, data: Any) -> List[RelevantLaw]:
        """Parse LawStatKG ACTUAL response → List[RelevantLaw].

        ACTUAL keys: act_id, act_title, section_no, section_title,
                     confidence_score, evidence_from_case, jurisdiction
        """
        if not data or not isinstance(data, dict):
            return []

        items = (
            data.get("relevant_laws")
            or data.get("applicable_laws")
            or data.get("results")
            or data.get("laws")
            or data.get("sections")
            or []
        )

        laws = []
        for item in items:
            if not isinstance(item, dict):
                continue
            laws.append(RelevantLaw(
                act_id=item.get("act_id") or item.get("id"),
                title=item.get("act_title") or item.get("title") or item.get("law_name"),
                section=item.get("section_no") or item.get("section") or item.get("section_id"),
                relevance_score=item.get("confidence_score") or item.get("relevance_score") or item.get("score"),
                content=item.get("section_title") or item.get("content") or item.get("description"),
            ))
        return laws

    def _parse_questions(self, data: Any) -> List[GeneratedQuestion]:
        if not data:
            return []

        text = ""
        if isinstance(data, dict):
            text = data.get("questions") or data.get("result") or str(data)
        elif isinstance(data, str):
            text = data

        if not text:
            return []

        questions = []
        qid = 0

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(r"^(?:\d+[\.\)\:\-]|\-|\*)\s*(.+)", line)
            if match:
                q_text = match.group(1).strip()
                if len(q_text) > 10:
                    qid += 1
                    questions.append(GeneratedQuestion(
                        question_id=qid,
                        question=q_text,
                    ))

        return questions