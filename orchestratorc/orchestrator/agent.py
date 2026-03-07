"""
JuriAid LangChain Agent - Gemini
"""

import logging
import json
from typing import Optional
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------- LLM Initialization ----------

_llm = None
_llm_initialized = False  # Track whether we've tried initialization


def _get_llm():
    """Lazy-load Gemini LLM via LangChain. Returns None if unavailable."""
    global _llm, _llm_initialized

    if _llm_initialized:
        return _llm

    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set - LLM features disabled")
        _llm_initialized = True
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=4096,
        )
        # Quick test to verify model works
        logger.info("Gemini LLM (gemini-2.5-flash) initialized successfully")
        _llm_initialized = True
        return _llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}")
        _llm_initialized = True
        _llm = None
        return None


def reset_llm():
    """Reset LLM so it can be re-initialized (useful after config change)."""
    global _llm, _llm_initialized
    _llm = None
    _llm_initialized = False


# ---------- Intent Detection ----------

async def detect_user_intent(prompt: str) -> dict:
    """
    Use LLM to detect user intent from their prompt.
    Determines: should we save the case? What's the analysis focus?
    Returns dict with 'should_save_case', 'analysis_focus', 'key_topics'.
    """
    default_intent = {
        "should_save_case": False,
        "analysis_focus": "general legal analysis",
        "key_topics": [],
    }

    llm = _get_llm()
    if not llm:
        # Fallback: simple keyword detection
        lower_prompt = prompt.lower()
        save_keywords = ["save", "store", "keep", "reference", "future", "remember"]
        default_intent["should_save_case"] = any(kw in lower_prompt for kw in save_keywords)
        return default_intent

    try:
        intent_prompt = f"""You are a legal AI assistant intent classifier. Analyze the user's prompt and return ONLY valid JSON.

User prompt: "{prompt}"

Return JSON with exactly these fields:
{{
  "should_save_case": true/false (true if user wants to save/store/keep this case for future reference),
  "analysis_focus": "brief description of what user wants analyzed",
  "key_topics": ["topic1", "topic2"] (legal topics mentioned)
}}

Return ONLY the JSON object, no markdown, no code blocks, no other text."""

        response = await llm.ainvoke(intent_prompt)
        content = response.content.strip()

        # Clean markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json) and last line (```)
            cleaned_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    continue
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines).strip()

        parsed = json.loads(content)

        return {
            "should_save_case": bool(parsed.get("should_save_case", False)),
            "analysis_focus": str(parsed.get("analysis_focus", "general legal analysis")),
            "key_topics": list(parsed.get("key_topics", [])),
        }

    except Exception as e:
        logger.warning(f"Intent detection failed, using keyword fallback: {e}")
        lower_prompt = prompt.lower()
        save_keywords = ["save", "store", "keep", "reference", "future", "remember"]
        default_intent["should_save_case"] = any(kw in lower_prompt for kw in save_keywords)
        return default_intent


# ---------- Case Summary Generation ----------

async def generate_case_summary(
    case_text: str,
    user_prompt: str,
    similar_cases_text: str,
    laws_text: str,
    conversation_history: list[dict] = None,
) -> str:
    """
    Use Gemini to generate a comprehensive case summary and legal analysis.
    Falls back to a basic text summary if LLM is unavailable.
    """
    llm = _get_llm()
    if not llm:
        return _fallback_summary(case_text)

    try:
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation context:\n"
            for msg in conversation_history[-5:]:
                history_context += f"- {msg['role']}: {msg['content'][:200]}\n"

        summary_prompt = f"""You are a senior Sri Lankan legal analyst with 20 years of experience.
Analyze this legal case document thoroughly based on the user's request.

USER REQUEST: {user_prompt}
{history_context}

CASE DOCUMENT (extracted from PDF):
{case_text[:6000]}

SIMILAR PAST CASES FOUND:
{similar_cases_text[:2000]}

APPLICABLE SRI LANKAN LAWS:
{laws_text[:2000]}

Provide a comprehensive legal analysis in this structured format:

## Case Overview
Brief overview identifying parties, court, case number, and nature of dispute

## Key Facts
Numbered list of the most important facts from the case document

## Legal Issues Identified
What are the core legal questions this case raises?

## Applicable Legal Framework
Which Sri Lankan laws, ordinances, and sections apply and how?

## Analysis Based on Similar Cases
How do the similar past cases inform the likely outcome?

## Preliminary Assessment & Recommendations
Strategic assessment and practical recommendations for the lawyer

Be specific to Sri Lankan law. Reference actual sections and ordinances where possible.
Be thorough but concise. Do NOT include raw PDF text - analyze and summarize it."""

        response = await llm.ainvoke(summary_prompt)
        summary = response.content.strip()

        if len(summary) < 100:
            logger.warning("LLM returned very short summary, using fallback")
            return _fallback_summary(case_text)

        return summary

    except Exception as e:
        logger.error(f"Case summary generation failed: {e}")
        return _fallback_summary(case_text)


def _fallback_summary(case_text: str) -> str:
    """Generate a basic structured summary when LLM is unavailable."""
    # Try to extract key info from the text
    lines = case_text[:3000].split("\n")
    clean_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("Page")]

    preview = "\n".join(clean_lines[:30])

    return f"""## Case Overview
(AI-powered analysis unavailable - showing extracted case information)

## Extracted Case Content
{preview}

## Note
The AI reasoning engine could not process this case at this time.
The case text ({len(case_text)} characters) has been extracted successfully.
Similar cases and applicable laws were retrieved from the knowledge base.
Please retry or review the extracted content manually."""


# ---------- Final Synthesis ----------

async def synthesize_analysis(
    case_summary: str,
    questions: str,
    user_prompt: str,
) -> str:
    """
    Final synthesis: refine case summary incorporating generated questions.
    If LLM unavailable, returns the case summary as-is.
    """
    llm = _get_llm()
    if not llm:
        return case_summary

    # If no questions were generated, skip synthesis
    if not questions or len(questions.strip()) < 20:
        return case_summary

    try:
        synthesis_prompt = f"""You are a senior Sri Lankan legal advisor agent performing a final review.

USER REQUEST: {user_prompt}

CASE ANALYSIS (from previous step):
{case_summary[:4000]}

GENERATED INVESTIGATION QUESTIONS:
{questions[:2000]}

Perform a final refinement:
1. Keep the original structured analysis format
2. Integrate insights from the generated questions where relevant
3. Highlight any gaps the questions reveal
4. Add a final "## Recommended Next Steps" section

Output the complete refined analysis. Maintain the structured format with ## headings."""

        response = await llm.ainvoke(synthesis_prompt)
        result = response.content.strip()

        if len(result) < 100:
            return case_summary

        return result

    except Exception as e:
        logger.warning(f"Synthesis failed, returning original analysis: {e}")
        return case_summary