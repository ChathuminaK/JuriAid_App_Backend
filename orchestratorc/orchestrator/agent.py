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


def _get_llm():
    """Lazy-load Gemini LLM via LangChain. Returns None if unavailable."""
    global _llm
    if _llm is not None:
        return _llm

    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set - LLM features disabled")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=2048,
        )
        logger.info("Gemini LLM initialized successfully")
        return _llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}")
        return None


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

Return ONLY the JSON, no other text."""

        response = await llm.ainvoke(intent_prompt)
        content = response.content.strip()

        # Clean markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        parsed = json.loads(content)

        return {
            "should_save_case": bool(parsed.get("should_save_case", False)),
            "analysis_focus": str(parsed.get("analysis_focus", "general legal analysis")),
            "key_topics": list(parsed.get("key_topics", [])),
        }

    except Exception as e:
        logger.warning(f"Intent detection failed, using defaults: {e}")
        # Fallback keyword detection
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
    Use Gemini to generate a comprehensive case summary.
    Falls back to a basic summary if LLM is unavailable.
    """
    llm = _get_llm()
    if not llm:
        # Fallback: return truncated case text as summary
        return f"## Case Summary\n\n{case_text[:1500]}\n\n*Note: AI summary unavailable.*"

    try:
        # Build context from conversation history
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation context:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                history_context += f"- {msg['role']}: {msg['content'][:200]}\n"

        summary_prompt = f"""You are a senior Sri Lankan legal analyst. Analyze this legal case thoroughly.

User's request: {user_prompt}
{history_context}

CASE DOCUMENT:
{case_text[:4000]}

SIMILAR PAST CASES:
{similar_cases_text[:2000]}

APPLICABLE LAWS:
{laws_text[:2000]}

Provide a comprehensive legal analysis in this format:

## Case Overview
Brief overview of the case

## Key Facts
Numbered list of important facts

## Legal Issues
Identified legal issues and concerns

## Applicable Legal Framework
Relevant Sri Lankan laws and how they apply

## Analysis based on Similar Cases
How past cases inform this case

## Preliminary Assessment
Strategic assessment and recommendations

Be specific to Sri Lankan law. Be thorough but concise."""

        response = await llm.ainvoke(summary_prompt)
        summary = response.content.strip()

        if len(summary) < 50:
            logger.warning("LLM returned very short summary")
            return f"## Case Summary\n\n{case_text[:1500]}"

        return summary

    except Exception as e:
        logger.error(f"Case summary generation failed: {e}")
        return f"## Case Summary\n\n{case_text[:1500]}\n\n*Note: AI analysis encountered an error.*"


# ---------- Final Synthesis ----------

async def synthesize_analysis(
    case_summary: str,
    questions: str,
    user_prompt: str,
) -> str:
    """
    Final synthesis pass: refine the case summary with generated questions.
    If LLM unavailable, returns the case summary as-is.
    """
    llm = _get_llm()
    if not llm:
        return case_summary

    try:
        synthesis_prompt = f"""You are a senior Sri Lankan legal advisor doing a final review.

USER REQUEST: {user_prompt}

CASE ANALYSIS:
{case_summary[:3000]}

GENERATED QUESTIONS FOR THE CASE:
{questions[:2000]}

Refine the analysis by:
1. Ensuring all key legal points are addressed
2. Highlighting any gaps the questions reveal
3. Adding strategic recommendations

Output the final refined analysis. Keep the same structured format."""

        response = await llm.ainvoke(synthesis_prompt)
        return response.content.strip() or case_summary

    except Exception as e:
        logger.warning(f"Synthesis failed, using original summary: {e}")
        return case_summary