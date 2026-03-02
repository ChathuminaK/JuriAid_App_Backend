"""
JuriAid LangChain Agent - Gemini
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env")
    logger.info(f"🤖 Using Gemini | Key: {api_key[:8]}...")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=4096,
    )


# ── Agent 1: Case Summarizer ─────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Sri Lankan legal analyst.
Analyze the case and provide a structured summary:
1. Case Overview (parties, court, case number, type)
2. Key Facts (chronological)
3. Legal Issues
4. Claims/Relief Sought
5. Applicable Sri Lankan Laws
6. Key Observations for counsel"""),
    ("human", "User request: {user_prompt}\n\n--- CASE ---\n{case_text}\n--- END ---\n\nProvide analysis:"""),
])


async def summarize_case(case_text: str, user_prompt: str = "Analyze this case") -> str:
    try:
        llm = _get_llm()
        chain = SUMMARY_PROMPT | llm | StrOutputParser()
        result = await chain.ainvoke({
            "case_text": case_text[:15000],
            "user_prompt": user_prompt,
        })
        logger.info(f"✅ Summary done: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"❌ summarize_case failed: {type(e).__name__}: {e}")
        raise


# ── Agent 2: Synthesizer ─────────────────────────────────

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the JuriAid legal analysis coordinator.
Synthesize the specialist agent outputs into one coherent legal analysis report.
Focus on Sri Lankan divorce/family law. Connect past cases, laws, and current case facts."""),
    ("human", """--- CASE SUMMARY ---
{summary}

--- SIMILAR PAST CASES ---
{similar_cases}

--- APPLICABLE LAWS ---
{laws}

--- GENERATED QUESTIONS ---
{questions}

Provide coordinated analysis:"""),
])


async def synthesize_analysis(
    summary: str,
    similar_cases_text: str,
    laws_text: str,
    questions_text: str,
) -> str:
    try:
        llm = _get_llm()
        chain = SYNTHESIS_PROMPT | llm | StrOutputParser()
        result = await chain.ainvoke({
            "summary": summary,
            "similar_cases": similar_cases_text,
            "laws": laws_text,
            "questions": questions_text,
        })
        logger.info(f"✅ Synthesis done: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"❌ synthesize_analysis failed: {type(e).__name__}: {e}")
        raise