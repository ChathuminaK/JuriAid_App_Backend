"""
JuriAid LangChain Agent
========================
Gemini-powered legal reasoning agent using LangChain.
This is the "brain" - it summarizes cases and synthesizes final analysis.

Research Objective: Agentic AI Framework Orchestrator
- Uses LangChain for agent constructs
- Google Gemini as the LLM backbone
- Structured legal reasoning for Sri Lankan divorce law
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════
#  LLM Instance
# ═══════════════════════════════════════════════════════════

def _get_llm() -> ChatGoogleGenerativeAI:
    """Create Gemini LLM instance via LangChain."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        max_output_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════
#  Agent 1: Case Summarizer
# ═══════════════════════════════════════════════════════════

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Sri Lankan legal analyst specializing in family and divorce law.

Analyze the legal case document and provide a structured summary:

1. **Case Overview**: Parties, court, case number, case type
2. **Key Facts**: Main facts in chronological order
3. **Legal Issues**: What legal questions arise
4. **Claims/Relief Sought**: What the petitioner/plaintiff seeks
5. **Applicable Legal Framework**: Relevant Sri Lankan laws 
   (e.g., Matrimonial Causes Ordinance No.19, Marriage Registration Ordinance,
    Muslim Marriage and Divorce Act, Kandyan Marriage and Divorce Act, 
    Maintenance Act No.37, Women's Rights Act)
6. **Key Observations**: Important points a lawyer should note

Be professional, accurate, and focused on legally relevant details.
Focus specifically on Sri Lankan jurisprudence and court procedures."""),

    ("human", """User's request: {user_prompt}

--- CASE DOCUMENT ---
{case_text}
--- END ---

Provide your structured legal analysis:"""),
])


async def summarize_case(case_text: str, user_prompt: str = "Analyze this case") -> str:
    """
    Generate comprehensive case summary using LangChain + Gemini.
    This is the Agent's "summarization" capability.
    """
    llm = _get_llm()
    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    # Truncate very long documents to stay within token limits
    truncated = case_text[:15000]

    summary = await chain.ainvoke({
        "case_text": truncated,
        "user_prompt": user_prompt,
    })

    logger.info(f"🤖 Agent summary: {len(summary)} chars")
    return summary


# ═══════════════════════════════════════════════════════════
#  Agent 2: Response Synthesizer
# ═══════════════════════════════════════════════════════════

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the JuriAid legal analysis coordinator.

You have received analysis results from multiple specialist agents:
- A case summary from the case analysis agent
- Similar past cases from the case retrieval agent  
- Applicable laws from the law research agent
- Generated legal questions from the question generation agent

Your job is to synthesize all these into a single coherent legal analysis report.
Focus on Sri Lankan divorce and family law context.
Identify connections between the past cases, applicable laws, and the current case.
Highlight the most important findings for the practicing lawyer."""),

    ("human", """--- CASE SUMMARY ---
{summary}

--- SIMILAR PAST CASES ---
{similar_cases}

--- APPLICABLE LAWS ---
{laws}

--- GENERATED QUESTIONS ---
{questions}

Synthesize a brief coordinated analysis connecting these findings:"""),
])


async def synthesize_analysis(
    summary: str,
    similar_cases_text: str,
    laws_text: str,
    questions_text: str,
) -> str:
    """
    Final synthesis - combine all agent outputs into coherent analysis.
    This represents the "Response Synthesis" functional requirement.
    """
    llm = _get_llm()
    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()

    result = await chain.ainvoke({
        "summary": summary,
        "similar_cases": similar_cases_text,
        "laws": laws_text,
        "questions": questions_text,
    })

    logger.info(f"🤖 Agent synthesis: {len(result)} chars")
    return result