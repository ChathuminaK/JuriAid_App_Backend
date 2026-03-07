
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))


_REASONING_PROMPT = PromptTemplate(
    input_variables=["case", "law", "cases"],
    template="""You are a senior Sri Lankan legal counsel preparing for court.

Analyze the case below and identify the core LEGAL ISSUES that must be proven or disproven.

FORMAT YOUR OUTPUT EXACTLY AS:
ISSUE 1: [one concise legal issue]
ISSUE 2: [one concise legal issue]
ISSUE 3: [one concise legal issue]
(continue up to 6 issues maximum)

Rules:
- Each issue must be a single, complete legal question
- Use formal legal language
- Focus on disputed facts and applicable law
- No explanations, no headings, no preamble

---
CURRENT CASE:
{case}

APPLICABLE LAW:
{law}

RELEVANT PRECEDENTS:
{cases}
---

List legal issues now:"""
)


def generate_questions(case_text: str, law_text: str, past_cases: str) -> str:
    """
    Identifies core legal issues from the case, applicable laws, and precedents.

    Returns a structured list of legal issues (findings/admissions).
    """
    if not case_text or not law_text:
        return "[ERROR] Case text and law text are required."

    final_prompt = _REASONING_PROMPT.format(
        case=case_text.strip(),
        law=law_text.strip(),
        cases=past_cases.strip() if past_cases else "No past cases provided."
    )

    response = llm.invoke(final_prompt)
    raw = response.content.strip()

    return _clean_issues(raw)


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _clean_issues(raw: str) -> str:
    """Normalize and clean the LLM output into structured issues."""
    import re
    lines = raw.split("\n")
    issues = []

    for line in lines:
        line = line.strip()
        # Match lines starting with ISSUE N: or numbered variants
        if re.match(r"^(ISSUE\s*\d+[\:\.]|^\d+[\.\)])", line, re.IGNORECASE):
            # Normalize to consistent format
            cleaned = re.sub(r"^(ISSUE\s*\d+[\:\.]|\d+[\.\)])\s*", "", line, flags=re.IGNORECASE).strip()
            if len(cleaned) > 10:
                issues.append(cleaned)

    if not issues:
        # Fallback: return raw output if parsing fails
        return raw

    return "\n".join(f"ISSUE {i}: {issue}" for i, issue in enumerate(issues, 1))