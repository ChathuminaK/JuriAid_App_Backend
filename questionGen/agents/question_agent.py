import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model="mistral",
    temperature=0.1,       # Deterministic output
    num_predict=768,        # Enough for 10-12 questions
)

# ── Prompts ────────────────────────────────────────────────────────────────────

_FINDINGS_PROMPT = PromptTemplate.from_template("""You are a Sri Lankan litigation attorney drafting FINDINGS OF FACT for court.

Based on the case, laws, and past precedents below, generate factual finding questions.

FINDINGS are questions about WHAT HAPPENED — disputed facts the court must determine.

FORMAT EXACTLY AS:
FINDING 1: [question]
FINDING 2: [question]
(up to 6 findings)

Rules:
- Each question starts with "Whether" or "Did" or "Was"
- One question per line, no sub-questions
- Focus on factual disputes only
- No legal conclusions, no explanations

---
CASE: {case}
APPLICABLE LAW: {law}
PRECEDENTS: {cases}
LEGAL ISSUES IDENTIFIED: {issues}
---

Generate findings now:""")


_ADMISSIONS_PROMPT = PromptTemplate.from_template("""You are a Sri Lankan litigation attorney preparing an ADMISSIONS checklist for court.

Based on the case and issues below, generate admission questions.

ADMISSIONS are facts both parties likely agree on — the undisputed foundation of the case.

FORMAT EXACTLY AS:
ADMISSION 1: [statement of fact]
ADMISSION 2: [statement of fact]
(up to 5 admissions)

Rules:
- Each admission is a declarative statement of agreed fact
- Start with "That the..." or "That both parties..."
- No disputed facts, no questions
- Concise and court-ready

---
CASE: {case}
LEGAL ISSUES: {issues}
---

Generate admissions now:""")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def question_agent(case: str, reasoning: str, law: str, cases: str) -> dict:
    """
    Generates structured FINDINGS and ADMISSIONS from the case inputs.

    Args:
        case:      The new case summary
        reasoning: Legal issues identified by reasoning_agent
        law:       Relevant laws from law_agent
        cases:     Relevant past cases from case_agent

    Returns:
        A dict with keys:
            - "findings":   list of finding questions
            - "admissions": list of admission statements
            - "formatted":  full formatted string for API response
    """
    case = case.strip()
    reasoning = reasoning.strip()
    law = law.strip()
    cases = cases.strip() if cases else "No past cases provided."

    findings_raw = _generate_findings(case, law, cases, reasoning)
    admissions_raw = _generate_admissions(case, reasoning)

    findings = _parse_labeled_list(findings_raw, label="FINDING")
    admissions = _parse_labeled_list(admissions_raw, label="ADMISSION")

    # Fallback: if structured parsing fails, extract plain questions
    if not findings:
        findings = _extract_plain_questions(findings_raw)
    if not admissions:
        admissions = _extract_plain_questions(admissions_raw)

    formatted = _format_output(findings, admissions)

    return {
        "findings": findings,
        "admissions": admissions,
        "formatted": formatted
    }


# ── Generation Helpers ────────────────────────────────────────────────────────

def _generate_findings(case: str, law: str, cases: str, issues: str) -> str:
    prompt = _FINDINGS_PROMPT.format(case=case, law=law, cases=cases, issues=issues)
    response = llm.invoke(prompt)
    return response.content.strip()


def _generate_admissions(case: str, issues: str) -> str:
    prompt = _ADMISSIONS_PROMPT.format(case=case, issues=issues)
    response = llm.invoke(prompt)
    return response.content.strip()


# ── Parsing Helpers ───────────────────────────────────────────────────────────

def _parse_labeled_list(raw: str, label: str) -> list[str]:
    """Parse 'LABEL N: text' formatted lines into a clean list."""
    results = []
    for line in raw.split("\n"):
        line = line.strip()
        match = re.match(rf"^{label}\s*\d+[\:\.]\s*(.+)", line, re.IGNORECASE)
        if match:
            item = match.group(1).strip()
            if len(item) > 8:
                results.append(item)
    return results


def _extract_plain_questions(raw: str) -> list[str]:
    """Fallback: extract any lines that look like questions or statements."""
    results = []
    for line in raw.split("\n"):
        line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line).strip()
        if len(line) > 10:
            results.append(line)
    return results[:8]


def _format_output(findings: list[str], admissions: list[str]) -> str:
    """Format findings and admissions into a clean court-ready output."""
    lines = []

    lines.append("=" * 60)
    lines.append("FINDINGS OF FACT")
    lines.append("=" * 60)
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("ADMISSIONS")
    lines.append("=" * 60)
    for i, a in enumerate(admissions, 1):
        lines.append(f"{i}. {a}")

    return "\n".join(lines)