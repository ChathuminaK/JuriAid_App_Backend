import os
import re
from typing import Optional


def law_agent(new_case: Optional[str] = None) -> str:
    """
    Retrieves relevant laws from laws.txt. If a new case is provided,
    returns only the sections most relevant to that case.

    Args:
        new_case: The new case text to match laws against.

    Returns:
        A clean, structured string of relevant legal provisions.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    laws_path = os.path.join(base_dir, "data", "laws.txt")

    try:
        with open(laws_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except FileNotFoundError:
        return "[ERROR] laws.txt not found."

    if not raw:
        return "[ERROR] laws.txt is empty."

    law_sections = _split_laws(raw)

    if not law_sections:
        return "[ERROR] No law sections parsed from file."

    if new_case:
        law_sections = _filter_relevant_laws(new_case, law_sections, top_n=5)

    return _format_laws(law_sections)


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _split_laws(raw: str) -> list[str]:
    """Split laws.txt into individual sections."""
    # Try splitting by section markers like "Section X", "Article X", or "---"
    if "---" in raw:
        sections = [s.strip() for s in raw.split("---") if s.strip()]
    elif re.search(r'\b(section|article|act|clause)\s+\d+', raw, re.IGNORECASE):
        # Split before each Section/Article heading
        sections = re.split(r'(?=\b(?:Section|Article|Clause)\s+\d+)', raw, flags=re.IGNORECASE)
        sections = [s.strip() for s in sections if s.strip()]
    else:
        sections = [s.strip() for s in re.split(r"\n{2,}", raw) if s.strip()]

    return sections


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful legal keywords from text."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "is",
        "was", "were", "to", "for", "with", "that", "this", "it", "by",
        "had", "has", "have", "be", "been", "from", "not", "are", "as",
        "he", "she", "they", "his", "her", "their", "its", "which", "who",
        "shall", "may", "such", "under", "upon", "any", "all", "each"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return {w for w in words if w not in stopwords}


def _score_section(new_case_keywords: set[str], section: str) -> int:
    """Score a law section by keyword overlap."""
    section_keywords = _extract_keywords(section)
    return len(new_case_keywords & section_keywords)


def _filter_relevant_laws(new_case: str, law_sections: list[str], top_n: int = 5) -> list[str]:
    """Return the top_n most relevant law sections for the given case."""
    new_keywords = _extract_keywords(new_case)
    scored = [(sec, _score_section(new_keywords, sec)) for sec in law_sections]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sec for sec, score in scored[:top_n] if score > 0] or law_sections[:top_n]


def _format_laws(law_sections: list[str]) -> str:
    """Format law sections for LLM consumption."""
    formatted = []
    for i, section in enumerate(law_sections, 1):
        formatted.append(f"[Law Section {i}]\n{section}")
    return "\n\n".join(formatted)