import os
import re
from typing import Optional


def case_agent(new_case: Optional[str] = None) -> str:
    """
    Retrieves past cases and extracts the most relevant ones
    based on key legal terms found in the new case.

    Args:
        new_case: The new case summary/text to match against. If provided,
                  only the most relevant past cases are returned.

    Returns:
        A clean, structured string of relevant past cases.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    past_cases_path = os.path.join(base_dir, "data", "past_cases.txt")

    try:
        with open(past_cases_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except FileNotFoundError:
        return "[ERROR] past_cases.txt not found."

    if not raw:
        return "[ERROR] past_cases.txt is empty."

    # Split cases by delimiter (supports blank lines or "---" separators)
    case_blocks = _split_cases(raw)

    if not case_blocks:
        return "[ERROR] No cases parsed from file."

    # If a new case is given, filter for the most relevant ones
    if new_case:
        case_blocks = _filter_relevant_cases(new_case, case_blocks, top_n=3)

    return _format_cases(case_blocks)


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _split_cases(raw: str) -> list[str]:
    """Split raw text into individual case blocks."""
    # Try splitting by '---' divider first
    if "---" in raw:
        blocks = [b.strip() for b in raw.split("---") if b.strip()]
    else:
        # Fall back to splitting by double newlines
        blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]

    return blocks


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful legal keywords from text (ignore stopwords)."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "is",
        "was", "were", "to", "for", "with", "that", "this", "it", "by",
        "had", "has", "have", "be", "been", "from", "not", "are", "as",
        "he", "she", "they", "his", "her", "their", "its", "which", "who"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return {w for w in words if w not in stopwords}


def _score_case(new_case_keywords: set[str], case_block: str) -> int:
    """Score a past case block by keyword overlap with the new case."""
    case_keywords = _extract_keywords(case_block)
    return len(new_case_keywords & case_keywords)


def _filter_relevant_cases(new_case: str, case_blocks: list[str], top_n: int = 3) -> list[str]:
    """Return the top_n most relevant case blocks."""
    new_keywords = _extract_keywords(new_case)
    scored = [(block, _score_case(new_keywords, block)) for block in case_blocks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [block for block, score in scored[:top_n] if score > 0] or case_blocks[:top_n]


def _format_cases(case_blocks: list[str]) -> str:
    """Format case blocks for LLM consumption."""
    formatted = []
    for i, block in enumerate(case_blocks, 1):
        formatted.append(f"[Past Case {i}]\n{block}")
    return "\n\n".join(formatted)


