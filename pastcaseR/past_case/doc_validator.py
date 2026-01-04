LEGAL_KEYWORDS = [
    "court", "judge", "petitioner", "respondent",
    "appeal", "judgment", "section", "act",
    "held", "case", "v.", "vs"
]

def is_legal_case(text, role_counts):
    text_lower = text.lower()
    keyword_hits = sum(1 for k in LEGAL_KEYWORDS if k in text_lower)

    total_roles = sum(role_counts.values())
    argument_ratio = role_counts.get("ARGUMENT", 0) / max(total_roles, 1)
    decision_ratio = role_counts.get("DECISION", 0) / max(total_roles, 1)

    # Heuristic decision rule (research acceptable baseline)
    if keyword_hits >= 3 and (argument_ratio + decision_ratio) >= 0.25:
        return True
    return False
