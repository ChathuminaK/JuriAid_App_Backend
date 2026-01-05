# backend/classifier.py
from typing import List

ROLE_LABELS = ["FACT", "ISSUE", "ARGUMENT", "DECISION", "OTHER"]

# Expanded Keyword Lists
FACT_KEYWORDS = ["facts", "background", "occurred", "incident", "factual", "averred", "plaint", "alleged"]
ISSUE_KEYWORDS = ["issue", "whether", "question", "point for consideration", "determination"]
ARGUMENT_KEYWORDS = ["submitted", "argued", "counsel", "submission", "contend", "learned", "advocate", "petitioner"]
DECISION_KEYWORDS = ["held", "decided", "ordered", "dismissed", "upheld", "concluded", "allowed", "decreed", "judgment"]

def heuristic_role(sent: str) -> str:
    s = sent.lower().strip()
    if not s: return "OTHER"

    # 1. Identify Headings (Bold/Uppercase/Short)
    if len(s) < 100:
        if any(k in s for k in FACT_KEYWORDS): return "FACT"
        if any(k in s for k in ISSUE_KEYWORDS): return "ISSUE"
        if any(k in s for k in ARGUMENT_KEYWORDS): return "ARGUMENT"
        if any(k in s for k in DECISION_KEYWORDS): return "DECISION"

    # 2. Identify by Content
    if any(k in s for k in ISSUE_KEYWORDS): return "ISSUE"
    if any(k in s for k in DECISION_KEYWORDS): return "DECISION"
    if any(k in s for k in ARGUMENT_KEYWORDS): return "ARGUMENT"
    if any(k in s for k in FACT_KEYWORDS): return "FACT"
    
    # 3. Legal stop-words (Common in legal prose but not specific to a role)
    LEGAL_PROSE = ["court", "pursuant", "article", "section", "act", "case"]
    if any(k in s for k in LEGAL_PROSE):
        return "FACT" # Default legal prose to FACT instead of OTHER

    return "OTHER"

def predict_roles(sentences: List[str], model_path: str = None) -> List[str]:
    return [heuristic_role(s) for s in sentences]