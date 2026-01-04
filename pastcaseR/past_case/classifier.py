# backend/classifier.py
from typing import List

ROLE_LABELS = ["FACT", "ISSUE", "ARGUMENT", "DECISION", "OTHER"]

FACT_KEYWORDS = ["facts", "background", "occurred", "incident", "factual"]
ISSUE_KEYWORDS = ["issue", "whether", "question", "issue for decision"]
ARGUMENT_KEYWORDS = ["submitted", "argued", "counsel", "submission", "contend", "argument"]
DECISION_KEYWORDS = ["held", "decided", "ordered", "dismissed", "upheld", "concluded", "allowed"]

def heuristic_role(sent: str) -> str:
    s = sent.lower()
    # Short headings often indicate role
    if len(sent) < 120 and (sent.strip().isupper() or ":" in sent[:40]):
        if any(k in s for k in FACT_KEYWORDS): return "FACT"
        if any(k in s for k in ISSUE_KEYWORDS): return "ISSUE"
        if any(k in s for k in ARGUMENT_KEYWORDS): return "ARGUMENT"
        if any(k in s for k in DECISION_KEYWORDS): return "DECISION"
    if any(k in s for k in FACT_KEYWORDS): return "FACT"
    if any(k in s for k in ISSUE_KEYWORDS): return "ISSUE"
    if any(k in s for k in ARGUMENT_KEYWORDS): return "ARGUMENT"
    if any(k in s for k in DECISION_KEYWORDS): return "DECISION"
    return "OTHER"

def predict_roles(sentences: List[str], model_path: str = None) -> List[str]:
    # If you later supply a fine-tuned model, implement HF inference here.
    return [heuristic_role(s) for s in sentences]
