import re

LEGAL_ISSUE_KEYWORDS = [
    "divorce",
    "malicious desertion",
    "child custody",
    "alimony",
    "breach of contract",
    "fraud",
    "negligence",
    "property dispute",
    "criminal liability",
    "constitutional violation"
]

def extract_legal_issues(text):

    text_lower = text.lower()

    detected = []

    for issue in LEGAL_ISSUE_KEYWORDS:
        if issue in text_lower:
            detected.append(issue)

    return list(set(detected))