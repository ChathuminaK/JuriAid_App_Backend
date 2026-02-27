import re

LEGAL_KEYWORDS = [
    "plaintiff",
    "defendant",
    "court",
    "judgment",
    "appeal",
    "petitioner",
    "respondent",
    "section",
    "act",
    "law",
    "order",
    "evidence",
    "trial"
]


def is_legal_document(text: str, threshold=5):
   

    if len(text) < 1000:   # very small documents reject
        return False

    text_lower = text.lower()
    count = 0

    for word in LEGAL_KEYWORDS:
        if re.search(rf"\b{word}\b", text_lower):
            count += 1

    # must contain at least 5 legal keywords
    if count < threshold:
        return False

    # Optional: ensure it contains court structure words
    must_have = ["plaintiff", "defendant", "judgment"]
    if not any(word in text_lower for word in must_have):
        return False

    return True