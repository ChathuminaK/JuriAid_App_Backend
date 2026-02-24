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


def is_legal_document(text: str, threshold=3):
    """
    Check if document contains enough legal terminology.
    """

    text_lower = text.lower()
    count = 0

    for word in LEGAL_KEYWORDS:
        if re.search(rf"\b{word}\b", text_lower):
            count += 1

    return count >= threshold