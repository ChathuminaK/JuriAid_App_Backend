import re

COMPLAINT_KEYWORDS = [
    "plaintiff",
    "petitioner",
    "appellant",
    "claims",
    "alleges",
    "filed this action",
    "submits that",
    "contends that"
]

DEFENSE_KEYWORDS = [
    "defendant",
    "respondent",
    "denies",
    "in answer",
    "in reply",
    "states that",
    "submits in reply"
]


def extract_complaint_defense(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)

    complaint_sentences = []
    defense_sentences = []

    for sentence in sentences:
        s_lower = sentence.lower()

        if any(keyword in s_lower for keyword in COMPLAINT_KEYWORDS):
            complaint_sentences.append(sentence.strip())

        elif any(keyword in s_lower for keyword in DEFENSE_KEYWORDS):
            defense_sentences.append(sentence.strip())

    complaint_text = "\n".join(complaint_sentences)
    defense_text = "\n".join(defense_sentences)

    return complaint_text, defense_text