import re
from transformers import AutoTokenizer, AutoModel

# ✅ DEFINE MODEL NAME FIRST
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


def classify_sentence(sentence: str):

    sentence_lower = sentence.lower()

    if any(x in sentence_lower for x in [
        "appeal dismissed",
        "conviction upheld",
        "ordered that",
        "held that",
        "accordingly",
        "therefore"
    ]):
        return "decisions"

    if any(x in sentence_lower for x in [
        "issue is",
        "question is",
        "whether",
        "for determination"
    ]):
        return "issues"

    if any(x in sentence_lower for x in [
        "submitted",
        "argued",
        "contended",
        "counsel"
    ]):
        return "arguments"

    return "facts"


def classify_text(text: str):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    roles = {
        "facts": [],
        "issues": [],
        "arguments": [],
        "decisions": []
    }

    for sentence in sentences:
        if len(sentence.strip()) < 20:
            continue

        role = classify_sentence(sentence)
        roles[role].append(sentence)

    return roles