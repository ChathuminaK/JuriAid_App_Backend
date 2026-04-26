import re
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


# --------------------------------
# Generate embedding using LegalBERT
# --------------------------------
def get_embedding(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding


# --------------------------------
# Reference sentences for classes
# --------------------------------
REFERENCE = {
    "facts": "background facts of the case and events between parties",
    "issues": "legal issue or legal question before the court",
    "arguments": "legal arguments presented by counsel or lawyers",
    "decisions": "final decision or judgment of the court"
}

REFERENCE_EMBEDDINGS = {
    k: get_embedding(v)
    for k, v in REFERENCE.items()
}


# --------------------------------
# Classify sentence
# --------------------------------
def classify_sentence(sentence):

    sentence_embedding = get_embedding(sentence)

    scores = {}

    for label, ref_embedding in REFERENCE_EMBEDDINGS.items():

        score = cosine_similarity(
            [sentence_embedding],
            [ref_embedding]
        )[0][0]

        scores[label] = score

    return max(scores, key=scores.get)


# --------------------------------
# Classify full text
# --------------------------------
def classify_text(text):

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