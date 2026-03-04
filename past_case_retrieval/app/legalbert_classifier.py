import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("LEGALBERT_MODEL", "nlpaueb/legal-bert-base-uncased")

# Lazy load to avoid OOM on free tier
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _model is None:
        logger.info(f"Loading LegalBERT model: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        logger.info("LegalBERT model loaded successfully")

def get_embedding(text: str):
    _load_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    # Use CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding

def classify_text(text: str, candidate_labels: list[str] = None):
    """Classify text by computing similarity to candidate labels."""
    _load_model()
    text_emb = get_embedding(text)
    
    if candidate_labels is None:
        candidate_labels = ["criminal", "civil", "constitutional", "commercial", "family"]
    
    best_label = None
    best_score = -1.0
    
    for label in candidate_labels:
        label_emb = get_embedding(label)
        score = F.cosine_similarity(text_emb.unsqueeze(0), label_emb.unsqueeze(0)).item()
        if score > best_score:
            best_score = score
            best_label = label
    
    return {"label": best_label, "score": best_score}