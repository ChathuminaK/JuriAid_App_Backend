import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("LEGALBERT_MODEL", "nlpaueb/legal-bert-base-uncased")

# Lazy load to avoid OOM on free tier
_model = None
_tokenizer = None

def get_model():
    """✅ Lazy load — only load when first request comes in, not at startup"""
    global _model, _tokenizer
    if _model is None:
        print("⏳ Loading LegalBERT model...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        _model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
        _model.eval()
        print("✅ LegalBERT loaded")
    return _model, _tokenizer

def get_embedding(text: str):
    _model, _tokenizer = get_model()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    # Use CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding

def classify_text(text: str, candidate_labels: list[str] = None):
    """Classify text by computing similarity to candidate labels."""
    _model, _tokenizer = get_model()
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