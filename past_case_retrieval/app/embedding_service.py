from sentence_transformers import SentenceTransformer

_embed_model = None

def get_embed_model():
    """✅ Lazy load embedding model"""
    global _embed_model
    if _embed_model is None:
        print("⏳ Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
        print("✅ Embedding model loaded")
    return _embed_model

def generate_embedding(text: str):
    return get_embed_model().encode(text).tolist()