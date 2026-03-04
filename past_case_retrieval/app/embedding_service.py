from sentence_transformers import SentenceTransformer

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def generate_embedding(text: str):
    return get_embedding_model().encode(text).tolist()