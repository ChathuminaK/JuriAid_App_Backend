# backend/config.py
import os

# ---------- Project root ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Changed: removed one dirname()

# ---------- Neo4j ----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "MyStrongPass123")

# ---------- FAISS ----------
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(INDEX_DIR, "meta.json")

# ---------- Embeddings ----------
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LEGAL_BERT_MODEL = None

# ---------- Data folders ----------
DATA_FOLDER = os.path.join(BASE_DIR, "data", "past_cases")
TMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "tmp_uploads")
os.makedirs(TMP_UPLOAD_FOLDER, exist_ok=True)

# ---------- Role weights ----------
ROLE_WEIGHTS = {
    "FACT": 0.4,
    "ISSUE": 0.1,
    "ARGUMENT": 0.4,
    "DECISION": 0.1,
    "OTHER": 0.05
}

# ---------- Hybrid weights ----------
ALPHA_VECTOR = 0.8
BETA_KG = 0.2
