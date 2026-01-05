# backend/config.py
import os

# ---------- Project root ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Neo4j ----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pastcase12")

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


# backend/config.py

LEGAL_KEYWORDS = [
    "court", "judge", "petitioner", "respondent", "plaintiff", "defendant",
    "appeal", "judgment", "section", "act", "held", "case", "v.", "vs", 
    "counsel", "advocate", "order", "citation", "jurisdiction", "affidavit",
    "bench", "petition", "writ", "suit"
]

def is_legal_case(text, role_counts):
    if not text:
        return False
        
    text_lower = text.lower()
    
    # 1. Keyword check (More flexible)
    # We check if at least 3 unique legal words exist anywhere in the doc
    found_keywords = [k for k in LEGAL_KEYWORDS if k in text_lower]
    keyword_hits = len(set(found_keywords)) # Use set to count unique words
    
    # 2. Structure check
    total_roles = sum(role_counts.values())
    if total_roles == 0:
        return False
        
    # How much of the document is NOT "OTHER"
    legal_structure_count = total_roles - role_counts.get("OTHER", 0)
    legal_structure_score = legal_structure_count / total_roles

    # Console Debugging (Watch your terminal)
    print(f"--- VALIDATION REPORT ---")
    print(f"Total Sentences: {total_roles}")
    print(f"Keyword Hits: {keyword_hits}")
    print(f"Structure Score: {legal_structure_score:.2f}")
    print(f"Keywords Found: {list(set(found_keywords))[:5]}")
    print(f"-------------------------")

    # PASSING CRITERIA:
    # Rule A: It has 4+ unique legal keywords (Very likely a legal doc)
    if keyword_hits >= 4:
        return True
    
    # Rule B: It has some keywords AND some legal structure
    if keyword_hits >= 2 and legal_structure_score >= 0.10:
        return True
        
    return False