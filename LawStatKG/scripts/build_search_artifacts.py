import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")

# Ensure app import works from project root
import sys
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.hybrid_search import HybridSearchEngine


if __name__ == "__main__":
    engine = HybridSearchEngine()
    print("Building artifacts (BM25 + embeddings) ...")
    engine.build_and_save_artifacts()
    print(f"Artifacts saved to: {engine.artifact_dir}")