import os
from dotenv import load_dotenv

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_ENV = os.path.join(os.path.dirname(PKG_DIR), ".env")
PKG_ENV = os.path.join(PKG_DIR, ".env")

# Load root .env first, then package .env (both optional)
for p in (ROOT_ENV, PKG_ENV):
    if os.path.exists(p):
        load_dotenv(p, override=False)

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val