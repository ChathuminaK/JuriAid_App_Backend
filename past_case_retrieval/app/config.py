import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

WEIGHTS = {
    "facts": 0.35,
    "issues": 0.25,
    "arguments": 0.20,
    "decisions": 0.20
}