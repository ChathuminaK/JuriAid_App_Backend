import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

WEIGHTS = {
    "facts": 0.30,
    "issues": 0.30,
    "arguments": 0.25,
    "decisions": 0.15
}