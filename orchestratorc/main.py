"""
JuriAid Orchestrator - Entry Point
Run with: python main.py
"""

import os
import uvicorn
from config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

