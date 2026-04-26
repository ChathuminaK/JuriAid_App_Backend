"""
Entry point for running with: python main.py
For development use: uvicorn app:app --reload --port 8000
"""
import uvicorn
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

