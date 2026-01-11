import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Orchestrator settings loaded from environment variables"""
    
    # Server Configuration
    HOST: str = os.getenv("ORCHESTRATOR_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("ORCHESTRATOR_PORT", "8000"))
    BASE_URL: str = os.getenv("ORCHESTRATOR_BASE_URL", "http://127.0.0.1:8000")
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    # Auth Service Integration
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://127.0.0.1:8001")
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "true").lower() == "true"
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_FILE_TYPES: list = os.getenv("ALLOWED_FILE_TYPES", ".pdf,.txt").split(",")
    
    # CORS Configuration
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    def __init__(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in .env file")

settings = Settings()