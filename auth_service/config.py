import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    HOST: str = os.getenv("AUTH_SERVICE_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("AUTH_SERVICE_PORT", "8001"))
    BASE_URL: str = os.getenv("AUTH_SERVICE_BASE_URL", "http://127.0.0.1:8001")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./users.db")
    
    # CORS Configuration
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    def __init__(self):
        if not self.JWT_SECRET_KEY:
            raise ValueError("JWT_SECRET_KEY must be set in .env file")

settings = Settings()