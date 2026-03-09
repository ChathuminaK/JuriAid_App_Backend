import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()


class Settings:
    """Orchestrator configuration loaded from .env"""

    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Auth
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://127.0.0.1:8001")
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "true").lower() == "true"

    # Microservice URLs
    PAST_CASE_SERVICE_URL: str = os.getenv("PAST_CASE_SERVICE_URL", "http://127.0.0.1:8002")
    LAWSTATKG_SERVICE_URL: str = os.getenv("LAWSTATKG_SERVICE_URL", "http://127.0.0.1:8003")
    QUESTIONGEN_SERVICE_URL: str = os.getenv("QUESTIONGEN_SERVICE_URL", "http://127.0.0.1:8004")

    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Redis
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))  # 24 hours

    # Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

    # Timeouts
    SERVICE_TIMEOUT: float = float(os.getenv("SERVICE_TIMEOUT", "120"))
    QUESTIONGEN_TIMEOUT: float = float(os.getenv("QUESTIONGEN_TIMEOUT", "600"))

    # Retry
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))

    # Memory
    SHORT_TERM_WINDOW: int = 5
    LONG_TERM_MAX_MESSAGES: int = 50


@lru_cache()
def get_settings() -> Settings:
    return Settings()