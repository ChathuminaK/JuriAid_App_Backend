"""
HTTP clients for all downstream microservices.
Each client matches the ACTUAL API endpoints of the running services.
"""

import httpx
import logging
from typing import Dict, Any
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceError(Exception):
    """Raised when a microservice call fails."""

    def __init__(self, service: str, status_code: int, detail: str):
        self.service = service
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{service}] HTTP {status_code}: {detail}")


# ═══════════════════════════════════════════════════════════
#  Past Case Retrieval (Port 8002)
# ═══════════════════════════════════════════════════════════

class PastCaseClient:
    """
    Endpoints:
      POST /search            → file upload → similar cases
      POST /admin/upload-case → file upload → save to Neo4j KG
    """

    def __init__(self):
        self.base_url = settings.PAST_CASE_SERVICE_URL.rstrip("/")
        self.timeout = settings.SERVICE_TIMEOUT

    async def search_similar(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """POST /search - upload PDF, get similar past cases."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/search",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                data = resp.json()
                count = len(data.get("similar_cases", []))
                logger.info(f"✅ PastCase /search → {count} similar cases")
                return data
            except httpx.HTTPStatusError as e:
                raise ServiceError("PastCase", e.response.status_code, e.response.text)
            except httpx.RequestError as e:
                raise ServiceError("PastCase", 503, str(e))

    async def save_case(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """POST /admin/upload-case - save PDF to Neo4j KG for future reference."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/admin/upload-case",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ PastCase /admin/upload-case → saved")
                return data
            except httpx.HTTPStatusError as e:
                raise ServiceError("PastCase", e.response.status_code, e.response.text)
            except httpx.RequestError as e:
                raise ServiceError("PastCase", 503, str(e))


# ═══════════════════════════════════════════════════════════
#  LawStatKG (Port 8003)
# ═══════════════════════════════════════════════════════════

class LawStatKGClient:
    """
    Endpoint:
      POST /case/laws → file upload → applicable laws for this case
    """

    def __init__(self):
        self.base_url = settings.LAWSTATKG_SERVICE_URL.rstrip("/")
        self.timeout = settings.SERVICE_TIMEOUT

    async def get_case_laws(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """POST /case/laws - upload PDF, get applicable Sri Lankan laws."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/case/laws",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ LawStatKG /case/laws → laws retrieved")
                return data
            except httpx.HTTPStatusError as e:
                raise ServiceError("LawStatKG", e.response.status_code, e.response.text)
            except httpx.RequestError as e:
                raise ServiceError("LawStatKG", 503, str(e))


# ═══════════════════════════════════════════════════════════
#  QuestionGen (Port 8004)
# ═══════════════════════════════════════════════════════════

class QuestionGenClient:
    """
    Endpoint:
      POST /generate-questions → JSON body → generated questions
    """

    def __init__(self):
        self.base_url = settings.QUESTIONGEN_SERVICE_URL.rstrip("/")
        self.timeout = settings.SERVICE_TIMEOUT

    async def generate(
        self, case_text: str, laws: str, cases: str
    ) -> Dict[str, Any]:
        """POST /generate-questions - send summary+laws+cases, get questions."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/generate-questions",
                    json={"case_text": case_text, "law": laws, "cases": cases},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ QuestionGen /generate-questions → questions ready")
                return data
            except httpx.HTTPStatusError as e:
                raise ServiceError("QuestionGen", e.response.status_code, e.response.text)
            except httpx.RequestError as e:
                raise ServiceError("QuestionGen", 503, str(e))


# ─── Singleton instances ──────────────────────────────────
past_case_client = PastCaseClient()
lawstatkg_client = LawStatKGClient()
questiongen_client = QuestionGenClient()