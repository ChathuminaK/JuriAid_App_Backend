"""
HTTP clients for downstream microservices.
"""

import httpx
import logging
from typing import Dict, Any
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ServiceError(Exception):
    def __init__(self, service: str, status_code: int, detail: str):
        self.service = service
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{service}] HTTP {status_code}: {detail}")


# ── PastCase (Port 8002) ──────────────────────────────────

class PastCaseClient:
    def __init__(self):
        self.base_url = settings.PAST_CASE_SERVICE_URL.rstrip("/")

    async def search_similar(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/search",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ PastCase /search → {len(data.get('similar_cases', []))} cases")
                return data
        except httpx.HTTPStatusError as e:
            raise ServiceError("PastCase", e.response.status_code, e.response.text[:300])
        except Exception as e:
            raise ServiceError("PastCase", 503, str(e))

    async def save_case(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/admin/upload-case",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                logger.info("✅ PastCase /admin/upload-case → saved")
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise ServiceError("PastCase-Save", e.response.status_code, e.response.text[:300])
        except Exception as e:
            raise ServiceError("PastCase-Save", 503, str(e))


# ── LawStatKG (Port 8003) ─────────────────────────────────

class LawStatKGClient:
    def __init__(self):
        self.base_url = settings.LAWSTATKG_SERVICE_URL.rstrip("/")

    async def get_case_laws(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/case/laws",
                    files={"file": (filename, file_bytes, "application/pdf")},
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ LawStatKG /case/laws → laws retrieved")
                return data
        except httpx.HTTPStatusError as e:
            raise ServiceError("LawStatKG", e.response.status_code, e.response.text[:300])
        except Exception as e:
            raise ServiceError("LawStatKG", 503, str(e))


# ── QuestionGen (Port 8004) ───────────────────────────────

class QuestionGenClient:
    def __init__(self):
        self.base_url = settings.QUESTIONGEN_SERVICE_URL.rstrip("/")

    async def generate(self, case_text: str, laws: str, cases: str) -> Dict[str, Any]:
        try:
            # QuestionGen uses Ollama locally → needs longer timeout (Ollama ~1-2 min)
            async with httpx.AsyncClient(timeout=300.0) as client:
                logger.info(f"❓ QuestionGen → sending {len(case_text)} chars case text...")
                resp = await client.post(
                    f"{self.base_url}/generate-questions",
                    json={
                        "case_text": case_text[:5000],  # trim to avoid overload
                        "law": laws,
                        "cases": cases,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(f"✅ QuestionGen → questions received")
                return data
        except httpx.ReadTimeout:
            raise ServiceError("QuestionGen", 504, "Ollama timeout - model took >300s")
        except httpx.HTTPStatusError as e:
            raise ServiceError("QuestionGen", e.response.status_code, e.response.text[:300])
        except Exception as e:
            raise ServiceError("QuestionGen", 503, str(e))


# Singletons
past_case_client = PastCaseClient()
lawstatkg_client = LawStatKGClient()
questiongen_client = QuestionGenClient()