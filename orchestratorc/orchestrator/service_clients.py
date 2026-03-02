"""
HTTP clients for downstream microservices.
"""

import httpx
import asyncio
import logging
from typing import Optional
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def _request_with_retry(
    method: str,
    url: str,
    retries: int = None,
    timeout: float = None,
    **kwargs,
) -> Optional[httpx.Response]:
    """
    Make HTTP request with retry logic and timeout.
    Returns response on success, None on failure (graceful degradation).
    """
    retries = retries if retries is not None else settings.MAX_RETRIES
    timeout = timeout or settings.SERVICE_TIMEOUT

    for attempt in range(1, retries + 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, **kwargs)

                if resp.status_code < 500:
                    return resp

                logger.warning(
                    f"Attempt {attempt}: {url} returned {resp.status_code}"
                )

        except httpx.TimeoutException:
            logger.warning(f"Attempt {attempt}: Timeout ({timeout}s) calling {url}")
        except httpx.RequestError as e:
            logger.warning(f"Attempt {attempt}: Connection error for {url}: {e}")

        if attempt <= retries:
            wait = settings.RETRY_DELAY * attempt
            logger.info(f"Retrying in {wait}s...")
            await asyncio.sleep(wait)

    logger.error(f"All {retries + 1} attempts failed for {url}")
    return None


# ---------- Past Case Retrieval (Port 8002) ----------

async def search_similar_cases(pdf_bytes: bytes, filename: str) -> dict:
    """
    POST file to :8002/search → returns similar past cases.
    Gracefully returns empty result on failure.
    """
    url = f"{settings.PAST_CASE_SERVICE_URL}/search"
    files = {"file": (filename, pdf_bytes, "application/pdf")}

    resp = await _request_with_retry("POST", url, files=files)

    if resp and resp.status_code == 200:
        data = resp.json()
        logger.info(f"Found {len(data.get('similar_cases', []))} similar cases")
        return data

    logger.warning("Past case search unavailable - continuing without similar cases")
    return {"similar_cases": [], "new_case_id": ""}


async def upload_case_to_kg(pdf_bytes: bytes, filename: str) -> dict:
    """
    POST file to :8002/admin/upload-case → saves case to Neo4j KG.
    Used when user intends to save case for future reference.
    """
    url = f"{settings.PAST_CASE_SERVICE_URL}/admin/upload-case"
    files = {"file": (filename, pdf_bytes, "application/pdf")}

    resp = await _request_with_retry("POST", url, files=files)

    if resp and resp.status_code == 200:
        data = resp.json()
        logger.info(f"Case saved to KG: {data.get('case_id', 'unknown')}")
        return data

    logger.warning("Failed to save case to KG")
    return {"message": "Save failed", "case_id": ""}


# ---------- LawStatKG (Port 8003) ----------

async def get_applicable_laws(pdf_bytes: bytes, filename: str) -> dict:
    """
    POST file to :8003/case/laws → returns applicable Sri Lankan laws.
    Gracefully returns empty result on failure.
    """
    url = f"{settings.LAWSTATKG_SERVICE_URL}/case/laws"
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data = {"as_of_date": "today"}

    resp = await _request_with_retry("POST", url, files=files, data=data)

    if resp and resp.status_code == 200:
        result = resp.json()
        logger.info(f"Found {len(result.get('applicable_laws', []))} applicable laws")
        return result

    logger.warning("LawStatKG unavailable - continuing without applicable laws")
    return {"applicable_laws": []}


# ---------- Question Generator (Port 8004) ----------

async def generate_questions(case_text: str, laws_text: str, cases_text: str) -> dict:
    """
    POST JSON to :8004/generate-questions → returns generated legal questions.
    Uses QUESTIONGEN_TIMEOUT (600s default) since Ollama CPU inference is slow.
    Gracefully returns empty result on failure.
    """
    url = f"{settings.QUESTIONGEN_SERVICE_URL}/generate-questions"
    json_body = {
        "case_text": case_text[:5000],
        "law": laws_text[:3000],
        "cases": cases_text[:3000],
    }

    # QuestionGen uses Ollama on CPU - needs much longer timeout
    questiongen_timeout = settings.QUESTIONGEN_TIMEOUT

    logger.info(f"Calling QuestionGen with timeout={questiongen_timeout}s (CPU inference)")

    resp = await _request_with_retry(
        "POST",
        url,
        json=json_body,
        timeout=questiongen_timeout,
        retries=1,  # Only 1 retry for QuestionGen (each attempt takes ~5 min)
    )

    if resp and resp.status_code == 200:
        data = resp.json()
        logger.info("Questions generated successfully")
        return data

    logger.warning("Question generation unavailable - continuing without questions")
    return {"questions": ""}