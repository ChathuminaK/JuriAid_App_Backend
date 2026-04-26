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
    retries = retries if retries is not None else settings.MAX_RETRIES
    timeout = timeout or settings.SERVICE_TIMEOUT

    for attempt in range(1, retries + 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, **kwargs)
                if resp.status_code < 500:
                    return resp
                logger.warning(f"Attempt {attempt}: {url} returned {resp.status_code}")
        except httpx.TimeoutException:
            logger.warning(f"Attempt {attempt}: Timeout calling {url}")
        except httpx.RequestError as e:
            logger.warning(f"Attempt {attempt}: Connection error for {url}: {e}")

        if attempt <= retries:
            wait = settings.RETRY_DELAY * attempt
            logger.info(f"Retrying in {wait}s...")
            await asyncio.sleep(wait)

    logger.error(f"All attempts failed for {url}")
    return None


# ---------- Past Case Retrieval (Port 8002) ----------

async def search_similar_cases(pdf_bytes: bytes, filename: str) -> dict:
    url = f"{settings.PAST_CASE_SERVICE_URL}/search"
    files = {"file": (filename, pdf_bytes, "application/pdf")}

    resp = await _request_with_retry("POST", url, files=files)

    if resp and resp.status_code == 200:
        raw = resp.json()
        logger.info(f"Past case search returned: {list(raw.keys())}")

        normalized_cases = []
        for c in raw.get("similar_cases", []):
            normalized_cases.append({
                "case_id": c.get("case_id", ""),
                "case_name": c.get("case_name", ""),
                "score": float(c.get("final_score", c.get("score", 0))),
                "judgment_preview": c.get("judgment_preview", ""),
                "reason": c.get("reason", ""),
                "shared_issues": c.get("shared_issues", []),
                "breakdown": c.get("breakdown", {}),
                "view_case_details": c.get("view_case_details", ""),
                "view_full_case_file": c.get("view_full_case_file", ""),
            })

        logger.info(f"Found {len(normalized_cases)} similar cases")
        return {
            "similar_cases": normalized_cases,
            "new_case_id": raw.get("new_case_id", ""),
        }

    logger.warning("Past case search unavailable - continuing without similar cases")
    return {"similar_cases": [], "new_case_id": ""}


async def get_case_judgment(case_id: str) -> dict:
    """Fetch full case details (judgment, complaint, defense) by case_id."""
    url = f"{settings.PAST_CASE_SERVICE_URL}/case/{case_id}"
    resp = await _request_with_retry("GET", url)
    if resp and resp.status_code == 200:
        return resp.json()
    logger.warning(f"Could not fetch case details for {case_id}")
    return {}


async def upload_case_to_kg(pdf_bytes: bytes, filename: str) -> dict:
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
    url = f"{settings.LAWSTATKG_SERVICE_URL}/case-law/retrieve"
    files = {"file": (filename, pdf_bytes, "application/pdf")}

    resp = await _request_with_retry("POST", url, files=files)

    if resp and resp.status_code == 200:
        raw_data = resp.json()
        logger.info(f"LawStatKG response keys: {list(raw_data.keys())}")

        case_laws = raw_data.get("relevant_case_laws", [])
        logger.info(f"Found {len(case_laws)} relevant case laws")

        return {
            "relevant_case_laws": case_laws,
            "queries_generated": raw_data.get("queries_generated", []),
            "results_count": raw_data.get("results_count", len(case_laws)),
        }

    logger.warning("LawStatKG unavailable - continuing without case laws")
    return {"relevant_case_laws": [], "queries_generated": [], "results_count": 0}


# ---------- Question Generator (Port 8004) ----------

async def generate_questions(case_text: str, laws_text: str, cases_text: str) -> dict:
    url = f"{settings.QUESTIONGEN_SERVICE_URL}/generate-questions"
    json_body = {
        "case_text": case_text[:5000],
        "law": laws_text[:3000],
        "cases": cases_text[:3000],
    }

    resp = await _request_with_retry(
        "POST", url, json=json_body, timeout=settings.QUESTIONGEN_TIMEOUT
    )

    if resp and resp.status_code == 200:
        data = resp.json()
        logger.info("Questions generated successfully")
        return data

    logger.warning("Question generation unavailable - continuing without questions")
    return {"questions": ""}