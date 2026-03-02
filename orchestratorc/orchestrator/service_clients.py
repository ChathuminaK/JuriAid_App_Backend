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

        # Normalize similar_cases from PastCase API format
        normalized_cases = []
        for c in raw.get("similar_cases", []):
            normalized_cases.append({
                "case_id": c.get("case_id", ""),
                "case_name": c.get("case_name", ""),
                "score": float(c.get("final_score", c.get("score", 0))),
                "summary": c.get("summary", ""),
                "reason": c.get("reason", ""),
                "complaint": c.get("complaint", ""),
                "defense": c.get("defense", ""),
                "breakdown": c.get("breakdown", {}),
            })

        logger.info(f"Found {len(normalized_cases)} similar cases")
        return {
            "similar_cases": normalized_cases,
            "new_case_id": raw.get("new_case_id", ""),
        }

    logger.warning("Past case search unavailable - continuing without similar cases")
    return {"similar_cases": [], "new_case_id": ""}


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
    url = f"{settings.LAWSTATKG_SERVICE_URL}/case/laws"
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    data = {"as_of_date": "today"}

    resp = await _request_with_retry("POST", url, files=files, data=data)

    if resp and resp.status_code == 200:
        raw_data = resp.json()
        logger.info(f"LawStatKG response keys: {list(raw_data.keys())}")

        # Normalize each law entry to match our schema
        normalized_laws = []
        for law in raw_data.get("relevant_laws", []):
            normalized_laws.append({
                "act_id": law.get("act_id", ""),
                "title": law.get("act_title", law.get("title", "")),
                "section": law.get("section_no", law.get("section", "")),
                "section_title": law.get("section_title", ""),
                "relevance_score": float(law.get("confidence_score", law.get("relevance_score", 0))),
                "content": law.get("evidence_from_case", law.get("content", "")),
                "jurisdiction": law.get("jurisdiction", ""),
            })

        personal_law = raw_data.get("personal_law", "Unknown")
        logger.info(f"Found {len(normalized_laws)} applicable laws (Personal law: {personal_law})")

        return {
            "applicable_laws": normalized_laws,
            "personal_law": personal_law,
            "results_count": raw_data.get("results_count", len(normalized_laws)),
        }

    logger.warning("LawStatKG unavailable - continuing without applicable laws")
    return {"applicable_laws": [], "personal_law": "Unknown", "results_count": 0}


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