from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import re
from datetime import date as _date
import os

from app.hybrid_search import HybridSearchEngine, clean_query, today_str
from app.kg_client import KGClient
from app.case_api import router as case_router

app = FastAPI(title="LawStatKG API", version="1.0.0")

# ✅ Register the case upload routes
app.include_router(case_router)

engine = HybridSearchEngine()
kg: Optional[KGClient] = None

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def clean_param(x: str) -> str:
    return (x or "").replace("\n", " ").replace("\r", " ").strip()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    jurisdiction: Optional[str] = None
    as_of_date: Optional[str] = None
    bm25_candidates: int = 80
    alpha: float = 0.65
    beta: float = 0.35
    min_match_ratio: float = 0.5
    min_semantic_cosine: float = 0.20


@app.on_event("startup")
def startup():
    global kg
    kg = KGClient()
    allow_build = os.getenv("ALLOW_BUILD_ON_STARTUP", "false").lower() == "true"
    engine.load(allow_build=allow_build)


@app.on_event("shutdown")
def shutdown():
    global kg
    if kg:
        kg.close()


@app.get("/health")
def health():
    return {"status": "ok", "neo4j": kg.ping() if kg else False, "search_loaded": engine.ready}


@app.post("/Lawsearch")
def law_search(req: SearchRequest):
    q = clean_query(req.query)
    as_of = req.as_of_date or today_str()

    results = engine.search(
        query=q,
        as_of_date=as_of,
        jurisdiction=req.jurisdiction,
        bm25_candidates=req.bm25_candidates,
        alpha=req.alpha,
        beta=req.beta,
        min_match_ratio=req.min_match_ratio,
        min_semantic_cosine=req.min_semantic_cosine,
    )
    return results


@app.get("/statute/{act_id}")
def statute(act_id: str, date: str = Query("today")):
    if not kg:
        raise HTTPException(status_code=500, detail="KGClient not initialized")

    act_id = clean_param(act_id)
    date_param = clean_param(date).lower()

    if date_param == "today":
        as_of = _date.today().isoformat()
    else:
        if not DATE_RE.match(date_param):
            raise HTTPException(status_code=400, detail="Invalid date. Use 'today' or YYYY-MM-DD")
        as_of = date_param

    return kg.get_statute_as_of(act_id=act_id, as_of_date=as_of)


@app.get("/timeline/{act_id}/{section_no}")
def timeline(act_id: str, section_no: str):
    if not kg:
        raise HTTPException(status_code=500, detail="KGClient not initialized")

    act_id = clean_param(act_id)
    section_no = clean_param(section_no)

    return kg.get_section_timeline(act_id=act_id, section_no=section_no)


@app.get("/timeline/change/{after_version_id}")
def timeline_change(after_version_id: str):
    if not kg:
        raise HTTPException(status_code=500, detail="KGClient not initialized")

    after_version_id = clean_param(after_version_id)
    return kg.get_change_detail(after_version_id=after_version_id)


@app.get("/amendments")
def amendments(date: str = Query("today")):
    if not kg:
        raise HTTPException(status_code=500, detail="KGClient not initialized")

    date_param = clean_param(date).lower()

    if date_param == "today":
        as_of = _date.today().isoformat()
    else:
        if not DATE_RE.match(date_param):
            raise HTTPException(status_code=400, detail="Invalid date. Use 'today' or YYYY-MM-DD")
        as_of = date_param

    return kg.get_amendments_by_date(as_of_date=as_of)