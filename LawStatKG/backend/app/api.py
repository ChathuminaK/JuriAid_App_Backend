from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date as dt_date

from app.hybrid_search import hybrid_strict_search, clean_query, today_str
from app.kg_client import KGClient

app = FastAPI(title="LawStatKG API", version="1.0.0")
kg = KGClient()



origins = [
    "http://localhost:19006",  # Expo web
    "http://127.0.0.1:19006",
    "exp://192.168.1.xxx:19000",  # Replace with your Expo LAN URL
    "*",  # Temporary: allow all origins for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    jurisdiction: Optional[str] = None
    as_of_date: Optional[str] = None
    top_k: int = 5



@app.post("/Lawsearch")
def lawsearch(req: SearchRequest):
    q = clean_query(req.query)
    as_of = req.as_of_date or today_str()
    return hybrid_strict_search(query=q, as_of_date=as_of, jurisdiction=req.jurisdiction, top_k=req.top_k)

@app.post("/Lawsearch/explain")
def lawsearch_explain(req: SearchRequest):
    q = clean_query(req.query)
    as_of = req.as_of_date or today_str()
    results = hybrid_strict_search(query=q, as_of_date=as_of, jurisdiction=req.jurisdiction, top_k=req.top_k)
    return {
        "query_clean": q,
        "as_of_date": as_of,
        "jurisdiction": req.jurisdiction or "ALL",
        "count": len(results),
        "results": results
    }

# ✅ Correct: statute uses ?date=
@app.get("/statute/{act_id}")
def statute(act_id: str, date: str = Query("today")):
    as_of = dt_date.today().isoformat() if date == "today" else date
    return kg.get_statute_as_of(act_id=act_id, as_of_date=as_of)

@app.get("/graph/{act_id}")
def graph(act_id: str, limit_sections: int = 50):
    return kg.get_act_graph(act_id=act_id, limit_sections=limit_sections)

@app.get("/timeline/{act_id}/{section_no}")
def timeline(act_id: str, section_no: str):
    return kg.get_section_timeline(act_id=act_id, section_no=section_no)