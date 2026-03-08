import re
from collections import Counter, defaultdict
from typing import Dict, Any, List

from app.case_law_engine import tokenize, clean_query

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
SECTION_PAT = re.compile(r"\bsection\s+(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
S_DOT_PAT = re.compile(r"\bs\.\s*(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)

STOPWORDS = {
    "the","and","or","to","of","in","on","for","a","an","is","are","was","were","be",
    "plaintiff","defendant","respondent","petitioner","court","honourable","case","number"
}


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_keywords(text: str, top_k: int = 30) -> List[str]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    toks = [w for w in toks if w not in STOPWORDS and len(w) >= 3]
    freq = Counter(toks)
    strong = [w for w, c in freq.most_common() if c >= 2]
    return strong[:top_k] if strong else [w for w, _ in freq.most_common(top_k)]


def extract_sections(text: str, limit: int = 10) -> List[str]:
    found = SECTION_PAT.findall(text) + S_DOT_PAT.findall(text)
    freq = Counter([x.strip() for x in found if x.strip()])
    return [s for s, _ in freq.most_common(limit)]


def build_queries(case_text: str) -> List[str]:
    t = clean_query(case_text)
    keywords = extract_keywords(t)
    sections = extract_sections(t)

    queries = []
    if keywords:
        queries.append(" ".join(keywords[:10]))
        queries.append(" ".join(keywords[:6]))

    for s in sections[:8]:
        queries.append(f"section {s}")
        if keywords:
            queries.append(f"section {s} " + " ".join(keywords[:6]))

    seen = set()
    out = []
    for q in queries:
        q = clean_query(q)
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:20]


def support_score(case_text: str, doc: Dict[str, Any]) -> float:
    blob = " ".join([
        doc.get("section_title") or "",
        doc.get("section_content") or "",
        doc.get("case_name") or "",
        doc.get("facts") or "",
        " ".join(doc.get("held") or []),
        " ".join(doc.get("principle") or []),
        doc.get("topic") or "",
    ])
    c = set(tokenize(case_text))
    d = set(tokenize(blob))
    if not c or not d:
        return 0.0
    return len(c & d) / (len(d) + 1e-6)


def retrieve_case_law_from_case(engine, case_text: str, top_k: int = 5) -> Dict[str, Any]:
    case_text = normalize_text(case_text)
    queries = build_queries(case_text)

    all_hits = []
    for q in queries:
        res = engine.search(
            query=q,
            top_k=15,
            bm25_candidates=120,
            alpha=0.65,
            beta=0.35,
            min_match_ratio=0.25,
            min_semantic_cosine=0.08
        )
        all_hits.append(res)

    bucket = defaultdict(lambda: {"best": None, "scores": [], "hits": 0})

    for hit_list in all_hits:
        for r in hit_list:
            d = r["doc"]
            key = d["case_id"]
            bucket[key]["hits"] += 1
            bucket[key]["scores"].append(r["score"])
            if bucket[key]["best"] is None or r["score"] > bucket[key]["best"]["score"]:
                bucket[key]["best"] = r

    merged = []
    for _, v in bucket.items():
        best = v["best"]
        if not best:
            continue
        sup = support_score(case_text, best["doc"])
        final = max(v["scores"]) + 0.10 * (v["hits"] - 1) + 0.25 * sup
        best["final_score"] = float(final)
        best["support_score"] = float(sup)
        best["query_hits"] = int(v["hits"])
        merged.append(best)

    merged.sort(key=lambda x: x["final_score"], reverse=True)

    out = []
    for r in merged[:top_k]:
        d = r["doc"]
        out.append({
            "case_id": d.get("case_id"),
            "case_name": d.get("case_name"),
            "citation": d.get("citation"),
            "topic": d.get("topic"),
            "section_number": d.get("section_number"),
            "section_title": d.get("section_title"),
            "principle": d.get("principle"),
            "held": d.get("held"),
            "facts": d.get("facts"),
            "relevant_laws": d.get("relevant_laws"),
            "relevant_sections": d.get("relevant_sections"),
            "court": d.get("court"),
            "amending_law": d.get("amending_law"),
            "confidence_score": round(r["final_score"], 3),
            "support_score": round(r["support_score"], 3),
            "query_hits": r["query_hits"],
            "detail_url": f"/case-law/{d.get('case_id')}"
        })

    return {
        "queries_generated": queries,
        "results_count": len(out),
        "relevant_case_laws": out
    }