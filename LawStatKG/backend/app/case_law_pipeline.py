import re
from collections import Counter, defaultdict
from typing import Dict, Any, List

from app.case_law_engine import tokenize, clean_query

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
SECTION_PAT = re.compile(r"\bsection\s+(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
S_DOT_PAT = re.compile(r"\bs\.\s*(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)

STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "on", "for", "a", "an", "is", "are", "was", "were", "be",
    "plaintiff", "defendant", "respondent", "petitioner", "court", "honourable", "case", "number",
    # OCR boilerplate from page headers and CSV metadata lines
    "page", "nature", "regular", "value", "true", "copy", "procedure",
    # high-frequency legal filler with zero topical signal
    "state", "states", "stated", "submit", "submitted", "pray", "prays",
    "respectfully", "hereby", "said", "herein", "above", "named",
}

# CHANGE:
# Topic detection added to keep only legally relevant results
TOPIC_KEYWORDS = {
    "adultery": ["adultery", "affair", "co-respondent", "co respondent"],
    "malicious_desertion": ["desertion", "malicious desertion", "constructive desertion", "abandon"],
    "cruelty": ["cruelty", "violent", "violence", "abuse"],
    "nullity_of_marriage": ["nullity", "void marriage", "invalid marriage", "null and void"],
    "alimony_and_financial": ["alimony", "maintenance", "financial support", "money", "payment"],
    "consummation": ["consummation", "copulate", "sexual intercourse"],
    "customary_marriage_and_presumption": ["customary marriage", "presumption of marriage", "habit and repute"],
    "jurisdiction_and_procedure": ["jurisdiction", "district court", "procedure", "plaint", "answer"],
    "condonation_and_connivance": ["condonation", "connivance", "forgiveness", "reconciliation"],
    "muslim_law": ["muslim", "quazi", "islamic", "repudiating contract of marriage"],
    "decree_nisi": ["decree nisi", "nisi declaration", "nisi absolute", "make absolute", "decree absolute"],
}

# Strips procedural boilerplate noise from PDF text before keyword extraction
_NOISE_RE = re.compile(
    r"(\[?Page\s*\d+\]?"              # [Page 1] or Page 1 (after bracket-strip)
    r"|Case Number\s*:.*"
    r"|No\.\s*\d+[/,].*"
    r"|Attorney.at.Law.*"
    r"|T\.?\s*P\.?\s*\d[\d\-]+"
    r"|In the District Court.*"
    r"|\d{4}[./-]\d{2}[./-]\d{2}"
    r"|\bon this\b.*?\byear\b.*?\d{4}\.?)",
    re.IGNORECASE
)


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_noise(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def detect_topics(text: str) -> List[str]:
    text = text.lower()
    detected = []

    for topic, keys in TOPIC_KEYWORDS.items():
        for key in keys:
            if key in text:
                detected.append(topic)
                break

    return detected


def build_queries(case_text: str) -> List[str]:
    t = clean_query(case_text)
    clean_t = _strip_noise(t)          # strip procedural noise before keyword extraction
    keywords = extract_keywords(clean_t)
    sections = extract_sections(t)     # sections from original text
    detected_topics = detect_topics(t)

    queries = []

    if keywords:
        queries.append(" ".join(keywords[:10]))
        queries.append(" ".join(keywords[:6]))

    for s in sections[:6]:
        queries.append(f"section {s}")
        if keywords:
            queries.append(f"section {s} " + " ".join(keywords[:6]))

    # CHANGE:
    # add topic names as strong queries
    for topic in detected_topics:
        topic_phrase = topic.replace("_", " ")
        queries.append(topic_phrase)

    seen = set()
    out = []
    for q in queries:
        q = clean_query(q)
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    # CHANGE:
    # fewer queries -> less noise
    return out[:8]


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
    intersection = len(c & d)
    return intersection / (len(c | d) + 1e-6)   # Jaccard: fair regardless of doc size


def retrieve_case_law_from_case(engine, case_text: str, top_k: int = 5) -> Dict[str, Any]:
    case_text = normalize_text(case_text)
    queries = build_queries(case_text)
    detected_topics = detect_topics(case_text)

    all_hits = []
    for q in queries:
        res = engine.search(
            query=q,
            top_k=15,
            bm25_candidates=120,
            alpha=0.55,              # CHANGE: align with stricter search
            beta=0.45,
            min_match_ratio=0.50,
            min_semantic_cosine=0.35
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

        # CHANGE:
        # topic filtering to remove unrelated laws
        if detected_topics:
            doc_topic = (best["doc"].get("topic") or "").strip().lower()
            if doc_topic not in [t.lower() for t in detected_topics]:
                continue

        sup = support_score(case_text, best["doc"])

        # CHANGE:
        # stronger support-score weight
        final = max(v["scores"]) + 0.10 * (v["hits"] - 1) + 0.45 * sup

        best["final_score"] = float(final)
        best["support_score"] = float(sup)
        best["query_hits"] = int(v["hits"])
        merged.append(best)

    # CHANGE:
    # remove weak matches
    merged = [m for m in merged if m["final_score"] > 0.45]

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
        "detected_topics": detected_topics,   # CHANGE: useful for debugging and evaluation
        "results_count": len(out),
        "relevant_case_laws": out
    }