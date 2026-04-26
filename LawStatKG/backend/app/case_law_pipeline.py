import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set

from app.case_law_engine import tokenize, clean_query

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
SECTION_PAT = re.compile(r"\bsection\s+(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
S_DOT_PAT = re.compile(r"\bs\.\s*(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)

STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "on", "for", "a", "an", "is", "are", "was", "were", "be",
    "plaintiff", "defendant", "respondent", "petitioner", "court", "honourable", "case", "number",
    "page", "nature", "regular", "value", "true", "copy", "procedure",
    "state", "states", "stated", "submit", "submitted", "pray", "prays",
    "respectfully", "hereby", "said", "herein", "above", "named",
}

TOPIC_KEYWORDS = {
    "adultery": [
        "adultery", "affair", "co-respondent", "co respondent", "illicit relationship"
    ],
    "malicious_desertion": [
        "malicious desertion", "constructive desertion", "desertion", "abandon",
        "abandoned", "left the matrimonial home", "left home", "animus deserendi"
    ],
    "cruelty": [
        "cruelty", "mental cruelty", "physical cruelty", "abuse", "assault",
        "violence", "violent", "harassment"
    ],
    "nullity_of_marriage": [
        "nullity", "void marriage", "invalid marriage", "fraud", "concealed pregnancy"
    ],
    "alimony_and_financial": [
        "alimony", "maintenance", "financial support", "money", "payment"
    ],
    "consummation": [
        "consummation", "non-consummation", "sexual intercourse", "impotence", "impotent"
    ],
    "condonation_and_connivance": [
        "condonation", "connivance", "forgiveness", "reconciliation"
    ],
    "decree_nisi": [
        "decree nisi", "nisi declaration", "decree absolute", "make absolute"
    ],
}

# stronger phrase-level legal fact patterns
FACT_PATTERNS = {
    "adultery": [
        "adultery", "co-respondent", "sexual relationship", "extramarital relationship", "affair"
    ],
    "desertion": [
        "desertion", "malicious desertion", "constructive desertion", "abandoned",
        "left the matrimonial home", "refused to return", "animus deserendi"
    ],
    "cruelty": [
        "cruelty", "mental cruelty", "physical cruelty", "abuse", "assault",
        "threat", "violence", "harassment"
    ],
    "maintenance": [
        "maintenance", "alimony", "financial support", "expenses", "money", "payment"
    ],
    "nullity": [
        "nullity", "fraud", "void marriage", "concealed pregnancy", "deception"
    ],
    "impotence": [
        "impotence", "impotent", "non consummation", "non-consummation", "sexual incapacity"
    ],
    "reconciliation": [
        "reconciliation", "forgiveness", "resumed cohabitation", "returned to live together"
    ],
}

_NOISE_RE = re.compile(
    r"(\[?Page\s*\d+\]?"
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


def extract_keywords(text: str, top_k: int = 25) -> List[str]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    toks = [w for w in toks if w not in STOPWORDS and len(w) >= 3]
    freq = Counter(toks)
    strong = [w for w, c in freq.most_common() if c >= 2]
    return strong[:top_k] if strong else [w for w, _ in freq.most_common(top_k)]


def extract_sections(text: str, limit: int = 8) -> List[str]:
    found = SECTION_PAT.findall(text) + S_DOT_PAT.findall(text)
    freq = Counter([x.strip() for x in found if x.strip()])
    return [s for s, _ in freq.most_common(limit)]


def detect_topics(text: str) -> List[str]:
    text_l = text.lower()
    detected = []
    for topic, keys in TOPIC_KEYWORDS.items():
        for key in keys:
            if key in text_l:
                detected.append(topic)
                break
    return detected


def extract_fact_terms(text: str) -> Set[str]:
    text_l = text.lower()
    found = set()
    for label, patterns in FACT_PATTERNS.items():
        for p in patterns:
            if p in text_l:
                found.add(label)
                break
    return found


def build_queries(case_text: str) -> List[str]:
    t = clean_query(case_text)
    clean_t = _strip_noise(t)
    keywords = extract_keywords(clean_t, top_k=20)
    sections = extract_sections(t)
    detected_topics = detect_topics(t)
    fact_terms = extract_fact_terms(t)

    queries = []

    if keywords:
        queries.append(" ".join(keywords[:10]))
        queries.append(" ".join(keywords[:6]))

    for topic in detected_topics:
        queries.append(topic.replace("_", " "))

    for fact in sorted(fact_terms):
        queries.append(fact)

    for s in sections[:4]:
        queries.append(f"section {s}")
        if keywords:
            queries.append(f"section {s} " + " ".join(keywords[:4]))

    if detected_topics and keywords:
        queries.append(f"{detected_topics[0].replace('_', ' ')} " + " ".join(keywords[:5]))

    seen = set()
    out = []
    for q in queries:
        q = clean_query(q)
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    return out[:10]


def _doc_blob(doc: Dict[str, Any]) -> str:
    return " ".join([
        doc.get("section_title") or "",
        doc.get("section_content") or "",
        doc.get("case_name") or "",
        doc.get("facts") or "",
        " ".join(doc.get("held") or []),
        " ".join(doc.get("principle") or []),
        doc.get("topic") or "",
        " ".join(doc.get("relevant_laws") or []),
        " ".join(doc.get("relevant_sections") or []),
    ]).strip()


def support_score(case_text: str, doc: Dict[str, Any]) -> float:
    blob = _doc_blob(doc)
    c = set(tokenize(case_text))
    d = set(tokenize(blob))
    if not c or not d:
        return 0.0
    return len(c & d) / (len(c | d) + 1e-6)


def issue_overlap_score(case_text: str, doc: Dict[str, Any]) -> float:
    case_topics = set(detect_topics(case_text))
    if not case_topics:
        return 0.0

    doc_topic = (doc.get("topic") or "").strip().lower()
    if not doc_topic:
        return 0.0

    return 1.0 if doc_topic in case_topics else 0.0


def fact_overlap_score(case_text: str, doc: Dict[str, Any]) -> float:
    case_facts = extract_fact_terms(case_text)
    if not case_facts:
        return 0.0

    blob = _doc_blob(doc).lower()
    matched = 0
    for fact_label, patterns in FACT_PATTERNS.items():
        if fact_label not in case_facts:
            continue
        if any(p in blob for p in patterns):
            matched += 1

    return matched / max(1, len(case_facts))


def phrase_match_score(case_text: str, doc: Dict[str, Any]) -> float:
    """
    Stronger exact phrase matching for decisive legal facts.
    """
    blob = _doc_blob(doc).lower()
    text_l = case_text.lower()

    all_phrases = []
    for _, patterns in FACT_PATTERNS.items():
        all_phrases.extend(patterns)

    case_phrases = [p for p in all_phrases if p in text_l]
    if not case_phrases:
        return 0.0

    matched = sum(1 for p in case_phrases if p in blob)
    return matched / max(1, len(case_phrases))


def citation_section_score(case_text: str, doc: Dict[str, Any]) -> float:
    """
    Reward explicit law/section overlap if the uploaded case mentions sections.
    """
    text_l = case_text.lower()
    doc_sections = [str(x).lower() for x in (doc.get("relevant_sections") or [])]
    if not doc_sections:
        return 0.0

    hits = 0
    for s in doc_sections:
        if s and s in text_l:
            hits += 1

    return hits / max(1, len(doc_sections))


def retrieve_case_law_from_case(engine, case_text: str, top_k: int = 5) -> Dict[str, Any]:
    case_text = normalize_text(case_text)
    queries = build_queries(case_text)
    detected_topics = detect_topics(case_text)

    all_hits = []
    for q in queries:
        res = engine.search(
            query=q,
            top_k=25,
            bm25_candidates=200,
            alpha=0.55,
            beta=0.45,
            min_match_ratio=0.30,
            min_semantic_cosine=0.30
        )
        all_hits.append(res)

    bucket = defaultdict(lambda: {"best": None, "scores": [], "hits": 0, "ranks": []})

    for hit_list in all_hits:
        for rank, r in enumerate(hit_list, start=1):
            d = r["doc"]
            key = d["case_id"]
            bucket[key]["hits"] += 1
            bucket[key]["scores"].append(r["score"])
            bucket[key]["ranks"].append(rank)

            if bucket[key]["best"] is None or r["score"] > bucket[key]["best"]["score"]:
                bucket[key]["best"] = r

    merged = []
    for _, v in bucket.items():
        best = v["best"]
        if not best:
            continue

        doc = best["doc"]
        doc_topic = (doc.get("topic") or "").strip().lower()

        # stronger topic penalty
        topic_penalty = 0.0
        if detected_topics and doc_topic and doc_topic not in [t.lower() for t in detected_topics]:
            topic_penalty = 0.25

        sup = support_score(case_text, doc)
        issue_score = issue_overlap_score(case_text, doc)
        fact_score = fact_overlap_score(case_text, doc)
        phrase_score = phrase_match_score(case_text, doc)
        section_score = citation_section_score(case_text, doc)

        max_score = max(v["scores"])
        avg_score = sum(v["scores"]) / len(v["scores"])
        hit_bonus = 0.06 * min(5, max(0, v["hits"] - 1))
        rank_bonus = 1.0 / min(v["ranks"])  # reciprocal-rank style

        final = (
            0.25 * max_score +
            0.15 * avg_score +
            0.15 * sup +
            0.15 * issue_score +
            0.15 * fact_score +
            0.10 * phrase_score +
            0.05 * section_score +
            0.05 * rank_bonus +
            hit_bonus -
            topic_penalty
        )

        best["final_score"] = float(final)
        best["support_score"] = float(sup)
        best["issue_score"] = float(issue_score)
        best["fact_score"] = float(fact_score)
        best["phrase_score"] = float(phrase_score)
        best["section_score"] = float(section_score)
        best["query_hits"] = int(v["hits"])
        merged.append(best)

    merged = [m for m in merged if m["final_score"] >= 0.30]

    merged.sort(
        key=lambda x: (
            x["final_score"],
            x["issue_score"],
            x["fact_score"],
            x["phrase_score"],
            x["support_score"],
            x["score"],
        ),
        reverse=True
    )

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
            "issue_score": round(r["issue_score"], 3),
            "fact_score": round(r["fact_score"], 3),
            "phrase_score": round(r["phrase_score"], 3),
            "section_score": round(r["section_score"], 3),
            "query_hits": r["query_hits"],
            "detail_url": f"/case-law/{d.get('case_id')}"
        })

    return {
        "queries_generated": queries,
        "detected_topics": detected_topics,
        "results_count": len(out),
        "relevant_case_laws": out
    }