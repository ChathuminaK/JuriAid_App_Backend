import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple

from app.hybrid_search import clean_query, today_str, tokenize

# -----------------------------
# Stopwords + legal boilerplate
# -----------------------------
STOPWORDS = {
    "the","and","or","of","to","in","for","on","at","by","with","from","as","is","are","was","were",
    "be","been","being","that","this","these","those","it","its","their","there","here","than","then",
    "such","any","all","may","shall","can","could","would","should","not","no","yes","into","within",
    "between","over","under","above","below","also","each","every","either","neither","more","most",
    "less","least","about","after","before","during","while","where","when","who","whom","which",

    # pleading boilerplate
    "plaintiff","defendant","respondent","petitioner","applicant","court","honourable","honorable",
    "states","state","says","say","submits","submit","prays","prayer","relief","case","number","dated",
    "district","colombo","day","month","year","copy","certified","true","facts","fact","matter",
    "above","below","hereby","thereof","herein","whereof","whereby","aforesaid",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
SECTION_PAT = re.compile(r"\bsection\s+(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
S_DOT_PAT = re.compile(r"\bs\.\s*(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)

GENERIC_TITLE_HINTS = (
    "interpretation", "definitions", "short title", "commencement",
    "regulations", "fees", "gazette", "forms", "general", "penalty"
)

MIN_CASE_CHARS = 600


# -----------------------------
# Personal law detection (evidence-based)
# -----------------------------
PERSONAL_LAW_PATTERNS = {
    "Muslim": [
        r"\bquazi\b", r"\bnikah\b", r"\bmahr\b", r"\bwali\b", r"\btalaq\b",
        r"\bmuslim\b", r"muslim marriage", r"muslim marriage and divorce",
    ],
    "Kandyan": [
        r"\bkandyan\b", r"kandyan marriage", r"kandyan marriage and divorce",
        r"\bbinna\b", r"\bdiga\b",
    ],
    "Jaffna": [  # Tesawalamai family
        r"\btesawalamai\b", r"\bjaffna\b", r"matrimonial rights.*tesawalamai",
    ],
}

# map personal law to what acts are "allowed"
# (no hardcoding specific act_ids; we filter by Act.jurisdiction field in KG / docs)
ALLOWED_JURISDICTIONS = {
    "General": {"General", "Sri Lanka", None, ""},
    "Muslim": {"Muslim"},
    "Kandyan": {"Kandyan"},
    "Jaffna": {"Jaffna", "Tesawalamai"},
}

# If evidence is extremely strong for a personal law, allow cross-check even if default is General
STRONG_EVIDENCE_THRESHOLD = 2


def normalize_case_text(text: str) -> str:
    t = (text or "").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # remove simple page markers
    t = re.sub(r"\n\s*page\s*\d+\s*\n", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"\n\s*\d+\s*\n", "\n", t)
    return t.strip()


def is_generic_section_title(title: str) -> bool:
    s = (title or "").strip().lower()
    return any(h in s for h in GENERIC_TITLE_HINTS)


def extract_section_mentions(text: str, limit: int = 12) -> List[str]:
    found = SECTION_PAT.findall(text) + S_DOT_PAT.findall(text)
    freq = Counter([x.strip() for x in found if x.strip()])
    return [s for s, _ in freq.most_common(limit)]


def extract_keywords(text: str, top_k: int = 40) -> List[str]:
    toks = [w.lower() for w in WORD_RE.findall(text)]
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    freq = Counter(toks)

    # prefer repeated terms first
    repeated = [w for w, c in freq.most_common() if c >= 2]
    if repeated:
        return repeated[:top_k]

    return [w for w, _ in freq.most_common(top_k)]


def extract_ngrams(tokens: List[str], n: int, top_k: int) -> List[str]:
    grams = []
    for i in range(len(tokens) - n + 1):
        g = " ".join(tokens[i:i+n])
        grams.append(g)
    freq = Counter(grams)
    # keep grams that repeat (signal)
    return [g for g, c in freq.most_common(top_k) if c >= 2]


def context_windows(text: str, keyword: str, window: int = 260, max_windows: int = 5) -> List[str]:
    out = []
    t = text
    k = keyword.lower()
    idx = 0
    while True:
        pos = t.lower().find(k, idx)
        if pos == -1:
            break
        start = max(0, pos - window)
        end = min(len(t), pos + len(k) + window)
        out.append(t[start:end].strip())
        idx = pos + len(k)
        if len(out) >= max_windows:
            break
    return out


def best_evidence_sentence(case_text: str, law_text: str) -> str:
    sents = [s.strip() for s in SENT_SPLIT.split(case_text) if s.strip()]
    if not sents:
        return ""
    law_tokens = set([t for t in tokenize(law_text or "") if t not in STOPWORDS])
    if not law_tokens:
        return sents[0][:450]

    best = ("", 0)
    for s in sents:
        stoks = set([t for t in tokenize(s) if t not in STOPWORDS])
        overlap = len(stoks & law_tokens)
        if overlap > best[1]:
            best = (s, overlap)
    return best[0][:500]


def detect_personal_law(case_text: str) -> Dict[str, Any]:
    t = case_text.lower()
    hits = {}
    evidence = {}
    for label, patterns in PERSONAL_LAW_PATTERNS.items():
        ev = []
        count = 0
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE):
                count += 1
                ev.append(p)
        hits[label] = count
        evidence[label] = ev[:6]

    # choose best
    best_label = "General"
    best_score = 0
    for label, c in hits.items():
        if c > best_score:
            best_score = c
            best_label = label

    # If weak evidence, default General
    if best_score == 0:
        best_label = "General"

    return {
        "personal_law": best_label,
        "personal_law_debug": {
            "hits": hits,
            "evidence": evidence,
            "rule": "pattern_best" if best_score > 0 else "default_general"
        }
    }


# -----------------------------
# KG helpers: disambiguate section numbers
# -----------------------------
def kg_acts_with_section(kg_client, section_no: str) -> List[Dict[str, str]]:
    cypher = """
    MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
    WHERE trim(s.section_no) = trim($section_no)
    RETURN a.act_id AS act_id, a.title AS act_title, a.jurisdiction AS jurisdiction
    """
    with kg_client.driver.session() as session:
        rows = session.run(cypher, section_no=section_no)
        return [dict(r) for r in rows]


# -----------------------------
# Query generator (MORE + BETTER)
# -----------------------------
def build_queries(engine, kg_client, case_text: str) -> List[str]:
    t = clean_query(case_text)
    keywords = extract_keywords(t, top_k=40)
    sections = extract_section_mentions(t, limit=12)

    # token list for n-grams
    token_list = [w.lower() for w in WORD_RE.findall(t)]
    token_list = [x for x in token_list if x not in STOPWORDS and len(x) >= 3]

    bigrams = extract_ngrams(token_list, n=2, top_k=25)
    trigrams = extract_ngrams(token_list, n=3, top_k=18)

    queries: List[str] = []

    # 1) Strong keyword bundles
    if keywords:
        queries.append(" ".join(keywords[:12]))
        queries.append(" ".join(keywords[:8]))
        queries.append(" ".join(keywords[:6]))

    # 2) N-gram bundles (captures phrases like "malicious desertion", "decree nisi")
    for g in trigrams[:8]:
        queries.append(g)
    for g in bigrams[:12]:
        queries.append(g)

    # 3) Section mentions + KG act-title anchoring + context windows
    for s in sections[:8]:
        queries.append(f"section {s}")

        # Act-title anchored queries (data-driven via KG)
        if kg_client:
            acts = kg_acts_with_section(kg_client, s)
            for a in acts[:6]:
                title = (a.get("act_title") or "").strip()
                if title:
                    queries.append(f"{title} section {s}")

        # Context-based queries near the section mention
        for w in context_windows(t, f"section {s}", window=220, max_windows=3):
            kw = extract_keywords(w, top_k=18)
            if kw:
                queries.append(f"section {s} " + " ".join(kw[:8]))

    # 4) Case metadata-ish cues (court, remedy words if present in text)
    # (not hardcoded to divorce; these are generic legal signals)
    meta_terms = []
    for w in ["divorce", "custody", "maintenance", "alimony", "adultery", "desertion",
              "interim", "injunction", "appeal", "execution", "plaint", "answer", "family court"]:
        if w in t.lower():
            meta_terms.append(w)
    if meta_terms:
        queries.append(" ".join(meta_terms[:8]))
        if keywords:
            queries.append(" ".join(meta_terms[:5] + keywords[:5]))

    # Deduplicate preserving order
    out, seen = [], set()
    for q in queries:
        q = clean_query(q)
        if q and q not in seen and len(q.split()) >= 1:
            seen.add(q)
            out.append(q)

    return out[:35]  # generate MORE queries


# -----------------------------
# Scoring + filtering
# -----------------------------
def support_score(case_tokens: set, law_title: str, law_text: str) -> float:
    combined = (law_title or "") + " " + (law_text or "")
    law_tokens = set([t for t in tokenize(combined) if t not in STOPWORDS])
    if not law_tokens:
        return 0.0

    overlap = len(case_tokens & law_tokens)
    if overlap == 0:
        return 0.0

    # normalize: punish long generic sections
    ratio = overlap / max(60, len(law_tokens))

    title_tokens = set([t for t in tokenize(law_title or "") if t not in STOPWORDS])
    title_overlap = len(case_tokens & title_tokens)

    return ratio + min(0.25, title_overlap * 0.03)


def jurisdiction_allowed(doc_jur: Optional[str], personal_law: str) -> bool:
    allowed = ALLOWED_JURISDICTIONS.get(personal_law, {"General", "Sri Lanka", None, ""})
    return (doc_jur in allowed)


def retrieve_case_applicable_laws(
    engine,
    kg_client,
    case_text: str,
    as_of_date: Optional[str] = "today",
    top_k: int = 12,
) -> Dict[str, Any]:

    # date
    if not as_of_date or str(as_of_date).strip().lower() == "today":
        as_of = today_str()
    else:
        as_of = str(as_of_date).strip()

    case_text = normalize_case_text(case_text)
    if len(case_text) < MIN_CASE_CHARS:
        return {
            "as_of_date": as_of,
            "queries_generated": [],
            "results_count": 0,
            "relevant_laws": [],
            "warning": "Extracted case text is too short. PDF may be scanned/image-only or extraction failed.",
        }

    # personal law detection
    pl = detect_personal_law(case_text)
    personal_law = pl["personal_law"]
    strong_hits = max(pl["personal_law_debug"]["hits"].values()) if pl["personal_law_debug"]["hits"] else 0

    # case tokens (stopword-free)
    case_tokens = set([t for t in tokenize(case_text) if t not in STOPWORDS])

    # generate many queries
    queries = build_queries(engine, kg_client, case_text)

    # Retrieve candidates (wide net)
    raw_hits: List[Dict[str, Any]] = []
    for q in queries:
        hits = engine.search(
            query=q,
            as_of_date=as_of,
            jurisdiction=None,           # IMPORTANT: do NOT prefilter here; we filter later by evidence
            top_k=50,
            bm25_candidates=500,
            alpha=0.60,
            beta=0.40,
            min_match_ratio=0.15,
            min_semantic_cosine=0.08,
        )
        # keep query info for consistency bonus
        for h in hits:
            h["_q"] = q
        raw_hits.extend(hits)

    # Merge duplicates + count query hits
    best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    hit_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for h in raw_hits:
        d = h.get("doc", {}) or {}
        key = (d.get("act_id"), d.get("section_no"), d.get("version_id"))
        hit_count[key] += 1

        if key not in best_by_key or float(h.get("score", 0.0)) > float(best_by_key[key].get("score", 0.0)):
            best_by_key[key] = h

    candidates = list(best_by_key.values())

    rescored = []
    for h in candidates:
        d = h.get("doc", {}) or {}
        act_jur = d.get("jurisdiction")
        title = d.get("section_title") or ""
        text = d.get("text") or ""

        # Filter generic sections unless strongly supported by overlap
        generic = is_generic_section_title(title)

        # Personal law filtering:
        # - if detected personal law is General and there is no strong evidence for others,
        #   then reject Muslim/Kandyan/Jaffna jurisdictions.
        if personal_law == "General" and strong_hits < STRONG_EVIDENCE_THRESHOLD:
            if act_jur in {"Muslim", "Kandyan", "Jaffna", "Tesawalamai"}:
                continue
        else:
            # if personal law is specific, keep only that jurisdiction (usually correct)
            if personal_law in {"Muslim", "Kandyan", "Jaffna"}:
                if not jurisdiction_allowed(act_jur, personal_law):
                    continue

        sup = support_score(case_tokens, title, text)

        # If generic title, demand higher support
        if generic and sup < 0.06:
            continue

        # Basic minimum support gate
        if sup < 0.03:
            continue

        # consistency bonus: appears across multiple generated queries
        key = (d.get("act_id"), d.get("section_no"), d.get("version_id"))
        qhits = hit_count.get(key, 1)
        consistency = 0.02 * min(8, (qhits - 1))

        # final score (support weighted more than raw search)
        base = float(h.get("score", 0.0))
        final = (0.30 * base) + (0.70 * sup) + consistency

        rescored.append((final, sup, qhits, h))

    rescored.sort(key=lambda x: x[0], reverse=True)

    # diversify so one Act doesn’t dominate unless truly supported
    out = []
    per_act = Counter()
    for final, sup, qhits, h in rescored:
        d = h.get("doc", {}) or {}
        act_id = d.get("act_id")
        if not act_id:
            continue
        if per_act[act_id] >= 4:
            continue

        evidence = best_evidence_sentence(case_text, (d.get("section_title","") + " " + d.get("text","")))

        out.append({
            "act_id": act_id,
            "act_title": d.get("act_title"),
            "jurisdiction": d.get("jurisdiction"),
            "section_no": d.get("section_no"),
            "section_title": d.get("section_title"),
            "version_id": d.get("version_id"),
            "valid_from": d.get("valid_from"),
            "valid_to": d.get("valid_to"),
            "confidence_score": round(final, 3),
            "support_score": round(sup, 3),
            "query_hits": int(qhits),
            "evidence_from_case": evidence,
            "links": {
                "statute_as_of": f"/statute/{act_id}?date={as_of}",
                "timeline": f"/timeline/{act_id}/{d.get('section_no')}",
            }
        })
        per_act[act_id] += 1
        if len(out) >= top_k:
            break

    return {
        **pl,
        "as_of_date": as_of,
        "queries_generated": queries,
        "results_count": len(out),
        "relevant_laws": out
    }