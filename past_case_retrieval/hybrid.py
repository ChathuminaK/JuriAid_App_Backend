# backend/hybrid.py
from vector_store import search_index
from knowledge_graph import get_direct_citation, get_reverse_citation, get_shared_neighbors_count
from config import ALPHA_VECTOR, BETA_KG

def compute_kg_score(query_case_id: str, candidate_case_id: str) -> float:
    """
    Simple weighted KG scoring:
      +0.30 if query cites candidate
      +0.20 if candidate cites query
      +0.10 * min(shared_neighbors, 3)
    Returns normalized kg score (0..~1)
    """
    score = 0.0
    if get_direct_citation(query_case_id, candidate_case_id):
        score += 0.30
    if get_reverse_citation(query_case_id, candidate_case_id):
        score += 0.20
    shared = get_shared_neighbors_count(query_case_id, candidate_case_id)
    score += 0.10 * min(shared, 3)
    # cap if necessary
    return float(score)

def hybrid_rank(query_vec, query_case_id="__QUERY__", topk=10, candidate_topk=50):
    """
    1) retrieve candidate_topk by vector similarity
    2) compute kg score for each
    3) combine using ALPHA_VECTOR and BETA_KG
    """
    vec_hits = search_index(query_vec, topk=candidate_topk)
    ranked = []
    for hit in vec_hits:
        meta = hit["meta"]
        candidate_case_id = meta.get("case_id")
        vector_score = float(hit["score"])
        kg_score = compute_kg_score(query_case_id, candidate_case_id)
        final_score = ALPHA_VECTOR * vector_score + BETA_KG * kg_score
        ranked.append({
            "case_id": candidate_case_id,
            "final_score": final_score,
            "vector_score": vector_score,
            "kg_score": kg_score,
            "role": meta.get("role"),
            "snippet": meta.get("snippet", "")[:800]
        })
    # sort
    ranked = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
    return ranked[:topk]
