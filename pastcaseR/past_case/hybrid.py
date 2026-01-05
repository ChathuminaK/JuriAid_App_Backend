# backend/hybrid.py
from vector_store import search_index
from knowledge_graph import (
    create_case_node,
    get_direct_citation,
    get_reverse_citation,
    get_shared_neighbors_count
)
from config import ALPHA_VECTOR, BETA_KG

def compute_kg_score(query_case_id: str, candidate_case_id: str) -> float:
    """
    Compute KG score between query node and candidate node
    """
    score = 0.0
    if get_direct_citation(query_case_id, candidate_case_id):
        score += 0.30
    if get_reverse_citation(query_case_id, candidate_case_id):
        score += 0.20
    shared = get_shared_neighbors_count(query_case_id, candidate_case_id)
    score += 0.10 * min(shared, 3)
    return float(score)

def hybrid_rank(query_vec, query_case_id=None, topk=10, candidate_topk=50):
    vec_hits = search_index(query_vec, topk=candidate_topk)
    ranked = []

    if query_case_id:
        create_case_node(query_case_id, title="__TEMP__")

    for hit in vec_hits:
        meta = hit["meta"]
        candidate_case_id = meta.get("case_id")
        vector_score = float(hit["score"]) # Usually between 0 and 1 for normalized vectors
        
        kg_score = 0.0
        if query_case_id:
            kg_score = compute_kg_score(query_case_id, candidate_case_id)

        final_score = (ALPHA_VECTOR * vector_score) + (BETA_KG * kg_score)

        # Only add results that meet a minimum similarity threshold
        # Adjust 0.3 based on your model's performance
        if final_score > 0.30:
            ranked.append({
                "case_id": candidate_case_id,
                "final_score": final_score,
                "vector_score": vector_score,
                "kg_score": kg_score,
                "role": meta.get("role"),
                "snippet": meta.get("snippet", "")[:800]
            })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked[:topk]