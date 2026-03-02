def explain(result):
    return {
        "case_id": result["case_id"],
        "final_score": result["final_score"],
        "explanation": result["breakdown"]
    }