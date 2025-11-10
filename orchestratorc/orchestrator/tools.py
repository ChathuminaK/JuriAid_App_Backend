import os, json
from datetime import datetime

def retrieve_family_law_statutes():
    return "Family Law Statutes: Marriage Registration Ordinance; Maintenance Act; Matrimonial Causes Ordinance."

def find_divorce_case_precedents():
    return "Divorce precedents: Case A v B (custody principles); Case C v D (maintenance)."

def generate_family_law_client_questions():
    return [
        "Date of marriage?",
        "Existing custody arrangements?",
        "Financial contributions of each spouse?",
        "Prior separation or mediation attempts?"
    ]

def retrieve_contract_law_statutes():
    return "Contract Law Sources: Contracts Ordinance; Sale of Goods Ordinance; breach & remedies case law."

def generate_contract_review_questions():
    return [
        "Key obligations of parties?",
        "Termination clause present?",
        "Penalty / liquidated damages clauses?",
        "Jurisdiction & governing law specified?"
    ]

def summarize_case(case_text: str):
    return (case_text[:350] + "...") if len(case_text) > 350 else case_text

def update_knowledge_base(case_text: str, metadata: dict, kb_path: str):
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {"entries": []}
    entry = {
        "id": len(kb["entries"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "case_type": metadata.get("case_type"),
        "length": len(case_text),
        "snippet": case_text[:400],
        "tags": metadata.get("tags", [])
    }
    kb["entries"].append(entry)
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    return {"status": "updated", "total_entries": len(kb["entries"])}