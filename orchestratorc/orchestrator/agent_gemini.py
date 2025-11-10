import os, json
from datetime import datetime
from typing import Dict, Any, List
import google.generativeai as genai

from ..config import require_env
from .core import analyze_case_content, generate_legal_summary
from .tools import (
    retrieve_family_law_statutes,
    find_divorce_case_precedents,
    generate_family_law_client_questions,
    retrieve_contract_law_statutes,
    generate_contract_review_questions,
    summarize_case,
    update_knowledge_base
)

TOOL_REGISTRY = {
    "family_statutes": retrieve_family_law_statutes,
    "family_precedents": find_divorce_case_precedents,
    "family_questions": generate_family_law_client_questions,
    "contract_statutes": retrieve_contract_law_statutes,
    "contract_questions": generate_contract_review_questions,
    "summarize": summarize_case,
    "update_kb": update_knowledge_base,
}

SYSTEM_PROMPT = """You are a legal planning agent. Output ONLY JSON:
{
 "steps":[
   {"tool":"tool_name","rationale":"short reason"}
 ],
 "notes":"overall reasoning"
}
Allowed tools: family_statutes, family_precedents, family_questions, contract_statutes,
contract_questions, summarize, update_kb.
Include summarize unless explicitly not needed. Use update_kb if user wants save/update/knowledge base.
Match family vs contract by case type or prompt keywords. No extra text outside JSON.
"""

# Configure Gemini using .env (loaded via config.py)
genai.configure(api_key=require_env("GEMINI_API_KEY"))

PREFERRED_MODELS = [
    "models/gemini-2.5-flash",            # fast, current
    "models/gemini-2.5-pro",              # higher quality
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-001",
]

def _resolve_model() -> str:
    """Pick first preferred model supporting generateContent; fall back to any flash/pro model."""
    all_supported = [
        m.name for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]
    # Try preferred list
    for pm in PREFERRED_MODELS:
        if pm in all_supported:
            return pm
    # Fallback: choose first flash/pro/gemma model
    for name in all_supported:
        low = name.lower()
        if any(k in low for k in ["2.5-flash", "2.5-pro", "flash-latest", "pro-latest", "2.0-flash", "gemini-flash", "gemini-pro"]):
            return name
    # Absolute last resort
    if all_supported:
        return all_supported[0]
    raise RuntimeError(f"No usable Gemini model found. Supported list empty.")

class GeminiPlannerAgent:
    def __init__(self, kb_path: str):
        model_name = _resolve_model()
        self.model = genai.GenerativeModel(model_name)
        self.kb_path = kb_path
        print(f"[GeminiPlannerAgent] Using model: {model_name}")

    def _build_user(self, case_text: str, prompt: str, case_type: str) -> str:
        return f"CaseType: {case_type}\nUserPrompt: {prompt}\nCaseSnippet:\n'''{case_text[:1200]}'''\nProduce plan."

    def _get_plan(self, case_text: str, prompt: str, case_type: str) -> Dict[str, Any]:
        resp = self.model.generate_content([SYSTEM_PROMPT, self._build_user(case_text, prompt, case_type)])
        raw = (resp.text or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Planner did not return JSON.")
        data = json.loads(raw[start:end+1])
        steps = [s for s in data.get("steps", []) if s.get("tool") in TOOL_REGISTRY]
        return {"steps": steps, "notes": data.get("notes", "")}

    def _tags(self, case_text: str) -> List[str]:
        t = case_text.lower()
        return [kw for kw in ["divorce","custody","maintenance","contract","breach","agreement"] if kw in t]

    def run(self, case_text: str, prompt: str) -> Dict[str, Any]:
        analysis = analyze_case_content(case_text)
        plan = self._get_plan(case_text, prompt, analysis["case_type"])
        results = []
        for step in plan["steps"]:
            name = step["tool"]
            fn = TOOL_REGISTRY.get(name)
            if not fn:
                results.append({"tool": name, "error": "not implemented"})
                continue
            if name == "update_kb":
                out = fn(case_text, {"case_type": analysis["case_type"], "tags": self._tags(case_text)}, self.kb_path)
            elif name == "summarize":
                out = fn(case_text)
            else:
                out = fn()
            results.append({"tool": name, "rationale": step.get("rationale",""), "output": out})
        results.append({"tool": "core_summary", "output": generate_legal_summary(case_text, analysis)})
        return {
            "success": True,
            "case_type": analysis["case_type"],
            "plan": plan["steps"],
            "notes": plan["notes"],
            "results": results,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }