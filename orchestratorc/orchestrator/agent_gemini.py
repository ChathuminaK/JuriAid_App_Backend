import os, json, asyncio
from datetime import datetime
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import google.generativeai as genai

# Use absolute import and get API key from config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

from orchestrator.core import analyze_case_content, generate_legal_summary
from orchestrator.tools import (
    search_law_statutes,
    search_past_cases,
    generate_legal_questions,
    summarize_case as summarize_case_tool,
    update_knowledge_base as update_kb_tool
)

# Configure Gemini with API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)

# ── Async-to-sync bridge ──────────────────────────────────────
def _run_async(coro):
    """Run an async tool synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def retrieve_family_law_statutes(case_text, **kwargs):
    result = _run_async(search_law_statutes.ainvoke(case_text[:500]))
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return [{"text": result}]
    return result

def find_divorce_case_precedents(case_text, **kwargs):
    result = _run_async(search_past_cases.ainvoke(case_text[:500]))
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return [{"text": result}]
    return result

def generate_family_law_client_questions(case_text, statutes=None, precedents=None, summary=None, **kwargs):
    law_ctx = json.dumps(statutes)[:500] if statutes else "N/A"
    cases_ctx = json.dumps(precedents)[:500] if precedents else "N/A"
    result = _run_async(generate_legal_questions.ainvoke({
        "case_text": case_text[:500],
        "law_context": law_ctx,
        "past_cases_context": cases_ctx
    }))
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"questions": [result]}
    return result

def summarize_case(case_text, **kwargs):
    result = _run_async(summarize_case_tool.ainvoke(case_text[:1000]))
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"summary": result}
    return result

def update_knowledge_base(case_text, meta=None, kb_path=None, **kwargs):
    case_type = "General"
    if isinstance(meta, dict):
        case_type = meta.get("case_type", "General")
    result = _run_async(update_kb_tool.ainvoke({
        "case_text": case_text[:500],
        "case_type": case_type
    }))
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"status": result}
    return result

# ── Tool registry (unchanged names for GeminiPlannerAgent) ────
TOOL_REGISTRY = {
    "family_statutes": retrieve_family_law_statutes,
    "family_precedents": find_divorce_case_precedents,
    "family_questions": generate_family_law_client_questions,
    "summarize": summarize_case,
    "update_kb": update_knowledge_base,
}

SYSTEM_PROMPT = """You are a legal planning agent for Sri Lankan law. Output ONLY JSON:
{
 "steps":[
   {"tool":"tool_name","rationale":"short reason"}
 ],
 "notes":"overall reasoning"
}

Allowed tools: family_statutes, family_precedents, family_questions, summarize, update_kb.

IMPORTANT RULES:
1. If user asks to "update knowledge base", "save case", "add to database", or "update knowledge graph":
   - ONLY use: update_kb (skip all other tools)

2. If user asks to "find statutes only", "show me laws", or "legal framework only":
   - ONLY use: family_statutes

3. If user asks for "precedents only" or "similar cases only":
   - ONLY use: family_precedents

4. If user asks for "questions only" or "client intake":
   - Use: family_statutes, family_precedents, family_questions (questions need context)

5. For normal analysis requests:
   - Use appropriate combination: family_statutes → family_precedents → family_questions → summarize

6. Always use update_kb if user explicitly asks to save/update/store the case.

No extra text outside JSON.
"""

PREFERRED_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash-exp",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
]

def _resolve_model() -> str:
    """Pick a Gemini model. Tries list_models first, falls back to hardcoded name."""
    try:
        all_supported = [
            m.name for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        for pm in PREFERRED_MODELS:
            if pm in all_supported:
                return pm
        for name in all_supported:
            low = name.lower()
            if any(k in low for k in ["2.5-flash", "2.5-pro", "flash-latest", "pro-latest", "gemini-flash", "gemini-pro"]):
                return name
        if all_supported:
            return all_supported[0]
    except Exception as e:
        print(f"[GeminiPlannerAgent] list_models() failed: {e}")
        print("[GeminiPlannerAgent] Using default model: gemini-2.5-flash")
    return "gemini-2.5-flash"

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

        context_data = {
            "case_text": case_text,
            "case_type": analysis["case_type"],
            "statutes": None,
            "precedents": None,
            "summary": None
        }

        print(f"[GeminiAgent] Executing {len(plan['steps'])} planned steps...")

        for step in plan["steps"]:
            name = step["tool"]
            fn = TOOL_REGISTRY.get(name)
            if not fn:
                results.append({"tool": name, "error": "not implemented"})
                continue

            print(f"[GeminiAgent] Executing tool: {name}")

            if name == "update_kb":
                out = fn(case_text, {"case_type": analysis["case_type"], "tags": self._tags(case_text)}, self.kb_path)
            elif name == "summarize":
                out = fn(case_text)
                context_data["summary"] = out
            elif name == "family_statutes":
                out = fn(case_text)
                context_data["statutes"] = out
            elif name == "family_precedents":
                out = fn(case_text)
                context_data["precedents"] = out
            elif name == "family_questions":
                out = fn(
                    case_text=case_text,
                    statutes=context_data.get("statutes"),
                    precedents=context_data.get("precedents"),
                    summary=context_data.get("summary")
                )
            else:
                out = fn()

            results.append({"tool": name, "rationale": step.get("rationale",""), "output": out})

        if context_data["summary"] is None:
            summary_result = summarize_case(case_text)
            results.append({"tool": "core_summary", "output": summary_result})

        return {
            "success": True,
            "case_type": analysis["case_type"],
            "plan": plan["steps"],
            "notes": plan["notes"],
            "results": results,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }

    async def create_plan(self, case_snippet: str, user_prompt: str = None):
        # VALIDATION: Prevent empty contents error
        if not case_snippet or case_snippet.strip() == "":
            raise ValueError("Case snippet cannot be empty")
        
        if not user_prompt:
            user_prompt = "Analyze this case comprehensively"
        
        # BUILD PROPER REQUEST
        prompt = f"""
                    You are a legal AI assistant for Sri Lankan law.

                    User Request: {user_prompt}

                    Case Document:
                    {case_snippet}

                    Create a detailed legal analysis plan...
                    """
        
        # SEND TO GEMINI (ensure contents is properly structured)
        response = self.model.generate_content([
            {"role": "user", "parts": [{"text": prompt}]}
        ])
        return response