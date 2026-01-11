import os, json
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
    retrieve_family_law_statutes,
    find_divorce_case_precedents,
    generate_family_law_client_questions,
    summarize_case,
    update_knowledge_base
)

# Configure Gemini with API key from settings
genai.configure(api_key=settings.GEMINI_API_KEY)

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
    "models/gemini-1.5-flash",            
    "models/gemini-1.5-pro",              
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash-exp",        
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
        if any(k in low for k in ["2.0-flash", "flash-latest", "pro-latest", "gemini-flash", "gemini-pro"]):
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
        
        # Store outputs from previous tools for context
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
            
            # Execute tools and capture their outputs
            if name == "update_kb":
                out = fn(case_text, {"case_type": analysis["case_type"], "tags": self._tags(case_text)}, self.kb_path)
            
            elif name == "summarize":
                out = fn(case_text)
                context_data["summary"] = out  # Store for family_questions
                print(f"[GeminiAgent] Summary stored: {out.get('title', 'N/A')}")
            
            elif name == "family_statutes":
                out = fn(case_text)
                context_data["statutes"] = out  # Store for family_questions
                print(f"[GeminiAgent] Statutes stored: {len(out)} items")
            
            elif name == "family_precedents":
                out = fn(case_text)
                context_data["precedents"] = out  # Store for family_questions
                print(f"[GeminiAgent] Precedents stored: {len(out)} items")
            
            elif name == "family_questions":
                # Pass ALL previous context
                print(f"[GeminiAgent] Generating family questions with context:")
                print(f"  - Statutes: {len(context_data['statutes']) if context_data['statutes'] else 0}")
                print(f"  - Precedents: {len(context_data['precedents']) if context_data['precedents'] else 0}")
                print(f"  - Summary: {'Yes' if context_data['summary'] else 'No'}")
                
                out = fn(
                    case_text=case_text,
                    statutes=context_data.get("statutes"),
                    precedents=context_data.get("precedents"),
                    summary=context_data.get("summary")
                )
            
            else:
                out = fn()
            
            results.append({"tool": name, "rationale": step.get("rationale",""), "output": out})
        
        # Generate comprehensive summary using AI if not already done
        if context_data["summary"] is None:
            print("[GeminiAgent] Generating core summary...")
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