from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from orchestrator.reasoning.analyzer import RuleBasedAnalyzer
from orchestrator.reasoning.planner  import StrategicPlanner

logger = logging.getLogger("juriaid.reasoning.engine")


class HybridReasoningEngine:
    """
    Mimics how a senior lawyer thinks through a case:
      1. Rule-based classification  (fast, deterministic)
      2. Strategic tool planning    (rule-driven priority ordering)
      3. LLM insight merge          (optional — enriches plan)
      4. Quality validation         (automated QC checks)
    """

    def __init__(self) -> None:
        self._analyzer = RuleBasedAnalyzer()
        self._planner  = StrategicPlanner()

    # ------------------------------------------------------------------
    # Public API (called by agent_langchain.py :: reason_node)
    # ------------------------------------------------------------------

    def analyze_and_plan(
        self,
        case_text:    str,
        user_prompt:  str,
        llm_analysis: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full hybrid analysis pipeline.

        Args:
            case_text:    Raw case document text
            user_prompt:  User's request / question
            llm_analysis: Optional JSON string from a prior LLM pass

        Returns:
            {
              "case_context":      { category, urgency, complexity, … },
              "recommended_tools": [ {tool, priority, rationale}, … ],
              "reasoning":         [ str, … ],
              "quality_checks":    [ {check, passed, message}, … ],
            }
        """
        logger.info("Hybrid reasoning started")

        # 1 ── Rule-based analysis
        context = self._analyzer.analyze(case_text, user_prompt)

        # 2 ── Strategic plan
        plan = self._planner.build_plan(context, user_prompt)

        # 3 ── Optionally enrich with LLM insights
        if llm_analysis:
            plan = self._merge_llm_insights(plan, llm_analysis)

        # 4 ── Quality controls
        plan = self._planner.add_quality_controls(plan, context)

        logger.info(
            "Reasoning complete | category=%s urgency=%s complexity=%.2f tools=%s",
            context.category.value,
            context.urgency.value,
            context.complexity_score,
            [s["tool"] for s in plan["tools"]],
        )

        return {
            "case_context":      context.to_dict(),
            "recommended_tools": plan["tools"],
            "reasoning":         plan["reasoning"],
            "quality_checks":    plan["quality_checks"],
        }

    # ------------------------------------------------------------------
    # LLM insight merge (private)
    # ------------------------------------------------------------------

    def _merge_llm_insights(
        self, plan: Dict[str, Any], llm_analysis: str
    ) -> Dict[str, Any]:
        """
        Attempts to parse structured JSON from the LLM.
        Falls back to appending the raw text as a reasoning note.
        """
        try:
            data = json.loads(llm_analysis)

            # Additional free-text considerations
            if "additional_considerations" in data:
                plan["reasoning"].extend(data["additional_considerations"])

            # Extra tool suggestions (appended at lowest priority)
            existing = {s["tool"] for s in plan["tools"]}
            for tool_name in data.get("suggested_tools", []):
                if tool_name not in existing:
                    plan["tools"].append({
                        "tool":      tool_name,
                        "priority":  99,
                        "rationale": "Suggested by LLM analysis",
                    })
                    existing.add(tool_name)

        except (json.JSONDecodeError, TypeError):
            # Non-JSON LLM output → treat as a reasoning note
            snippet = llm_analysis[:200].replace("\n", " ")
            plan["reasoning"].append(f"AI insight: {snippet}")

        return plan


# ---------------------------------------------------------------------------
# Factory (keeps existing import in agent_langchain.py working unchanged)
# ---------------------------------------------------------------------------

def create_reasoning_engine() -> HybridReasoningEngine:
    """Returns a ready-to-use HybridReasoningEngine instance."""
    return HybridReasoningEngine()