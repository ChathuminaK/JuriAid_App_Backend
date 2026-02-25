from __future__ import annotations

import logging
from typing import Any, Dict, List

from orchestrator.reasoning.classifier import CaseCategory, UrgencyLevel
from orchestrator.reasoning.models import CaseContext

logger = logging.getLogger("juriaid.reasoning.planner")

# Type alias for a single tool step
ToolStep = Dict[str, Any]


class StrategicPlanner:
    """
    Produces an ordered execution plan based on case context.
    Think of this as the 'senior partner deciding what the associates should do'.
    """

    def build_plan(
        self,
        context: CaseContext,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Build tool plan + reasoning narrative.

        Returns:
            {
              "tools":         [ToolStep, ...],   sorted by priority
              "reasoning":     [str, ...],
              "quality_checks": [...]             filled later by add_quality_controls
            }
        """
        tools:     List[ToolStep] = []
        reasoning: List[str]      = []

        self._rule_complex_summary(context, tools, reasoning)
        self._rule_critical_urgency(context, tools, reasoning)
        self._rule_category_tools(context, tools)
        self._rule_questions(context, tools, reasoning)
        self._rule_knowledge_base(user_prompt, tools, reasoning)

        # Deduplicate (same tool may be added by multiple rules)
        seen:   set[str]      = set()
        unique: List[ToolStep] = []
        for step in tools:
            if step["tool"] not in seen:
                seen.add(step["tool"])
                unique.append(step)

        unique.sort(key=lambda s: s["priority"])

        logger.debug("Plan: %s", [s["tool"] for s in unique])

        return {"tools": unique, "reasoning": reasoning, "quality_checks": []}

    # ------------------------------------------------------------------
    # Individual planning rules
    # ------------------------------------------------------------------

    def _rule_complex_summary(
        self, ctx: CaseContext, tools: List[ToolStep], reasoning: List[str]
    ) -> None:
        """High-complexity cases need a summary first."""
        if ctx.complexity_score > 0.6:
            tools.append({
                "tool":      "summarize_case",
                "priority":  1,
                "rationale": f"High complexity ({ctx.complexity_score:.2f}) — structured summary first",
            })
            reasoning.append("Complex case detected: starting with comprehensive summarisation")

    def _rule_critical_urgency(
        self, ctx: CaseContext, tools: List[ToolStep], reasoning: List[str]
    ) -> None:
        """Critical urgency → statute check jumps to priority 1."""
        if ctx.urgency == UrgencyLevel.CRITICAL:
            tools.append({
                "tool":      "search_law_statutes",
                "priority":  1,
                "rationale": "CRITICAL urgency — immediate statutory framework verification",
            })
            reasoning.append("URGENT: checking statutes for immediate legal options")

    def _rule_category_tools(
        self, ctx: CaseContext, tools: List[ToolStep]
    ) -> None:
        """Add statute/precedent tools according to case category."""
        category_label = ctx.category.value.replace("_", " ").title()

        # Rationale strings by category
        statute_rationale: Dict[CaseCategory, str] = {
            CaseCategory.FAMILY_LAW:         "Family law — checking Divorce Act, Maintenance Act",
            CaseCategory.CONTRACT_LAW:       "Contract law — checking Contracts Ordinance, Sale of Goods",
            CaseCategory.CRIMINAL_LAW:       "Criminal law — checking Penal Code provisions",
            CaseCategory.PROPERTY_LAW:       "Property law — checking Registration of Documents Ordinance",
            CaseCategory.LABOR_LAW:          "Labour law — checking Termination of Employment Act",
            CaseCategory.TORT_LAW:           "Tort law — checking Civil Liability provisions",
            CaseCategory.CONSTITUTIONAL_LAW: "Constitutional law — checking fundamental rights provisions",
        }
        precedent_rationale: Dict[CaseCategory, str] = {
            CaseCategory.FAMILY_LAW:         "Searching Sri Lankan family law precedents",
            CaseCategory.CONTRACT_LAW:       "Searching contract breach precedents",
            CaseCategory.CRIMINAL_LAW:       "Searching criminal case precedents",
            CaseCategory.PROPERTY_LAW:       "Searching property dispute precedents",
            CaseCategory.LABOR_LAW:          "Searching wrongful dismissal precedents",
            CaseCategory.TORT_LAW:           "Searching negligence & tort precedents",
            CaseCategory.CONSTITUTIONAL_LAW: "Searching fundamental rights case precedents",
        }

        if ctx.requires_statutes:
            tools.append({
                "tool":      "search_law_statutes",
                "priority":  2,
                "rationale": statute_rationale.get(
                    ctx.category, f"{category_label} — checking relevant statutes"
                ),
            })

        if ctx.requires_precedents:
            tools.append({
                "tool":      "search_past_cases",
                "priority":  3,
                "rationale": precedent_rationale.get(
                    ctx.category, f"Searching {category_label} precedents"
                ),
            })

    def _rule_questions(
        self, ctx: CaseContext, tools: List[ToolStep], reasoning: List[str]
    ) -> None:
        """Add question generation when context requires it."""
        if ctx.requires_questions:
            tools.append({
                "tool":      "generate_legal_questions",
                "priority":  4,
                "rationale": "Generating targeted questions to gather complete case information",
            })
            reasoning.append("Preparing client intake questions to fill information gaps")

    def _rule_knowledge_base(
        self, user_prompt: str, tools: List[ToolStep], reasoning: List[str]
    ) -> None:
        """Honour explicit save/update requests."""
        save_keywords = ("save", "update", "store", "knowledge base", "add to database")
        if any(kw in user_prompt.lower() for kw in save_keywords):
            tools.append({
                "tool":      "update_knowledge_base",
                "priority":  5,
                "rationale": "User requested case storage in knowledge base",
            })
            reasoning.append("Saving case to knowledge base for future reference")

    # ------------------------------------------------------------------
    # Quality checks (run after plan is built)
    # ------------------------------------------------------------------

    def add_quality_controls(
        self, plan: Dict[str, Any], context: CaseContext
    ) -> Dict[str, Any]:
        """Append quality-control check results to the plan."""
        checks = []
        tool_names = {s["tool"] for s in plan["tools"]}

        # QC-1: Critical/High urgency must have statute search
        if context.urgency in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH):
            checks.append({
                "check":   "statutory_verification",
                "passed":  "search_law_statutes" in tool_names,
                "message": "Critical/High urgency cases must verify statutory provisions",
            })

        # QC-2: Family law should always generate intake questions
        if context.category == CaseCategory.FAMILY_LAW:
            checks.append({
                "check":   "family_law_intake",
                "passed":  "generate_legal_questions" in tool_names,
                "message": "Family law cases should generate client intake questions",
            })

        # QC-3: Complex cases (>0.7) should include precedent research
        if context.complexity_score > 0.7:
            checks.append({
                "check":   "precedent_research",
                "passed":  "search_past_cases" in tool_names,
                "message": "Complex cases should include precedent research",
            })

        plan["quality_checks"] = checks
        return plan