from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple

from orchestrator.reasoning.classifier import (
    CaseCategory,
    UrgencyLevel,
    CATEGORY_KEYWORDS,
    URGENCY_KEYWORDS,
    COMPLEX_CATEGORIES,
)
from orchestrator.reasoning.models import CaseContext

logger = logging.getLogger("juriaid.reasoning.analyzer")


class RuleBasedAnalyzer:
    """
    Deterministic analysis layer.
    No LLM — fast, predictable, auditable.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, case_text: str, user_prompt: str) -> CaseContext:
        """Run all rule-based checks and return a populated CaseContext."""
        combined = case_text.lower() + " " + user_prompt.lower()

        category   = self._classify_category(combined)
        urgency    = self._determine_urgency(combined)
        key_facts  = self._extract_key_facts(case_text)
        issues     = self._identify_legal_issues(combined, category)
        parties    = self._extract_parties(case_text)
        req_s, req_p, req_q = self._decide_tool_requirements(
            user_prompt.lower(), category
        )
        complexity = self._calculate_complexity(
            case_text, key_facts, issues, category
        )

        logger.debug(
            "Rule analysis → category=%s urgency=%s complexity=%.2f",
            category.value, urgency.value, complexity,
        )

        return CaseContext(
            category=category,
            urgency=urgency,
            key_facts=key_facts,
            legal_issues=issues,
            parties=parties,
            requires_statutes=req_s,
            requires_precedents=req_p,
            requires_questions=req_q,
            complexity_score=complexity,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_category(self, text: str) -> CaseCategory:
        scores = {
            cat: sum(1 for kw in kws if kw in text)
            for cat, kws in CATEGORY_KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else CaseCategory.UNKNOWN

    def _determine_urgency(self, text: str) -> UrgencyLevel:
        for level, keywords in URGENCY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return level
        return UrgencyLevel.MEDIUM

    def _extract_key_facts(self, case_text: str) -> List[str]:
        facts: List[str] = []

        # Dates
        dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", case_text)
        if dates:
            facts.append(f"Key dates: {', '.join(dates[:3])}")

        # Monetary amounts (LKR / Rs)
        amounts = re.findall(
            r"(?:Rs\.?|LKR)\s*[\d,]+(?:\.\d{2})?", case_text, re.IGNORECASE
        )
        if amounts:
            facts.append(f"Monetary amounts: {', '.join(amounts[:3])}")

        # Sentences containing core legal verbs
        trigger_terms = {"alleged", "claim", "dispute", "filed"}
        for sentence in case_text.split(".")[:10]:
            if any(t in sentence.lower() for t in trigger_terms):
                facts.append(sentence.strip()[:100] + "…")
            if len(facts) >= 5:
                break

        return facts

    def _identify_legal_issues(
        self, text: str, category: CaseCategory
    ) -> List[str]:
        issues: List[str] = []

        if category == CaseCategory.FAMILY_LAW:
            if "divorce"     in text: issues.append("Grounds for divorce – Divorce Act")
            if "custody"     in text: issues.append("Child custody determination")
            if "maintenance" in text or "alimony" in text:
                issues.append("Spousal/child maintenance – Maintenance Act")

        elif category == CaseCategory.CONTRACT_LAW:
            if "breach"             in text: issues.append("Breach of contract claim")
            if "damages"            in text: issues.append("Assessment of damages")
            if "specific performance" in text: issues.append("Specific performance remedy")

        elif category == CaseCategory.CRIMINAL_LAW:
            if "penal code" in text: issues.append("Penal Code violation")
            if "evidence"   in text: issues.append("Admissibility of evidence")

        elif category == CaseCategory.PROPERTY_LAW:
            if "ownership" in text or "title" in text:
                issues.append("Title / ownership dispute")
            if "mortgage" in text:
                issues.append("Mortgage enforcement")

        elif category == CaseCategory.LABOR_LAW:
            if "termination" in text or "dismissal" in text:
                issues.append("Wrongful termination claim")
            if "wages" in text:
                issues.append("Unpaid wages / compensation")

        if not issues:
            issues.append(f"General {category.value.replace('_', ' ')} matter")

        return issues

    def _extract_parties(self, case_text: str) -> Dict[str, str]:
        parties: Dict[str, str] = {}

        for role, pattern in [
            ("plaintiff", r"(?:plaintiff|petitioner|claimant)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
            ("defendant", r"(?:defendant|respondent)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
        ]:
            m = re.search(pattern, case_text, re.IGNORECASE)
            if m:
                parties[role] = m.group(1)

        return parties or {"plaintiff": "Party A", "defendant": "Party B"}

    def _decide_tool_requirements(
        self, user_prompt: str, category: CaseCategory
    ) -> Tuple[bool, bool, bool]:
        """Returns (requires_statutes, requires_precedents, requires_questions)"""

        # Explicit single-tool requests
        if any(w in user_prompt for w in ("statute", "law", "act")):
            return (True, False, False)
        if any(w in user_prompt for w in ("precedent", "similar case")):
            return (False, True, False)
        if any(w in user_prompt for w in ("question", "intake", "ask")):
            return (True, True, True)   # questions need statute + precedent context
        if any(w in user_prompt for w in ("save", "update", "knowledge base")):
            return (False, False, False)

        # Default: full analysis
        return (True, True, True)

    def _calculate_complexity(
        self,
        case_text: str,
        key_facts: List[str],
        legal_issues: List[str],
        category: CaseCategory,
    ) -> float:
        score = 0.0

        # Text length
        length = len(case_text)
        score += 0.3 if length > 5000 else (0.2 if length > 2000 else 0.1)

        # Facts & issues
        score += min(len(key_facts)   * 0.05, 0.2)
        score += min(len(legal_issues) * 0.10, 0.3)

        # Inherently complex categories
        if category in COMPLEX_CATEGORIES:
            score += 0.2

        return min(score, 1.0)