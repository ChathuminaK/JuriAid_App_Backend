from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from orchestrator.reasoning.classifier import CaseCategory, UrgencyLevel


@dataclass
class CaseContext:
    """
    Structured case context produced by RuleBasedAnalyzer.
    Passed into the planner and quality checker.
    """
    category:            CaseCategory
    urgency:             UrgencyLevel
    key_facts:           List[str]
    legal_issues:        List[str]
    parties:             Dict[str, str]
    requires_statutes:   bool
    requires_precedents: bool
    requires_questions:  bool
    complexity_score:    float          # 0.0 – 1.0

    def to_dict(self) -> dict:
        """Serialize for JSON / LLM prompt injection"""
        return {
            "category":            self.category.value,
            "urgency":             self.urgency.value,
            "complexity":          round(self.complexity_score, 2),
            "key_facts":           self.key_facts,
            "legal_issues":        self.legal_issues,
            "parties":             self.parties,
            "requires_statutes":   self.requires_statutes,
            "requires_precedents": self.requires_precedents,
            "requires_questions":  self.requires_questions,
        }