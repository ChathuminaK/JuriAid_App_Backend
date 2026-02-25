from __future__ import annotations
from enum import Enum


class CaseCategory(Enum):
    """Sri Lankan legal case categories"""
    FAMILY_LAW          = "family_law"
    CONTRACT_LAW        = "contract_law"
    CRIMINAL_LAW        = "criminal_law"
    PROPERTY_LAW        = "property_law"
    LABOR_LAW           = "labor_law"
    TORT_LAW            = "tort_law"
    CONSTITUTIONAL_LAW  = "constitutional_law"
    UNKNOWN             = "unknown"


class UrgencyLevel(Enum):
    """Case urgency levels"""
    CRITICAL = "critical"   # Immediate legal action required
    HIGH     = "high"       # Time-sensitive matter
    MEDIUM   = "medium"     # Standard legal timeline
    LOW      = "low"        # No immediate deadline


# ---------------------------------------------------------------------------
# Keyword maps (used by RuleBasedAnalyzer)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[CaseCategory, list[str]] = {
    CaseCategory.FAMILY_LAW: [
        "divorce", "custody", "alimony", "maintenance", "marriage",
        "adoption", "guardianship", "matrimonial", "child support",
        "separation", "domestic violence", "dowry",
    ],
    CaseCategory.CONTRACT_LAW: [
        "contract", "breach", "agreement", "terms", "warranty",
        "consideration", "offer", "acceptance", "damages",
        "specific performance",
    ],
    CaseCategory.CRIMINAL_LAW: [
        "theft", "assault", "murder", "fraud", "criminal", "penal code",
        "arrest", "prosecution", "accused", "victim", "crime",
    ],
    CaseCategory.PROPERTY_LAW: [
        "property", "land", "deed", "ownership", "title", "lease",
        "mortgage", "easement", "boundary", "immovable", "conveyance",
    ],
    CaseCategory.LABOR_LAW: [
        "employment", "termination", "wrongful dismissal", "workplace",
        "labor", "employee", "employer", "discrimination", "wages",
    ],
    CaseCategory.TORT_LAW: [
        "negligence", "defamation", "liability", "injury", "tort",
        "damages", "compensation", "accident", "personal injury",
    ],
    CaseCategory.CONSTITUTIONAL_LAW: [
        "fundamental rights", "constitution", "judicial review",
        "supreme court", "constitutional", "article", "writ",
    ],
}

URGENCY_KEYWORDS: dict[UrgencyLevel, list[str]] = {
    UrgencyLevel.CRITICAL: [
        "immediate", "urgent", "emergency", "interim relief",
        "injunction", "restraining order", "imminent", "arrest warrant",
    ],
    UrgencyLevel.HIGH: [
        "time-sensitive", "deadline", "expiring",
        "statute of limitations", "notice period", "response required",
    ],
    UrgencyLevel.MEDIUM: [
        "pending", "scheduled", "hearing date", "filing required",
    ],
}

# Categories that add extra complexity weight
COMPLEX_CATEGORIES: list[CaseCategory] = [
    CaseCategory.CONSTITUTIONAL_LAW,
    CaseCategory.CRIMINAL_LAW,
    CaseCategory.PROPERTY_LAW,
]