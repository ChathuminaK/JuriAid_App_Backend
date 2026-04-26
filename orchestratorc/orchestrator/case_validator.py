import logging

logger = logging.getLogger(__name__)

# Keywords that indicate a Sri Lankan divorce/matrimonial case
# Covers both English and transliterated Sinhala/Tamil legal terms
_DIVORCE_KEYWORDS = [
    # English legal terms
    "divorce",
    "dissolution of marriage",
    "malicious desertion",
    "matrimonial",
    "matrimonial fault",
    "adultery",
    "marriage certificate",
    "married life",
    "marital home",
    "matrimonial home",
    "custody",
    "child custody",
    "legal custody",
    "physical custody",
    "maintenance",
    "alimony",
    "separation",
    "cohabitation",
    "conjugal",
    "husband",
    "wife",
    "spouse",
    "decree nisi",
    "decree absolute",
    "vinculo matrimonii",
    # Sri Lankan court/legal identifiers
    "district court",
    "civil procedure code",
    "marriage registration ordinance",
    "marriages ordinance",
    "matrimonial causes",
    "general marriages",
    # Case nature identifiers from Sri Lankan courts
    "/d/",       # e.g. 6421/D/24
    "/dv/",      # e.g. DV 16553
    "nature: divorce",
    "nature : divorce",
    # Transliterated Sinhala terms commonly found in translated cases
    "plaint",
    "plaintiff",
    "defendant",
    "prayer",
]

# Keywords that indicate it is NOT a divorce case
_EXCLUSION_KEYWORDS = [
    "murder",
    "homicide",
    "theft",
    "robbery",
    "drug trafficking",
    "customs ordinance",
    "income tax",
    "company law",
    "intellectual property",
    "patent",
    "trademark",
]

# Minimum number of divorce keywords that must match
_MIN_KEYWORD_MATCHES = 3

# Minimum number of strong indicators (these alone can validate)
_STRONG_INDICATORS = [
    "divorce",
    "dissolution of marriage",
    "malicious desertion",
    "matrimonial fault",
    "matrimonial causes",
    "decree nisi",
    "decree absolute",
    "vinculo matrimonii",
    "nature: divorce",
    "nature : divorce",
]
_MIN_STRONG_MATCHES = 1


def validate_divorce_case(case_text: str) -> tuple[bool, dict]:
    """
    Validate that the extracted PDF text is a Sri Lankan divorce/matrimonial case.

    Args:
        case_text: Text extracted from the uploaded PDF.

    Returns:
        (is_valid, details_dict)
        - is_valid: True if the case appears to be a Sri Lankan divorce case
        - details_dict: Contains reason, matched_keywords count, etc.
    """
    if not case_text or len(case_text.strip()) < 100:
        return False, {
            "reason": "Document contains insufficient text. Please upload a valid Sri Lankan divorce case PDF.",
            "matched_keywords": 0,
            "strong_matches": 0,
        }

    text_lower = case_text.lower()

    # Check for exclusion keywords first
    exclusion_matches = [kw for kw in _EXCLUSION_KEYWORDS if kw in text_lower]
    if len(exclusion_matches) >= 2:
        return False, {
            "reason": f"This document appears to be a non-matrimonial case ({', '.join(exclusion_matches[:3])}). JuriAid only supports Sri Lankan divorce and matrimonial cases.",
            "matched_keywords": 0,
            "strong_matches": 0,
            "exclusion_matches": exclusion_matches,
        }

    # Count matching divorce keywords
    matched = [kw for kw in _DIVORCE_KEYWORDS if kw in text_lower]
    strong_matched = [kw for kw in _STRONG_INDICATORS if kw in text_lower]

    is_valid = (
        len(strong_matched) >= _MIN_STRONG_MATCHES
        or len(matched) >= _MIN_KEYWORD_MATCHES
    )

    details = {
        "matched_keywords": len(matched),
        "strong_matches": len(strong_matched),
    }

    if is_valid:
        logger.info(
            f"Case validation PASSED: {len(matched)} keywords, "
            f"{len(strong_matched)} strong indicators"
        )
        details["reason"] = "Valid Sri Lankan divorce/matrimonial case"
    else:
        logger.warning(
            f"Case validation FAILED: only {len(matched)} keywords, "
            f"{len(strong_matched)} strong indicators"
        )
        details["reason"] = (
            "This document does not appear to be a Sri Lankan divorce or matrimonial case. "
            "JuriAid is specialized for Sri Lankan divorce case analysis. "
            "Please upload a valid divorce plaint, answer, or judgment."
        )

    return is_valid, details