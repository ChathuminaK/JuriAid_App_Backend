import re
from collections import Counter

# Legal keywords list
LEGAL_KEYWORDS = [
    "plaintiff", "defendant", "court", "judgment", "appeal",
    "petitioner", "respondent", "section", "act", "law",
    "order", "evidence", "trial"
]

def is_legal_document(text: str, threshold=5, min_words=50, debug=False):
    """
    Production-ready check if a document is legal.
    
    Parameters:
        text (str): Document text
        threshold (int): Minimum number of legal keywords required
        min_words (int): Minimum words in document
        debug (bool): If True, print found keywords info
    
    Returns:
        bool: True if legal, False otherwise
    """
    
    # Quick reject: too short document
    words_in_doc = text.split()
    if len(words_in_doc) < min_words:
        if debug: print(f"Rejected: too short ({len(words_in_doc)} words)")
        return False

    text_lower = text.lower()

    # Count keyword occurrences
    keyword_counts = Counter()
    for word in LEGAL_KEYWORDS:
        matches = re.findall(rf"\b{word}\b", text_lower)
        if matches:
            keyword_counts[word] = len(matches)

    total_keywords_found = sum(keyword_counts.values())
    
    if debug: 
        print(f"Keywords found: {keyword_counts}")
        print(f"Total keywords found: {total_keywords_found}")

    # Threshold check
    if total_keywords_found < threshold:
        if debug: print(f"Rejected: less than threshold ({threshold})")
        return False

    # Passed all checks
    return True