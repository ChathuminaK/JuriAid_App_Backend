import re



def extract_year(text: str):
    # Extract 4-digit year between 1950–2035
    match = re.search(r"(19[5-9]\d|20[0-3]\d)", text)
    return int(match.group()) if match else None

def extract_case_number(text: str):

    patterns = [
        r"C\.?\s*A\.?\s*\d+\/\d+",
        r"S\.?\s*C\.?\s*\d+",
        r"D\/\d+",
        r"\d+\/\d+\/D",
        r"Case\s*No\.?\s*\d+\/\d+",
        r"Case\s*Number\s*:\s*\d+\/\d+\/?[A-Z]*"
    ]

    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group().strip()

    return "Unknown Case Number"

def extract_case_name(text: str):

    lines = text.split("\n")

    # remove empty lines
    lines = [l.strip() for l in lines if l.strip()]

    # Pattern 1 — detect "A v B"
    for line in lines[:30]:
        if re.search(r"\b(v\.?|vs\.?|versus)\b", line, re.IGNORECASE):
            return line.strip()

    # Pattern 2 — detect Plaintiff / Defendant structure
    plaintiff = None
    defendant = None

    for i, line in enumerate(lines[:50]):

        if "plaintiff" in line.lower():
            plaintiff = lines[i-1] if i > 0 else None

        if "defendant" in line.lower():
            defendant = lines[i-1] if i > 0 else None

        if plaintiff and defendant:
            return f"{plaintiff} v {defendant}"

    # Pattern 3 — fallback (avoid court headings)
    for line in lines[:10]:
        if "court" not in line.lower():
            return line[:150]

    return "Unknown Case"


def extract_legal_issues(roles: dict):
    # Use issue sentences as legal issue keywords
    issues = []
    for sentence in roles.get("issues", []):
        words = sentence.split()
        if len(words) > 2:
            issues.append(" ".join(words[:3]))
    return list(set(issues))[:5]

    