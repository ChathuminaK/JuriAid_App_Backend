import re



def extract_year(text: str):
    # Extract 4-digit year between 1950–2035
    match = re.search(r"(19[5-9]\d|20[0-3]\d)", text)
    return int(match.group()) if match else None


def extract_case_name(text: str):
    lines = text.split("\n")

    # Check first 15 lines
    for line in lines[:15]:
        if " v. " in line.lower() or " vs " in line.lower():
            return line.strip()

    # fallback if not found
    return lines[0][:150]


def extract_legal_issues(roles: dict):
    # Use issue sentences as legal issue keywords
    issues = []
    for sentence in roles.get("issues", []):
        words = sentence.split()
        if len(words) > 2:
            issues.append(" ".join(words[:3]))
    return list(set(issues))[:5]

    