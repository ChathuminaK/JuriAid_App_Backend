def classify_text(text: str):
    sentences = text.split(".")
    n = len(sentences)

    return {
        "facts": sentences[:n//3],
        "issues": sentences[n//3:2*n//3],
        "arguments": sentences[2*n//3:]
    }