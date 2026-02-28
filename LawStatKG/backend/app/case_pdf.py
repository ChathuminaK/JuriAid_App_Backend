import fitz  # PyMuPDF

def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for page in doc:
        txt = page.get_text("text") or ""
        parts.append(txt)
    doc.close()
    # normalize whitespace a bit
    text = "\n".join(parts)
    text = text.replace("\r", "\n")
    return text