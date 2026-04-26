import fitz  # PyMuPDF
import re


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []

    for page in doc:
        txt = page.get_text("text")
        if txt:
            parts.append(txt)

    doc.close()

    full_text = "\n".join(parts)
    full_text = _clean_pdf_text(full_text)

    # keep more content than before; 6000 words is much safer for factual extraction
    words = full_text.split()
    return " ".join(words[:6000])