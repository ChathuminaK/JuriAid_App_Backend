import io
import re
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract and clean text from PDF bytes using PyMuPDF."""
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)

    doc.close()
    full_text = "\n".join(pages)

    # Clean
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"^\s*\d+\s*$", "", full_text, flags=re.MULTILINE)
    full_text = full_text.strip()

    if not full_text:
        raise ValueError("Could not extract text from PDF")

    logger.info(f"📄 Extracted {len(full_text)} chars from {len(pages)} pages")
    return full_text