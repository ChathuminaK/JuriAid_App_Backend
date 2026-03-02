import fitz  # PyMuPDF
import logging
import io

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.
    Returns extracted text or empty string on failure.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text.strip())

        doc.close()

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            logger.warning("PDF opened but no text extracted (possibly scanned/image PDF)")
            return ""

        logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} pages")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""