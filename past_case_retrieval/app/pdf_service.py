import fitz  # PyMuPDF
import io

def extract_text_from_pdf_bytes(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text