# backend/extractor.py
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import io
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_pdfplumber(path):
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)
    except Exception:
        return ""
    return "\n".join(text)

def extract_text_pymupdf(path):
    try:
        doc = fitz.open(path)
        pages = []
        for p in doc:
            t = p.get_text("text")
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception:
        return ""

def ocr_pdf(path):
    doc = fitz.open(path)
    pages_text = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img)
        pages_text.append(txt)
    return "\n".join(pages_text)

def extract_text(path, prefer_ocr=True):
    text = extract_text_pdfplumber(path)
    if len(text) < 50:
        text = extract_text_pymupdf(path)
    if len(text) < 50 or prefer_ocr:
        text = ocr_pdf(path)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()
