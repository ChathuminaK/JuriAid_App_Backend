import fitz
import pdfplumber
import pytesseract
from PIL import Image
import io
import re

# Update this path to your local Tesseract installation
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
    """Fallback OCR function for scanned images/PDFs"""
    try:
        doc = fitz.open(path)
        pages_text = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(dpi=300) 
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img)
            pages_text.append(txt)
        return "\n".join(pages_text)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def extract_text(path, prefer_ocr=False): # Changed default to False
    # 1. Try PDFPlumber (Best for tables/formatted text)
    text = extract_text_pdfplumber(path)
    
    # 2. Try PyMuPDF if first attempt was empty
    if len(text.strip()) < 100:
        text = extract_text_pymupdf(path)
    
    # 3. Only OCR if the text is still missing or user explicitly wants it
    if len(text.strip()) < 100 or prefer_ocr:
        print(f"[DEBUG] Direct extraction failed for {path}. Starting OCR...")
        text = ocr_pdf(path)
    
    # Clean up whitespace
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()