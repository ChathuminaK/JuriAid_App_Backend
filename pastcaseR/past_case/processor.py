# backend/processor.py
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    return text.replace("\r", "\n").strip()

def split_into_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
