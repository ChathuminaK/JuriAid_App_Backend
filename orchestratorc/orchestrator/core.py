import os
from datetime import datetime
from typing import Dict, Any

# --- Configuration ---
ORCHESTRATOR_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(ORCHESTRATOR_DIR, "uploads")
OUTPUTS_DIR = os.path.join(ORCHESTRATOR_DIR, "outputs")

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

def read_input_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError("Unsupported file type. Use .pdf or .txt")

def analyze_case_content(case_text: str) -> Dict[str, Any]:
    tl = case_text.lower()
    if any(k in tl for k in ["divorce", "marriage", "custody", "maintenance"]):
        case_type = "Family Law"
    elif any(k in tl for k in ["contract", "agreement", "breach"]):
        case_type = "Contract Law"
    else:
        case_type = "General Legal Matter"
    return {"case_type": case_type, "length": len(case_text)}

def generate_legal_summary(case_text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "executive_summary": f"This case involves a {analysis['case_type']}. "
                             f"Approx. {analysis['length']} characters processed.",
        "recommendations": [
            "Collect all supporting documents",
            "Clarify facts and timeline",
            "Identify governing statutes"
        ],
        "next_steps": [
            "Review extracted content",
            "Refine issues and questions",
            "Prepare for client discussion"
        ],
    }

def create_detailed_report(case_text: str, analysis: Dict[str, Any],
                           summary: Dict[str, Any], output_path: str,
                           prompt: str | None = None) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\nJuriAid: Legal Case Analysis Report\n" + "="*70 + "\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Case Type: {analysis['case_type']}\n")
        if prompt:
            f.write(f"User Prompt: {prompt}\n")
        f.write("\nEXECUTIVE SUMMARY\n------------------\n")
        f.write(summary["executive_summary"] + "\n\n")
        f.write("RECOMMENDATIONS\n----------------\n")
        for i, rec in enumerate(summary["recommendations"], 1):
            f.write(f"{i}. {rec}\n")
        f.write("\nNEXT STEPS\n----------\n")
        for i, step in enumerate(summary["next_steps"], 1):
            f.write(f"{i}. {step}\n")
        f.write("\n")

def find_input_file() -> str | None:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    for name in os.listdir(UPLOADS_DIR):
        if name.lower().endswith((".pdf", ".txt")):
            return os.path.join(UPLOADS_DIR, name)
    return None

def process_single_file(input_filepath: str, prompt: str | None = None) -> Dict[str, Any]:
    try:
        case_text = read_input_file(input_filepath)
        analysis = analyze_case_content(case_text)
        base = os.path.splitext(os.path.basename(input_filepath))[0]
        out_name = f"{base}_analysis_{datetime.now():%Y%m%d_%H%M%S}.txt"
        out_path = os.path.join(OUTPUTS_DIR, out_name)
        create_detailed_report(case_text, analysis,
                               generate_legal_summary(case_text, analysis),
                               out_path, prompt=prompt)
        return {
            "success": True,
            "input_file": input_filepath,
            "output_file": out_path,
            "case_type": analysis["case_type"],
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e), "input_file": input_filepath}

def run_full_process():
    fp = find_input_file()
    if not fp:
        print(f"❌ No case file found in {UPLOADS_DIR}")
        return
    result = process_single_file(fp)
    if result.get("success"):
        print(f"✅ Output: {result['output_file']}")
    else:
        print(f"❌ Error: {result.get('error')}")

