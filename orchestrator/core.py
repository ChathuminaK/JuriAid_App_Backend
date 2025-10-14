import os
import time
import PyPDF2
from datetime import datetime

# --- Configuration ---
UPLOADS_DIR = os.path.join("uploads")
OUTPUTS_DIR = os.path.join("outputs")

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    print(f"📄 Extracting text from PDF: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"   - PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"   - Processing page {page_num}...")
                page_text = page.extract_text()
                text += page_text + "\n"
            
            print(f"   - ✅ Successfully extracted {len(text)} characters")
            return text
            
    except Exception as e:
        print(f"   - ❌ Error extracting PDF: {e}")
        return None

def read_input_file(filepath):
    """Reads the content of the uploaded case file (PDF or TXT)."""
    print(f"1. 📂 Reading input file from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Input file not found at {filepath}")
        return None
    
    # Check file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.txt':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   - ✅ Successfully read {len(content)} characters from text file")
                return content
        except Exception as e:
            print(f"❌ ERROR reading text file: {e}")
            return None
    else:
        print(f"❌ ERROR: Unsupported file type {ext}. Use .pdf or .txt")
        return None

def analyze_case_content(case_text):
    """
    Simple analysis of the case content to extract key information.
    """
    print("2. 🧠 Analyzing case content...")
    
    analysis = {
        'case_type': 'General Legal Matter',
        'key_issues': [],
        'parties_involved': [],
        'important_dates': [],
        'legal_areas': []
    }
    
    text_lower = case_text.lower()
    
    # Detect case type
    if any(word in text_lower for word in ["divorce", "custody", "alimony", "marriage", "matrimonial"]):
        analysis['case_type'] = "Family Law"
        analysis['legal_areas'].append("Family Law")
        
    elif any(word in text_lower for word in ["contract", "agreement", "breach", "damages", "obligation"]):
        analysis['case_type'] = "Contract Law"
        analysis['legal_areas'].append("Contract Law")
        
    elif any(word in text_lower for word in ["criminal", "theft", "assault", "murder", "fraud", "crime"]):
        analysis['case_type'] = "Criminal Law"
        analysis['legal_areas'].append("Criminal Law")
        
    elif any(word in text_lower for word in ["property", "land", "ownership", "title", "real estate"]):
        analysis['case_type'] = "Property Law"
        analysis['legal_areas'].append("Property Law")
        
    elif any(word in text_lower for word in ["employment", "termination", "salary", "workplace", "labor"]):
        analysis['case_type'] = "Employment Law"
        analysis['legal_areas'].append("Employment Law")
    
    # Extract potential issues
    issue_keywords = ["issue", "problem", "dispute", "conflict", "matter", "claim"]
    for keyword in issue_keywords:
        if keyword in text_lower:
            # Find sentences containing these keywords
            sentences = case_text.split('.')
            for sentence in sentences:
                if keyword in sentence.lower() and len(sentence.strip()) > 20:
                    analysis['key_issues'].append(sentence.strip())
    
    # Extract potential party names (simple heuristic)
    party_keywords = ["plaintiff", "defendant", "petitioner", "respondent", "appellant", "applicant"]
    for keyword in party_keywords:
        if keyword in text_lower:
            analysis['parties_involved'].append(f"Case involves {keyword}")
    
    print(f"   - 🎯 Case Type: {analysis['case_type']}")
    print(f"   - 📋 Legal Areas: {', '.join(analysis['legal_areas']) if analysis['legal_areas'] else 'General'}")
    print(f"   - ⚖️ Key Issues Found: {len(analysis['key_issues'])}")
    
    return analysis

def generate_legal_summary(case_text, analysis):
    """
    Generate a comprehensive legal summary based on the case content.
    """
    print("3. 📝 Generating legal summary...")
    
    summary = {
        'executive_summary': '',
        'legal_analysis': '',
        'recommendations': [],
        'next_steps': []
    }
    
    # Executive Summary
    text_preview = case_text[:500] + "..." if len(case_text) > 500 else case_text
    summary['executive_summary'] = f"This case involves a {analysis['case_type'].lower()} matter. " + \
                                 f"The case content contains approximately {len(case_text)} characters of legal text."
    
    # Legal Analysis based on case type
    if analysis['case_type'] == "Family Law":
        summary['legal_analysis'] = """
        Family law matters in Sri Lanka are governed by multiple personal laws depending on the parties involved:
        - General Law (Roman-Dutch Law)
        - Kandyan Law 
        - Muslim Law
        - Thesawalamai Law
        
        Key considerations include matrimonial property, custody arrangements, and maintenance obligations.
        """
        summary['recommendations'] = [
            "Determine which personal law applies to the parties",
            "Gather all relevant matrimonial documents",
            "Consider mediation before litigation",
            "Assess financial circumstances for maintenance calculations"
        ]
        
    elif analysis['case_type'] == "Contract Law":
        summary['legal_analysis'] = """
        Contract law in Sri Lanka follows Roman-Dutch principles with statutory modifications.
        Key legislation includes the Sale of Goods Ordinance and various commercial statutes.
        
        Essential elements to establish: offer, acceptance, consideration, intention to create legal relations.
        """
        summary['recommendations'] = [
            "Review all contract documents thoroughly",
            "Identify specific breached terms",
            "Calculate actual damages suffered",
            "Consider alternative dispute resolution"
        ]
        
    elif analysis['case_type'] == "Property Law":
        summary['legal_analysis'] = """
        Property law in Sri Lanka involves complex title systems and registration requirements.
        Key legislation includes the Registration of Documents Ordinance and Land Development Ordinance.
        
        Title verification and proper documentation are crucial for property transactions.
        """
        summary['recommendations'] = [
            "Conduct thorough title search",
            "Verify all registration documents",
            "Check for encumbrances or liens",
            "Ensure proper survey and boundaries"
        ]
        
    else:
        summary['legal_analysis'] = """
        This legal matter requires careful analysis under Sri Lankan law.
        The legal system combines Roman-Dutch law, English law, and local statutes.
        
        Proper legal research and case law analysis will be essential.
        """
        summary['recommendations'] = [
            "Conduct comprehensive legal research",
            "Review relevant case precedents",
            "Identify applicable statutes",
            "Consider all available remedies"
        ]
    
    # Common next steps
    summary['next_steps'] = [
        "Gather all relevant documentation",
        "Prepare detailed statement of facts",
        "Research applicable legal precedents",
        "Consider time limitations and procedural requirements",
        "Evaluate strength of legal position"
    ]
    
    return summary

def create_detailed_report(case_text, analysis, summary, output_path):
    """
    Create a comprehensive legal analysis report.
    """
    print(f"4. 📄 Creating detailed report at: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("           🏛️ JuriAid: Legal Case Analysis Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Case Type: {analysis['case_type']}\n\n")
        
        # Executive Summary
        f.write("📋 EXECUTIVE SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"{summary['executive_summary']}\n\n")
        
        # Case Content Preview
        f.write("📄 CASE CONTENT PREVIEW\n")
        f.write("-" * 25 + "\n")
        preview = case_text[:800] + "...\n" if len(case_text) > 800 else case_text + "\n"
        f.write(preview + "\n")
        
        # Legal Analysis
        f.write("⚖️ LEGAL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(summary['legal_analysis'] + "\n\n")
        
        # Key Issues
        if analysis['key_issues']:
            f.write("🎯 IDENTIFIED ISSUES\n")
            f.write("-" * 20 + "\n")
            for i, issue in enumerate(analysis['key_issues'][:5], 1):  # Limit to 5 issues
                f.write(f"{i}. {issue}\n")
            f.write("\n")
        
        # Recommendations
        f.write("💡 RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        for i, rec in enumerate(summary['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")
        
        # Next Steps
        f.write("📅 SUGGESTED NEXT STEPS\n")
        f.write("-" * 25 + "\n")
        for i, step in enumerate(summary['next_steps'], 1):
            f.write(f"{i}. {step}\n")
        f.write("\n")
        
        # Footer
        f.write("="*70 + "\n")
        f.write("This report was generated by JuriAid AI Legal Assistant\n")
        f.write("For educational and preliminary analysis purposes only.\n")
        f.write("Consult with a qualified legal professional for official advice.\n")
        f.write("="*70 + "\n")
    
    print("   - ✅ Report saved successfully!")

def find_input_file():
    """Find the input file in uploads directory."""
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
        
    # Look for any PDF or TXT file in uploads
    for filename in os.listdir(UPLOADS_DIR):
        if filename.lower().endswith(('.pdf', '.txt')):
            return os.path.join(UPLOADS_DIR, filename)
    
    return None

def run_full_process():
    """The main function to run the entire orchestration process."""
    print("\n🚀 JuriAid Case Analysis Started")
    print("="*50)
    
    # Find input file
    input_filepath = find_input_file()
    
    if not input_filepath:
        print(f"❌ No case file found!")
        print(f"📁 Please place your file in: {UPLOADS_DIR}/")
        print(f"   Supported formats: .pdf, .txt")
        print(f"\nExample:")
        print(f"   {UPLOADS_DIR}/my_case.pdf")
        print(f"   {UPLOADS_DIR}/legal_document.txt")
        return
    
    print(f"📂 Found input file: {input_filepath}")
    
    # Process the case file
    case_content = read_input_file(input_filepath)
    
    if case_content:
        analysis = analyze_case_content(case_content)
        summary = generate_legal_summary(case_content, analysis)
        
        # Generate output filename based on input
        input_name = os.path.splitext(os.path.basename(input_filepath))[0]
        output_filename = f"{input_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_filepath = os.path.join(OUTPUTS_DIR, output_filename)
        
        create_detailed_report(case_content, analysis, summary, output_filepath)
        
        print(f"\n🎉 SUCCESS! Your legal analysis is ready:")
        print(f"📁 Input:  {input_filepath}")
        print(f"📄 Output: {output_filepath}")
        print(f"📊 Case Type: {analysis['case_type']}")
        print(f"📝 Report Length: {os.path.getsize(output_filepath)} bytes")
        
    else:
        print("❌ Failed to process input file.")