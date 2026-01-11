import os, json, httpx, time  # Added 'time' here
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai

# Import settings at module level
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

# Configure Gemini for tool functions
genai.configure(api_key=settings.GEMINI_API_KEY)

# ==================== FAMILY LAW TOOLS ====================

def retrieve_family_law_statutes(case_context: str = "") -> List[Dict[str, Any]]:
    """
    Retrieve relevant Sri Lankan family law statutes with detailed information.
    Returns structured data matching the React Native screen format.
    """
    # Static knowledge base of Sri Lankan family law
    statutes_db = [
        {
            "id": "1",
            "code": "Divorce Act (Chapter 51) - Section 2",
            "title": "Grounds for Divorce - Cruelty",
            "description": "Defines cruelty as grounds for divorce under Sri Lankan matrimonial law",
            "applicability": "High",
            "keyPoints": [
                "Physical violence causing bodily harm",
                "Mental cruelty causing reasonable apprehension of harm",
                "Persistent pattern of conduct making cohabitation impossible",
                "Courts assess cruelty based on conduct, not subjective feelings"
            ]
        },
        {
            "id": "2",
            "code": "Divorce Act - Section 3",
            "title": "Malicious Abandonment",
            "description": "Addresses abandonment as grounds for divorce without reasonable cause",
            "applicability": "High",
            "keyPoints": [
                "Intentional separation for continuous period of one year or more",
                "Abandonment must be without reasonable cause",
                "Failure to provide maintenance during abandonment",
                "No intention to return to matrimonial home"
            ]
        },
        {
            "id": "3",
            "code": "Maintenance Act (Chapter 37)",
            "title": "Maintenance for Wife and Children",
            "description": "Governs financial support obligations for dependents",
            "applicability": "High",
            "keyPoints": [
                "Court considers earning capacity of both parties",
                "Standard of living during marriage",
                "Needs of children including education and healthcare",
                "Court may order interim and permanent maintenance"
            ]
        },
        {
            "id": "4",
            "code": "Kandyan Marriage and Divorce Act (Chapter 113)",
            "title": "Kandyan Law Matrimonial Provisions",
            "description": "Special provisions for marriages under Kandyan law",
            "applicability": "Medium",
            "keyPoints": [
                "Applies to Kandyan Sinhalese community",
                "Different property division rules",
                "Custody based on best interests of child",
                "Matrimonial home ownership considerations"
            ]
        },
        {
            "id": "5",
            "code": "Marriage Registration Ordinance (Chapter 112)",
            "title": "Marriage Registration Requirements",
            "description": "Legal requirements for valid marriage registration in Sri Lanka",
            "applicability": "Medium",
            "keyPoints": [
                "Registration required for legal validity",
                "Jurisdiction and venue requirements",
                "Documentation and witness requirements",
                "Certificate of marriage as evidence"
            ]
        }
    ]
    
    # Filter by context if provided (simple keyword matching)
    if case_context:
        context_lower = case_context.lower()
        keywords = {
            "custody": ["custody", "child", "children", "minor"],
            "maintenance": ["maintenance", "support", "financial", "alimony"],
            "cruelty": ["cruelty", "violence", "abuse", "assault"],
            "abandonment": ["abandon", "desert", "separation", "left"],
            "kandyan": ["kandyan", "customary"]
        }
        
        # Score each statute based on relevance
        for statute in statutes_db:
            score = 0
            for category, words in keywords.items():
                if any(word in context_lower for word in words):
                    if category in statute["title"].lower() or category in statute["description"].lower():
                        score += 1
            statute["_relevance_score"] = score
        
        # Sort by relevance and return top results
        statutes_db.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
        results = statutes_db[:3]  # Top 3 most relevant
        
        # Clean up temporary scoring field
        for s in results:
            s.pop("_relevance_score", None)
        
        return results
    
    return statutes_db[:3]  # Return top 3 by default


def find_divorce_case_precedents(case_context: str = "") -> List[Dict[str, Any]]:
    """
    Find relevant Sri Lankan divorce case precedents.
    Returns structured data matching the React Native screen format.
    """
    precedents_db = [
        {
            "id": "1",
            "title": "Fernando v Fernando (2018)",
            "citation": "2 SLR 145",
            "court": "Supreme Court of Sri Lanka",
            "year": "2018",
            "relevance": "High",
            "summary": "Landmark case interpreting the definition of cruelty under the Divorce Act, particularly mental cruelty and its impact on matrimonial relationship",
            "keyHolding": "Mental cruelty includes conduct that causes reasonable apprehension of danger to life, limb or health, whether mental or physical. The test is objective - whether a reasonable person would find the conduct unbearable.",
            "applicablePoints": [
                "Definition and scope of mental cruelty",
                "Burden of proof lies on petitioner to establish cruelty",
                "Pattern of conduct to be considered, not isolated incidents",
                "Court must assess impact on petitioner's wellbeing"
            ],
            "outcome": "Petitioner granted divorce - mental cruelty established through pattern of degrading treatment"
        },
        {
            "id": "2",
            "title": "Perera v Perera (2016)",
            "citation": "1 SLR 298",
            "court": "Court of Appeal, Sri Lanka",
            "year": "2016",
            "relevance": "High",
            "summary": "Addressed the requirements for proving malicious abandonment and its interaction with maintenance obligations",
            "keyHolding": "Abandonment must be willful and without just cause. Mere physical separation does not constitute abandonment if there is reasonable cause. Failure to provide maintenance strengthens the case for abandonment.",
            "applicablePoints": [
                "Elements required to prove abandonment",
                "Relevance of financial support during separation",
                "Intention to permanently desert must be demonstrated",
                "Communication attempts and reconciliation efforts considered"
            ],
            "outcome": "Divorce granted on grounds of abandonment - husband left without cause and failed to maintain wife"
        },
        {
            "id": "3",
            "title": "Silva v Silva (2020)",
            "citation": "3 SLR 67",
            "court": "High Court of Colombo",
            "year": "2020",
            "relevance": "Medium",
            "summary": "Established principles for determining maintenance amounts in divorce proceedings with custody considerations",
            "keyHolding": "Maintenance must be calculated considering the standard of living during marriage, earning capacity of both parties, and the best interests of children. Court may order interim maintenance pending final decree.",
            "applicablePoints": [
                "Factors for calculating maintenance payments",
                "Interim vs. permanent maintenance orders",
                "Child custody and its impact on maintenance",
                "Enforcement mechanisms for maintenance orders"
            ],
            "outcome": "Monthly maintenance of Rs. 45,000 awarded to wife with child custody"
        },
        {
            "id": "4",
            "title": "De Silva v De Silva (2019)",
            "citation": "2 SLR 234",
            "court": "District Court of Colombo",
            "year": "2019",
            "relevance": "Medium",
            "summary": "Child custody determination based on best interests principle and parental fitness assessment",
            "keyHolding": "In custody disputes, the paramount consideration is the welfare and best interests of the child. Court considers age, sex, character, wishes of child (if of sufficient age), parental conduct and fitness.",
            "applicablePoints": [
                "Best interests of child as paramount consideration",
                "Factors affecting custody determination",
                "Joint custody vs. sole custody considerations",
                "Visitation and access rights for non-custodial parent"
            ],
            "outcome": "Mother granted custody with father's visitation rights - child's preference and stability considered"
        }
    ]
    
    # Filter by context
    if case_context:
        context_lower = case_context.lower()
        for precedent in precedents_db:
            score = 0
            if any(word in context_lower for word in ["cruel", "abuse", "violence"]):
                if "cruelty" in precedent["title"].lower() or "cruelty" in precedent["summary"].lower():
                    score += 2
            if any(word in context_lower for word in ["abandon", "desert", "left"]):
                if "abandon" in precedent["title"].lower() or "abandon" in precedent["summary"].lower():
                    score += 2
            if any(word in context_lower for word in ["custody", "child", "children"]):
                if "custody" in precedent["summary"].lower():
                    score += 1
            if any(word in context_lower for word in ["maintenance", "support", "financial"]):
                if "maintenance" in precedent["summary"].lower():
                    score += 1
            
            precedent["_score"] = score
        
        precedents_db.sort(key=lambda x: x.get("_score", 0), reverse=True)
        results = precedents_db[:3]
        for p in results:
            p.pop("_score", None)
        return results
    
    return precedents_db[:3]


def generate_family_law_client_questions(
    case_text: str = "",
    statutes: Optional[List[Dict[str, Any]]] = None,
    precedents: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate intelligent client questions using Gemini AI based on:
    - Case text
    - Identified statutes
    - Relevant precedents
    - Case summary
    """
    print(f"[DEBUG] generate_family_law_client_questions called")
    print(f"[DEBUG] - case_text length: {len(case_text) if case_text else 0}")
    print(f"[DEBUG] - statutes provided: {statutes is not None} ({len(statutes) if statutes else 0} items)")
    print(f"[DEBUG] - precedents provided: {precedents is not None} ({len(precedents) if precedents else 0} items)")
    print(f"[DEBUG] - summary provided: {summary is not None}")
    
    # Check if we have sufficient context
    if not case_text or len(case_text.strip()) < 50:
        print("[DEBUG] Insufficient case text, returning fallback questions")
        return _get_fallback_questions()
    
    # If no additional context provided, return fallback
    if not statutes and not precedents and not summary:
        print("[DEBUG] No additional context (statutes/precedents/summary), returning fallback")
        return _get_fallback_questions()
    
    # Build comprehensive context for AI
    context_parts = [f"CASE CONTEXT:\n{case_text[:2000]}"]
    
    if summary:
        context_parts.append(f"\n\nCASE SUMMARY:\n{json.dumps(summary, indent=2)}")
    
    if statutes:
        statutes_text = "\n\nRELEVANT STATUTES:\n"
        for statute in statutes:
            statutes_text += f"- {statute.get('code', '')}: {statute.get('title', '')}\n"
            for point in statute.get('keyPoints', [])[:3]:
                statutes_text += f"  • {point}\n"
        context_parts.append(statutes_text)
    
    if precedents:
        precedents_text = "\n\nRELEVANT CASE PRECEDENTS:\n"
        for case in precedents:
            precedents_text += f"- {case.get('title', '')}: {case.get('keyHolding', '')[:200]}\n"
        context_parts.append(precedents_text)
    
    full_context = "".join(context_parts)
    
    print(f"[DEBUG] Full context length: {len(full_context)}")
    
    # Use Gemini to generate contextual questions with retry logic
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Use stable model instead of experimental
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            
            prompt = f"""You are a Sri Lankan family law attorney. Based on the case information, statutes, and precedents below, generate 10-15 detailed questions to ask the client during intake interview.

{full_context}

Generate questions that:
1. Clarify facts needed to prove the legal grounds (based on the statutes mentioned)
2. Gather evidence similar to successful precedent cases referenced above
3. Assess financial situation for maintenance calculations
4. Understand child custody considerations
5. Identify potential witnesses or documentation
6. Address gaps in the current case information

IMPORTANT: Reference the specific statutes and cases mentioned above in your questions.

Return ONLY a JSON array of question strings (no explanations, no markdown):
["Question 1?", "Question 2?", ...]
"""
            
            print(f"[DEBUG] Calling Gemini API (attempt {attempt + 1}/{max_retries})...")
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            print(f"[DEBUG] Gemini response length: {len(raw_text)}")
            
            # Extract JSON array
            start_idx = raw_text.find("[")
            end_idx = raw_text.rfind("]") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = raw_text[start_idx:end_idx]
                questions = json.loads(json_text)
                print(f"[DEBUG] Successfully generated {len(questions)} AI questions")
                return questions[:15]  # Limit to 15 questions
            else:
                print("[DEBUG] Could not find JSON array in response")
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Error generating questions with Gemini (attempt {attempt + 1}): {error_msg}")
            
            # Check if it's a quota error
            if "quota" in error_msg.lower() or "429" in error_msg:
                print(f"[ERROR] Quota exceeded. Falling back to static questions.")
                break  # Don't retry on quota errors
            
            # Wait before retry for other errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"[DEBUG] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                import traceback
                traceback.print_exc()
    
    # Fallback if AI generation fails
    print("[DEBUG] Falling back to context-enhanced static questions")
    return _get_context_enhanced_questions(statutes, precedents, summary)


def _get_context_enhanced_questions(
    statutes: Optional[List[Dict[str, Any]]] = None,
    precedents: Optional[List[Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Return enhanced fallback questions that reference the provided context.
    This provides some intelligence even when AI quota is exceeded.
    """
    questions = []
    
    # Basic questions
    questions.extend([
        "What is the exact date of marriage and where was it registered?",
        "What are the specific grounds for seeking divorce?",
    ])
    
    # Add statute-specific questions
    if statutes:
        for statute in statutes:
            title_lower = statute.get('title', '').lower()
            if 'cruelty' in title_lower:
                questions.append(
                    f"Based on the {statute.get('code', 'Divorce Act')} regarding cruelty, "
                    "can you provide specific dates and descriptions of each incident?"
                )
            elif 'abandon' in title_lower:
                questions.append(
                    f"Under {statute.get('code', 'the law')} on abandonment, "
                    "when did the separation occur and has there been any communication since?"
                )
            elif 'maintenance' in title_lower:
                questions.append(
                    f"For {statute.get('code', 'maintenance purposes')}, "
                    "what is your current monthly income and financial obligations?"
                )
    
    # Add precedent-specific questions
    if precedents:
        for precedent in precedents[:2]:  # Top 2 precedents
            title = precedent.get('title', '')
            if 'custody' in precedent.get('summary', '').lower():
                questions.append(
                    f"Similar to {title}, what custody arrangement do you believe is in "
                    "the best interests of the children?"
                )
    
    # Generic questions
    questions.extend([
        "Are there any minor children from this marriage? Please provide their names and ages.",
        "Do you have documentation such as medical reports, police complaints, or witness statements?",
        "Has the spouse provided any financial support since separation?",
        "What is your spouse's estimated monthly income?",
        "What property was acquired during the marriage?",
        "Have you attempted mediation or counseling?",
        "Are there any ongoing court cases involving you and your spouse?",
        "What is your preferred visitation arrangement?"
    ])
    
    return questions[:15]


def _get_fallback_questions() -> List[str]:
    """Return basic fallback questions when no context available"""
    return [
        "What is the exact date of marriage and where was it registered?",
        "What are the specific grounds for seeking divorce (e.g., cruelty, abandonment, adultery)?",
        "Are there any minor children from this marriage? Please provide their names, ages, and current living arrangements.",
        "What custody arrangement are you seeking and why?",
        "Please describe any incidents of physical violence or mental cruelty with dates and details.",
        "When did the separation occur and who left the matrimonial home?",
        "Has the spouse provided any financial support since separation? If yes, how much and how regularly?",
        "What is your current monthly income, employment status, and financial obligations?",
        "What is your spouse's estimated monthly income and employment details?",
        "Do you have documentation such as medical reports, police complaints, photographs, or witness statements?",
        "What immovable and movable property was acquired during the marriage?",
        "Have you attempted mediation, counseling, or reconciliation? If yes, when and what was the outcome?",
        "Are there any ongoing court cases or legal proceedings involving you and your spouse?",
        "What is your preferred visitation arrangement if custody is not granted to you?"
    ]


def summarize_case(case_text: str) -> Dict[str, Any]:
    """
    Generate comprehensive case summary using Gemini AI.
    Returns structured summary matching React Native format.
    """
    try:
        # Use stable model
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        prompt = f"""You are a Sri Lankan legal analyst. Analyze this case and provide a structured summary.

Case Text:
{case_text[:3000]}

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "title": "Brief case title (e.g., 'Divorce Case - Cruelty and Abandonment')",
  "type": "Case type (e.g., 'Matrimonial Law', 'Contract Law')",
  "description": "2-3 sentence comprehensive case description",
  "parties": {{
    "plaintiff": "Plaintiff/Petitioner name or description",
    "defendant": "Defendant/Respondent name or description"
  }},
  "dateFilied": "Filing date if mentioned, otherwise 'Not specified'",
  "damagesClaimed": "Relief sought or damages claimed",
  "recommendedActions": [
    {{
      "id": "1",
      "action": "Action title",
      "priority": "High",
      "description": "Detailed action description"
    }}
  ]
}}
"""
        
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Extract JSON
        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_text = raw_text[start_idx:end_idx]
            summary = json.loads(json_text)
            return summary
    
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Error in AI summarization: {error_msg}")
        if "quota" in error_msg.lower() or "429" in error_msg:
            print("[ERROR] Quota exceeded for summarization, using fallback")
    
    # Fallback summary
    return {
        "title": "Legal Case Analysis",
        "type": "General Legal Matter",
        "description": case_text[:500] + "..." if len(case_text) > 500 else case_text,
        "parties": {
            "plaintiff": "Petitioner",
            "defendant": "Respondent"
        },
        "dateFilied": "Not specified",
        "damagesClaimed": "To be determined",
        "recommendedActions": [
            {
                "id": "1",
                "action": "Review Case Documents",
                "priority": "High",
                "description": "Collect and review all relevant case documents and evidence"
            },
            {
                "id": "2",
                "action": "Gather Evidence",
                "priority": "High",
                "description": "Compile medical records, police reports, and witness statements"
            },
            {
                "id": "3",
                "action": "Financial Assessment",
                "priority": "Medium",
                "description": "Prepare detailed financial affidavit for maintenance determination"
            }
        ]
    }


def update_knowledge_base(case_text: str, metadata: dict, kb_path: str):
    """Save case to knowledge base"""
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {"entries": []}
    
    entry = {
        "id": len(kb["entries"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "case_type": metadata.get("case_type"),
        "length": len(case_text),
        "snippet": case_text[:400],
        "tags": metadata.get("tags", [])
    }
    kb["entries"].append(entry)
    
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    
    return {"status": "updated", "total_entries": len(kb["entries"])}


# ==================== INTEGRATION WITH PAST CASE RETRIEVAL ====================

async def query_past_cases(case_text: str, topk: int = 5) -> List[Dict[str, Any]]:
    """
    Query the past_case_retrieval service for similar cases.
    This integrates with your past case retrieval microservice.
    """
    try:
        # Create temporary file for upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(case_text)
            tmp_path = tmp.name
        
        # Call past case retrieval API
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(tmp_path, 'rb') as f:
                files = {'file': ('case.txt', f, 'text/plain')}
                params = {'topk': topk}
                response = await client.post(
                    "http://localhost:8002/upload_and_search",
                    files=files,
                    params=params
                )
                
                if response.status_code == 200:
                    return response.json()
        
        os.unlink(tmp_path)
    
    except Exception as e:
        print(f"Error querying past cases: {e}")
        return []