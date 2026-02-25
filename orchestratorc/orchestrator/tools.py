from __future__ import annotations

import json
import logging
from typing import Dict, Any

import httpx
from langchain_core.tools import tool

from config import settings

logger = logging.getLogger("juriaid.tools")

# ---------------------------------------------------------------------------
# Service URLs from environment/config
# ---------------------------------------------------------------------------
LAWSTAT_KG_URL = settings.LAWSTATKG_SERVICE_URL  # http://lawstatkg_service:8003
PAST_CASES_URL = settings.PAST_CASE_SERVICE_URL  # http://past_case_service:8002
QUESTION_GEN_URL = settings.QUESTIONGEN_SERVICE_URL  # http://questiongen_service:8004


# ---------------------------------------------------------------------------
# 1. Search Law Statutes (LawStatKG Integration)
# ---------------------------------------------------------------------------

@tool
async def search_law_statutes(query: str) -> str:
    """Search Sri Lankan statutes, Acts, and Sections using hybrid semantic + keyword search.
    
    This tool connects to the LawStatKG service to find relevant legal statutes
    based on the user's query. It performs hybrid search combining semantic 
    understanding with keyword matching.
    
    Args:
        query: Legal query or keywords to search for relevant statutes
        
    Returns:
        JSON string containing matching statute sections with metadata
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LAWSTAT_KG_URL}/api/search",
                json={"query": query, "top_k": 5}
            )
            response.raise_for_status()
            data = response.json()
            
            # Format results for LLM consumption
            if isinstance(data, dict) and "results" in data:
                formatted_results = []
                for result in data["results"]:
                    formatted_results.append({
                        "act": result.get("act_name", "Unknown"),
                        "section": result.get("section_number", "N/A"),
                        "text": result.get("section_text", ""),
                        "relevance": result.get("score", 0.0)
                    })
                return json.dumps(formatted_results, indent=2)
            return json.dumps(data, indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"LawStatKG search failed: {e}")
        return json.dumps({
            "error": "Statute search service temporarily unavailable",
            "fallback": "Please try searching manually in the Legal Database"
        })


@tool
async def get_statute_by_act(act_id: str) -> str:
    """Retrieve the full text of a specific Sri Lankan Act by its ID.
    
    Args:
        act_id: The Act identifier (e.g., "35_2010" for Act No. 35 of 2010)
        
    Returns:
        JSON string containing the full Act text and metadata
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{LAWSTAT_KG_URL}/api/act/{act_id}"
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"Get statute by Act failed: {e}")
        return json.dumps({
            "error": f"Could not retrieve Act {act_id}",
            "message": "Act may not exist or service is unavailable"
        })


# ---------------------------------------------------------------------------
# 2. Search Past Cases (Past Case Retrieval Integration)
# ---------------------------------------------------------------------------

@tool
async def search_past_cases(query: str, top_k: int = 5) -> str:
    """Search for relevant Sri Lankan court precedents using hybrid semantic + citation search.
    
    This tool connects to the Past Case Retrieval service which uses knowledge
    graphs and hybrid search to find the most relevant legal precedents.
    
    Args:
        query: Legal query describing the case facts or legal issues
        top_k: Number of similar cases to retrieve (default: 5)
        
    Returns:
        JSON string containing matching case precedents with metadata
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{PAST_CASES_URL}/search",
                json={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            data = response.json()
            
            # Format results for better LLM comprehension
            if isinstance(data, dict) and "results" in data:
                formatted_cases = []
                for case in data["results"]:
                    formatted_cases.append({
                        "case_name": case.get("case_name", "Unknown"),
                        "citation": case.get("citation", "N/A"),
                        "facts": case.get("facts", "")[:500],  # Truncate long facts
                        "holding": case.get("holding", ""),
                        "relevance_score": case.get("score", 0.0)
                    })
                return json.dumps(formatted_cases, indent=2)
            return json.dumps(data, indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"Past cases search failed: {e}")
        return json.dumps({
            "error": "Past cases search service temporarily unavailable",
            "fallback": "Consider manual research in case law databases"
        })


# ---------------------------------------------------------------------------
# 3. Generate Legal Questions (Question Generator Integration)
# ---------------------------------------------------------------------------

@tool
async def generate_legal_questions(
    case_text: str,
    law_context: str = "",
    past_cases_context: str = ""
) -> str:
    """Generate relevant legal questions for client intake or case preparation.
    
    This tool uses AI to generate contextually relevant questions that a lawyer
    should ask to gather complete information about a case.
    
    Args:
        case_text: The case document text or summary
        law_context: Optional relevant statutes/laws context
        past_cases_context: Optional relevant precedents context
        
    Returns:
        JSON string containing generated questions categorized by topic
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{QUESTION_GEN_URL}/generate_questions",
                json={
                    "case_text": case_text[:2000],  # Limit to avoid token overflow
                    "law_context": law_context[:1000],
                    "past_cases_context": past_cases_context[:1000]
                }
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"Question generation failed: {e}")
        return json.dumps({
            "error": "Question generation service temporarily unavailable",
            "fallback_questions": [
                "What are the key facts of your case?",
                "What is the complete timeline of events?",
                "Who are all the parties involved?",
                "What remedies are you seeking?",
                "Do you have any supporting documents?"
            ]
        })


# ---------------------------------------------------------------------------
# 4. Summarize Case
# ---------------------------------------------------------------------------

@tool
async def summarize_case(case_text: str) -> str:
    """Produce a structured JSON summary of a legal case document.
    
    Args:
        case_text: The full case document text
        
    Returns:
        JSON string with structured summary (parties, facts, issues, holdings, etc.)
    """
    try:
        # Use the question generation service's summarization endpoint
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{QUESTION_GEN_URL}/summarize",
                json={"case_text": case_text[:3000]}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"Case summarization failed: {e}")
        # Fallback: basic text extraction
        lines = [l.strip() for l in case_text.split('\n') if l.strip()][:10]
        return json.dumps({
            "error": "Summarization service temporarily unavailable",
            "summary": "Service error - manual review required",
            "first_lines": lines
        })


# ---------------------------------------------------------------------------
# 5. Update Knowledge Base (Save Case to Neo4j/Vector Store)
# ---------------------------------------------------------------------------

@tool
async def update_knowledge_base(case_text: str, case_type: str = "General") -> str:
    """Save a case document to the knowledge base for future retrieval.
    
    This adds the case to the Neo4j knowledge graph and vector indexes
    so it can be found in future searches.
    
    Args:
        case_text: The case document text to save
        case_type: Category/type of the case (e.g., "Family Law", "Contract")
        
    Returns:
        JSON string with save confirmation and case ID
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{PAST_CASES_URL}/add_case",
                json={
                    "case_text": case_text,
                    "case_type": case_type,
                    "metadata": {
                        "source": "user_upload",
                        "upload_date": "2025-01-01"  # Use actual date
                    }
                }
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
            
    except httpx.HTTPError as e:
        logger.error(f"Knowledge base update failed: {e}")
        return json.dumps({
            "status": "error",
            "message": "Could not save case to knowledge base",
            "error": str(e)
        })


# ---------------------------------------------------------------------------
# Export all tools for LangChain agent
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_law_statutes,
    get_statute_by_act,
    search_past_cases,
    generate_legal_questions,
    summarize_case,
    update_knowledge_base,
]