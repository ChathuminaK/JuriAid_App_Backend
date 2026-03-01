from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# --- Similar Past Cases ---
class SimilarCase(BaseModel):
    case_id: str
    case_name: Optional[str] = None
    score: Optional[float] = None
    complaint: Optional[str] = None
    defense: Optional[str] = None


# --- Relevant Laws ---
class RelevantLaw(BaseModel):
    act_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    relevance_score: Optional[float] = None
    content: Optional[str] = None


# --- Generated Questions ---
class GeneratedQuestion(BaseModel):
    question_id: int
    question: str


# --- Main Analysis Response ---
class CaseAnalysisResponse(BaseModel):
    analysis_id: str
    status: str = "completed"
    case_summary: str
    similar_cases: List[SimilarCase] = Field(default_factory=list)
    relevant_laws: List[RelevantLaw] = Field(default_factory=list)
    generated_questions: List[GeneratedQuestion] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processing_time_seconds: Optional[float] = None


# --- Save Case Response ---
class CaseSaveResponse(BaseModel):
    saved: bool
    case_id: Optional[str] = None
    message: str