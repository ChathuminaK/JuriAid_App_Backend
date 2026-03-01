from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from datetime import datetime


class SimilarCase(BaseModel):
    case_id: Optional[str] = None
    case_name: Optional[str] = None
    score: Optional[float] = None
    complaint: Optional[str] = None
    defense: Optional[str] = None


class RelevantLaw(BaseModel):
    act_id: Optional[str] = None
    title: Optional[str] = None
    section: Optional[str] = None
    relevance_score: Optional[float] = None
    content: Optional[str] = None


class GeneratedQuestion(BaseModel):
    question_id: int
    question: str


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


class CaseSaveResponse(BaseModel):
    saved: bool
    case_id: Optional[str] = None
    message: str