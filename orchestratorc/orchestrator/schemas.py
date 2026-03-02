from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ---------- Internal models ----------

class UserIntent(BaseModel):
    """LLM-detected intent from user prompt."""
    should_save_case: bool = False
    analysis_focus: str = "general legal analysis"
    key_topics: list[str] = Field(default_factory=list)


class ServiceResult(BaseModel):
    """Wrapper for any downstream service call result."""
    success: bool = True
    data: Optional[dict] = None
    error: Optional[str] = None


# ---------- Response models ----------

class SimilarCase(BaseModel):
    case_id: str = ""
    case_name: str = ""
    score: float = 0.0
    summary: str = ""
    reason: str = ""
    complaint: str = ""
    defense: str = ""


class RelevantLaw(BaseModel):
    act_id: str = ""
    title: str = ""
    section: str = ""
    section_title: str = ""
    relevance_score: float = 0.0
    content: str = ""


class GeneratedQuestion(BaseModel):
    question_id: int
    question: str


class AnalysisMetadata(BaseModel):
    filename: str
    file_size_mb: float
    text_length: int
    user_id: int
    user_prompt: str
    saved_for_reference: bool = False


class AnalysisResponse(BaseModel):
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "completed"
    case_summary: str = ""
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    relevant_laws: list[RelevantLaw] = Field(default_factory=list)
    generated_questions: list[GeneratedQuestion] = Field(default_factory=list)
    metadata: Optional[AnalysisMetadata] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    processing_time_seconds: float = 0.0


class CaseSaveResponse(BaseModel):
    saved: bool = True
    case_id: str = ""
    message: str = "Case saved and indexed for future reference"


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "JuriAid Orchestrator"
    version: str = "2.0.0"