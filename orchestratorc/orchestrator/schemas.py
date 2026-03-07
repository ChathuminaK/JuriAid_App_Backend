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
    reason: str = ""
    judgment_preview: str = ""
    shared_issues: list[str] = Field(default_factory=list)
    breakdown: dict = Field(default_factory=dict)
    view_case_details: str = ""
    view_full_case_file: str = ""


class RelevantLaw(BaseModel):
    case_id: str = ""
    case_name: str = ""
    citation: str = ""
    topic: str = ""
    section_number: str = ""
    section_title: str = ""
    principle: list[str] = Field(default_factory=list)
    held: list[str] = Field(default_factory=list)
    facts: str = ""
    referenced_laws: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    court: str = ""
    amending_law: str = ""
    confidence_score: float = 0.0
    support_score: float = 0.0
    query_hits: int = 0
    detail_url: str = ""


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
    generated_questions: str = ""
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