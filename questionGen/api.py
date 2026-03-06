import sys
from pathlib import Path

# Add the parent directory of questionGen to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .orchestrator import run_question_generation

app = FastAPI()


class QuestionRequest(BaseModel):
    case_text: str
    law: str
    cases: str


@app.post("/generate-questions")
def generate_questions(data: QuestionRequest):
    try:
        result = run_question_generation(
            case_text=data.case_text,
            law_text=data.law,
            past_cases=data.cases
        )
        return {"questions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}