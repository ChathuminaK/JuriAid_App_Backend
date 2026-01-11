from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama

app = FastAPI()

llm = ChatOllama(
    model="mistral",
    temperature=0.3
)

class QuestionRequest(BaseModel):
    case_text: str
    law: str
    cases: str

@app.post("/generate-questions")
def generate_questions(data: QuestionRequest):

    prompt = f"""
Generate ONLY legal questions.
Number them.
One question per line.

Case:
{data.case_text}

Law:
{data.law}

Past Case:
{data.cases}
"""

    response = llm.invoke(prompt)

    return {"questions": response.content}
