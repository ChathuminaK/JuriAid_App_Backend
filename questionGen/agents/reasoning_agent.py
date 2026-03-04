from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

# Load local LLM
llm = ChatOllama(model="mistral")

def generate_questions(case_text: str, law_text: str, past_cases: str) -> str:
    prompt = PromptTemplate(
        input_variables=["case", "law", "cases"],
        template="""
You are an AI legal assistant.

Your task is to generate ONLY clear, legally relevant QUESTIONS.

Rules:
- Output ONLY questions
- Use numbered list
- One question per line
- No explanations
- No headings
- No answers

Case Summary:
{case}

Relevant Law:
{law}

Relevant Past Cases:
{cases}

Generate 5–8 legal questions.
"""
    )

    final_prompt = prompt.format(
        case=case_text,
        law=law_text,
        cases=past_cases
    )

    response = llm.invoke(final_prompt)
    return response.content
