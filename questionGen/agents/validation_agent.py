import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))

def validation_agent(questions):
    prompt = (
        "You are a Sri Lankan litigation attorney reviewing court documents.\n"
        "Review the FINDINGS OF FACT and ADMISSIONS below and fix any grammar, clarity, or legal accuracy issues.\n\n"
        "STRICT RULES:\n"
        "- Return ONLY the corrected FINDINGS OF FACT and ADMISSIONS sections\n"
        "- Do NOT add any notes, explanations, summaries, or 'Refinements' section\n"
        "- Keep the exact same structure and numbering format\n"
        "- Do NOT remove or add any items unless absolutely necessary\n\n"
        + questions
    )
    response = llm.invoke(prompt)
    return response.content
