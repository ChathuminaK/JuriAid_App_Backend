import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))

def validation_agent(questions):
    response = llm.invoke(
        "Review and refine these legal questions:\n" + questions
    )
    return response.content
