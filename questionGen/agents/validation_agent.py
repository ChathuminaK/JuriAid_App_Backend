from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral")

def validation_agent(questions):
    response = llm.invoke(
        "Review and refine these legal questions:\n" + questions
    )
    return response.content
