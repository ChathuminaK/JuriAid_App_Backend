from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="mistral")

def question_agent(case, reasoning, law, cases):
    prompt = PromptTemplate.from_template("""
Generate legal questions based on the following.

CASE:
{case}

REASONING:
{reasoning}

LAW:
{law}

PAST CASES:
{cases}
""")

    final_prompt = prompt.format(
        case=case,
        reasoning=reasoning,
        law=law,
        cases=cases
    )

    response = llm.invoke(final_prompt)
    return response.content
