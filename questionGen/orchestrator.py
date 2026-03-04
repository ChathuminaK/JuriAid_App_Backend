from agents.reasoning_agent import reasoning_agent

def run_question_generation(case_text: str, law_text: str, past_cases: str) -> str:
    """
    Orchestrates the question generation process.
    """

    questions = reasoning_agent(
        case_text=case_text,
        law_text=law_text,
        past_cases=past_cases
    )

    return questions
