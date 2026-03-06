from questionGen.agents.reasoning_agent import generate_questions as reasoning_agent
from questionGen.agents.question_agent import question_agent
from questionGen.agents.validation_agent import validation_agent

def run_question_generation(case_text: str, law_text: str, past_cases: str) -> str:

    # Step 1 - Identify core legal issues
    reasoning = reasoning_agent(
        case_text=case_text,
        law_text=law_text,
        past_cases=past_cases
    )

    # Step 2 - Generate structured Findings + Admissions
    question_result = question_agent(
        case=case_text,
        reasoning=reasoning,
        law=law_text,
        cases=past_cases
    )

    # Safely extract string from question_agent (returns dict or str)
    if isinstance(question_result, dict):
        formatted = question_result.get("formatted", str(question_result))
    else:
        formatted = str(question_result)

    # Step 3 - Validate and refine
    validation_result = validation_agent(formatted)

    # Safely extract string from validation_agent (returns dict or str)
    if isinstance(validation_result, dict):
        return validation_result.get("validated", formatted)
    else:
        return str(validation_result)