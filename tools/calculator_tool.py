# tools/calculator_tool.py
from langchain_core.tools import tool
from tools.utils import call_llm

@tool
def calculator(input: str) -> str:
    """
    Uses an LLM to evaluate math expressions. Return only the final answer.
    """
    prompt = f"""
Solve the following math problem and return ONLY the final numeric answer:
{input}
"""
    try:
        response, model_used = call_llm(prompt)
        return {
            "response": response.strip(),
            "model": model_used
        }
    except Exception as e:
        return f"Error in calculation: {e}"

