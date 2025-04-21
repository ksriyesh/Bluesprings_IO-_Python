# assistant_agent/tools/general_tool.py

from langchain_core.tools import tool
from tools.utils import call_llm

@tool
def general_assistant(input: str) -> dict:
    """
    Uses an LLM to respond to general questions with step-by-step reasoning.
    Returns both response and model used.
    """
    prompt = f"""
You are a professional assistant who always thinks step-by-step.
Answer the following question thoroughly, with a logical explanation.

Question: {input}
"""
    try:
        response, model_used = call_llm(prompt)
        return {
            "response": response.strip(),
            "model": model_used
        }
    except Exception as e:
        return {
            "response": f"LLM error in general assistant: {e}",
            "model": "unknown"
        }
