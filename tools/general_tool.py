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
You are a concise AI assistant.

Answer the following question in **3–5 short bullet points**, with clarity and minimal verbosity.

Avoid over-explaining. Think step-by-step, but keep it brief.

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
