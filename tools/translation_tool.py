from langchain_core.tools import tool
from tools.utils import call_llm

@tool
def translate(input: str) -> dict:
    """
    Uses an LLM to translate English <-> German, returning the translated text and model used.
    """
    prompt = f"""
Translate the following phrase between English and German.
Only return the translated phrase. Do NOT include any explanation, context, or formatting.

Phrase: {input}
"""
    try:
        response, model_used = call_llm(prompt)
        return {
            "response": response.strip(),
            "model": model_used
        }
    except Exception as e:
        return {
            "response": f"Error in translation: {e}",
            "model": "unknown"
        }
