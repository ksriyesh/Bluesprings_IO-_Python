# nodes/splitter.py

from tools.utils import call_llm
from typing import List, Literal
from pydantic import BaseModel, ValidationError
import json
import re

class Task(BaseModel):
    tool: Literal["calculator", "translator", "general"]
    query: str

def extract_json_block(response: str) -> str:
    """Extract JSON array from mixed LLM response."""
    match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)
    return match.group(0) if match else None

def split_and_classify(user_input: str) -> List[dict]:
    prompt = f"""
You are an AI assistant that breaks down compound queries into subtasks.

Classify each subtask into one of:
- "calculator" (for math operations like add, multiply, etc.)
- "translator" (ONLY for English <-> German translations)
- "general" (everything else, including other languages like Hindi or French)

If a translation is requested in a language other than German, classify it as "general".

Return only a raw JSON list like:
[
  {{ "tool": "translator", "query": "Translate 'Hello' to German" }},
  {{ "tool": "calculator", "query": "Multiply 5 and 6" }}
]

User input: "{user_input}"
"""

    try:
        response, model_used = call_llm(prompt)

        json_block = extract_json_block(response)
        if not json_block:
            raise ValueError("No JSON array found in LLM response.")

        raw_data = json.loads(json_block)
        tasks = [Task(**task).dict() for task in raw_data]

        # Optional: Add model used to each task for traceability
        for task in tasks:
            task["model_used"] = model_used

        return tasks

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        return [{
            "tool": "general",
            "query": f"Unable to split query: {str(e)}",
            "model_used": "unknown"
        }]
