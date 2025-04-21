import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

# Fallback models in priority order
FALLBACK_MODELS = [
    'llama3-70b-8192',
    'llama-3-70b-versatile',
    'gemma2-9b-it',
    'mistral-saba-24b',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'llama3-8b-8192',
    'llama-3-8b-instant',
    'llama-3-8b-preview',
    'qwen-2.5-32b',
    'qwen-2.5-coder-32b',
    'qwen-qwq-32b',
    'deepseek-r1-distill-llama-70b',
    'deepseek-r1-distill-qwen-32b',
]

def fetch_available_models() -> list:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        res = requests.get(GROQ_MODELS_ENDPOINT, headers=headers)
        res.raise_for_status()
        models = res.json().get("data", [])
        return [m["id"] for m in models]
    except Exception as e:
        print("⚠️ Failed to fetch available models:", e)
        return []

def sanitize_response(content: str) -> str:
    return re.sub(r"<\/?think>", "", content).strip()

def call_llm(prompt: str) -> tuple[str, str]:
    """
    Returns: (clean_response, model_used)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    available_models = fetch_available_models()
    usable_models = [m for m in FALLBACK_MODELS if m in available_models] or FALLBACK_MODELS

    for model in usable_models:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            clean = sanitize_response(content)
            return clean, model
        except Exception as e:
            print(f"❌ Model '{model}' failed. Trying next... ({e})")

    raise RuntimeError("All fallback models failed.")
