"""
Unified LLM Client
==================
Local  -> Ollama @ localhost:11434 using llama3.2
HF     -> HuggingFace InferenceClient using meta-llama//Llama-3.1-8B-Instruct
         via "novita" provider (works with HF_TOKEN only, no extra API key)

HF_TOKEN must be set as a Space secret.
"""

import os
import requests

def is_hf_space() -> bool:
    if os.environ.get("SPACE_ID"):
        return True
    if os.environ.get("SYSTEM") == "spaces":
        return True
    if os.environ.get("HF_SPACE_ID"):
        return True
    return False

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = "llama3.2"

HF_MODEL    = os.environ.get("HF_MODEL", "meta-llama//Llama-3.1-8B-Instruct")
#HF_MODEL    = os.environ.get("HF_MODEL", "meta-llama/mistralai/Mistral-7B-Instruct-v0.2")
HF_PROVIDER = os.environ.get("HF_PROVIDER", "novita")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")


def check_llm_health() -> dict:
    if is_hf_space():
        return {
            "running": True, "model_ready": True,
            "backend": "huggingface", "model": HF_MODEL,
            "note": f"HuggingFace/{HF_PROVIDER}",
        }
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models      = [m["name"] for m in resp.json().get("models", [])]
            model_ready = any(OLLAMA_MODEL in m for m in models)
            return {
                "running": True, "model_ready": model_ready,
                "backend": "ollama", "model": OLLAMA_MODEL,
                "models_available": models, "note": "Ollama local inference",
            }
    except Exception:
        pass
    return {
        "running": False, "model_ready": False,
        "backend": "ollama", "model": OLLAMA_MODEL,
        "error": "Ollama not reachable. Run: ollama serve && ollama pull llama3.2",
    }


def _invoke_ollama(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": OLLAMA_MODEL, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _invoke_hf(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    from huggingface_hub import InferenceClient
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Add it as a Space secret.")
    client = InferenceClient(provider=HF_PROVIDER, api_key=HF_TOKEN)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _sanitize(text: str) -> str:
    """Strip JSON wrapping that some model versions return e.g. {"text":"...","type":"text"}"""
    import json
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed.get("text") or parsed.get("content") or text
        except (json.JSONDecodeError, ValueError):
            pass
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                parts = [item.get("text") or item.get("content") or "" 
                         for item in parsed if isinstance(item, dict)]
                return "\n".join(p for p in parts if p) or text
        except (json.JSONDecodeError, ValueError):
            pass
    return text


def invoke(
    prompt:      str,
    system:      str   = "",
    model:       str   = None,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
) -> str:
    if is_hf_space():
        return _sanitize(_invoke_hf(prompt, system, temperature, max_tokens))
    try:
        return _sanitize(_invoke_ollama(prompt, system, temperature, max_tokens))
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _sanitize(_invoke_hf(prompt, system, temperature, max_tokens))