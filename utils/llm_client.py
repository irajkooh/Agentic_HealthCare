"""
Unified LLM Client
==================
Local  -> Ollama @ localhost:11434 using llama3.2
HF     -> HuggingFace InferenceClient using meta-llama/Llama-3.2-3B-Instruct

HF_TOKEN must be set as a Space secret (required by HF Inference Providers).
"""

import os
import time
import requests

# ── Environment detection ──────────────────────────────────────────────────────

def is_hf_space() -> bool:
    if os.environ.get("SPACE_ID"):
        return True
    if os.environ.get("SYSTEM") == "spaces":
        return True
    if os.environ.get("HF_SPACE_ID"):
        return True
    return False


# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = "llama3.2"

HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── Health check ───────────────────────────────────────────────────────────────

def check_llm_health() -> dict:
    if is_hf_space():
        return {
            "running": True, "model_ready": True,
            "backend": "huggingface", "model": HF_MODEL,
            "note": "HuggingFace Inference Providers",
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


# ── Ollama backend ─────────────────────────────────────────────────────────────

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


# ── HuggingFace Inference Providers backend ────────────────────────────────────

def _invoke_hf(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Add it to requirements.txt")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set. Add it as a Space secret.")

    client = InferenceClient(
        provider="together",
        api_key=HF_TOKEN,
    )

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


# ── Public interface ───────────────────────────────────────────────────────────

def invoke(
    prompt:      str,
    system:      str   = "",
    model:       str   = None,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
) -> str:
    if is_hf_space():
        return _invoke_hf(prompt, system, temperature, max_tokens)
    try:
        return _invoke_ollama(prompt, system, temperature, max_tokens)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _invoke_hf(prompt, system, temperature, max_tokens)