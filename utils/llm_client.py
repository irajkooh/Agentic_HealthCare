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

# HF Inference Providers now require credits — even "free" models.
# We use Groq instead: free tier, fast, supports llama3/mixtral, no credits needed.
# Get a free API key at: https://console.groq.com
# Set GROQ_API_KEY as a Space secret.
#
# Fallback chain: Groq → HF Inference API (if GROQ_API_KEY not set)
HF_MODEL   = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# llama3-8b-8192 is free on Groq, fast, and good for clinical tasks
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def check_llm_health() -> dict:
    if is_hf_space():
        return {
            "running": True, "model_ready": True,
            "backend": "huggingface", "model": HF_MODEL,
            "note": "HuggingFace",
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


def _invoke_groq(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    """Call Groq API — free tier, fast, no credits needed. Get key at console.groq.com"""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    messages = []
    if system:
        if len(system) > 3000:
            system = system[:3000] + "\n...[truncated]"
        messages.append({"role": "system", "content": system})
    if len(prompt) > 4000:
        prompt = prompt[:4000] + "\n...[truncated]"
    messages.append({"role": "user", "content": prompt})
    print(f"[GROQ] Calling {GROQ_MODEL}, system={len(system)}chars, prompt={len(prompt)}chars", flush=True)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _invoke_hf(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    from huggingface_hub import InferenceClient
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Go to Space Settings → Variables and Secrets "
            "and add HF_TOKEN as a secret with your HuggingFace token."
        )
    client = InferenceClient(token=HF_TOKEN)
    messages = []
    if system:
        # Trim system prompt to avoid token limit errors on HF Inference API
        # Llama-3.1-8B context = 8k tokens; keep system under ~3000 chars
        if len(system) > 3000:
            system = system[:3000] + "\n...[truncated]"
        messages.append({"role": "system", "content": system})
    # Trim prompt too
    if len(prompt) > 4000:
        prompt = prompt[:4000] + "\n...[truncated]"
    messages.append({"role": "user", "content": prompt})
    print(f"[HF] Calling {HF_MODEL}, system={len(system)}chars, prompt={len(prompt)}chars", flush=True)
    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[HF ERROR] {type(e).__name__}: {e}", flush=True)
        raise


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
        if GROQ_API_KEY:
            return _sanitize(_invoke_groq(prompt, system, temperature, max_tokens))
        # Fallback to HF if no Groq key (will use credits)
        return _sanitize(_invoke_hf(prompt, system, temperature, max_tokens))
    try:
        return _sanitize(_invoke_ollama(prompt, system, temperature, max_tokens))
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if GROQ_API_KEY:
            return _sanitize(_invoke_groq(prompt, system, temperature, max_tokens))
        return _sanitize(_invoke_hf(prompt, system, temperature, max_tokens))