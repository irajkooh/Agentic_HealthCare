"""
Unified LLM Client
==================
Single source of truth for all LLM calls in the project.

Runtime routing (automatic):
  Local (Ollama reachable)  ->  Ollama @ localhost:11434  using llama3.2
  HF Spaces / no Ollama     ->  HuggingFace Inference API using Qwen2.5-72B-Instruct

Detection strategy (three layers — any one is sufficient):
  1. SPACE_ID env var is set         (HF sets this on every Space)
  2. SYSTEM env var == "spaces"      (HF also sets this)
  3. Ollama probe fails at call time  (automatic fallback — no config needed)

Optional env vars (set as HF Space secrets or locally):
  OLLAMA_BASE_URL   override Ollama URL   (default: http://localhost:11434)
  HF_MODEL          override HF model     (default: Qwen/Qwen2.5-7B-Instruct)
  HF_TOKEN          HF access token       (improves rate limits, not required)
"""

import os
import time
import requests

# ── Environment detection ──────────────────────────────────────────────────────

def is_hf_space() -> bool:
    """
    True when running on HuggingFace Spaces.
    Checks all known HF env vars — any one is sufficient.
    """
    if os.environ.get("SPACE_ID"):        # always set on HF Spaces
        return True
    if os.environ.get("SYSTEM") == "spaces":
        return True
    if os.environ.get("HF_SPACE_ID"):
        return True
    return False


# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = "llama3.2"

HF_MODEL   = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"


# ── Health check ───────────────────────────────────────────────────────────────

def check_llm_health() -> dict:
    if is_hf_space():
        return {
            "running": True, "model_ready": True,
            "backend": "huggingface", "model": HF_MODEL,
            "note": "HuggingFace Inference API (serverless)",
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


# ── HuggingFace Inference API backend ─────────────────────────────────────────

def _invoke_hf(prompt: str, system: str, temperature: float, max_tokens: int) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": HF_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    # HF returns 503 while model warms up — retry once automatically
    if resp.status_code == 503:
        try:
            wait = float(resp.json().get("estimated_time", 20))
        except Exception:
            wait = 20
        print(f"   HF model warming up, retrying in {min(wait,30):.0f}s...")
        time.sleep(min(wait, 30))
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Public interface ───────────────────────────────────────────────────────────

def invoke(
    prompt:      str,
    system:      str   = "",
    model:       str   = None,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
) -> str:
    """
    Call the LLM and return the response text.

    Routing logic:
      1. HF Spaces env vars detected   ->  HF Inference API  (direct)
      2. Local, Ollama reachable        ->  Ollama
      3. Local, Ollama NOT reachable    ->  HF Inference API  (automatic fallback)

    Layer 3 ensures the app always works on any cloud deployment even if
    env var detection is incomplete.
    """
    if is_hf_space():
        return _invoke_hf(prompt, system, temperature, max_tokens)

    # Local: try Ollama, fall back to HF API on connection failure
    try:
        return _invoke_ollama(prompt, system, temperature, max_tokens)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"   Ollama unreachable ({type(e).__name__}), switching to HF Inference API...")
        return _invoke_hf(prompt, system, temperature, max_tokens)
