"""
Ollama client wrapper for healthcare agents.
Handles connection, health checks, and streaming.
"""

import json
import requests
from typing import Optional


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def check_ollama_health() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            model_available = any(DEFAULT_MODEL in m for m in models)
            return {
                "running": True,
                "models": models,
                "model_ready": model_available,
                "target_model": DEFAULT_MODEL,
            }
    except requests.exceptions.ConnectionError:
        pass
    return {
        "running": False,
        "models": [],
        "model_ready": False,
        "target_model": DEFAULT_MODEL,
        "error": "Ollama not reachable at http://localhost:11434. Run: ollama serve"
    }


def pull_model_if_needed(model: str = DEFAULT_MODEL) -> bool:
    """Pull the model if not already available."""
    health = check_ollama_health()
    if not health["running"]:
        return False
    if health["model_ready"]:
        return True
    print(f"📥 Pulling model {model}...")
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model},
            stream=True,
            timeout=300
        )
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if data.get("status") == "success":
                    print(f"✅ Model {model} ready.")
                    return True
    except Exception as e:
        print(f"❌ Pull failed: {e}")
    return False


def invoke(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    Send a prompt to Ollama and return the response text.
    Uses /api/chat endpoint (OpenAI-compatible messages format).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "❌ Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("❌ Ollama request timed out (>120s). Model may be loading.")
    except Exception as e:
        raise RuntimeError(f"❌ Ollama error: {e}")
