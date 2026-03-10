"""
HuggingFace Space Entry Point
==============================
On HuggingFace Spaces, we run backend (FastAPI) in a background thread
and Gradio in the main thread (as HF requires Gradio to be the main process).

Ollama is not available on HF Spaces — uses Anthropic Claude API as fallback
via the HF Inference API or any compatible endpoint configured via env vars.

Environment Variables (set in HF Space secrets):
  OLLAMA_BASE_URL  - override if using external Ollama endpoint
  HF_MODEL         - HuggingFace model for inference (fallback)
"""

import sys
import os
import threading
import time
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Start FastAPI in background thread ──────────────────────────────────────

def run_backend():
    """Run FastAPI backend in background thread for HF Spaces."""
    from backend import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


backend_thread = threading.Thread(target=run_backend, daemon=True)
backend_thread.start()

# Give backend a moment to start
time.sleep(3)
print("✅ Backend started on port 8000")

# ─── Launch Gradio as main process ────────────────────────────────────────────

from frontend import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
