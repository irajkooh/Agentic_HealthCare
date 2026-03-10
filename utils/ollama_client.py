# Compatibility shim — all calls redirected to llm_client.py
from utils.llm_client import invoke, check_llm_health as check_ollama_health
