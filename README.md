---
title: Healthcare AI Clinical Decision Support
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: mit
short_description: Multi-Agent Clinical Decision Support — Triage, Diagnosis, Treatment
---

# 🏥 Healthcare AI Clinical Decision Support

Multi-agent AI system for clinical decision support powered by LangGraph.

## Agents

| Agent | Role |
|---|---|
| **Supervisor** | Deterministic routing through the pipeline |
| **Triage** | Urgency: CRITICAL / HIGH / MEDIUM / LOW |
| **Diagnosis** | Differential diagnoses with supporting evidence |
| **Treatment** | Evidence-based treatment plan + pharmacology |

## Tech Stack

- **Agent Framework**: LangGraph StateGraph
- **LLM (local)**: Ollama with llama3.2
- **LLM (HF Spaces)**: HuggingFace Inference API (Qwen2.5-72B-Instruct)
- **Backend**: FastAPI
- **Frontend**: Gradio 5
- **Device**: Auto-detects MPS / CUDA / CPU

## Local Setup

```bash
pip install -r requirements.txt
ollama serve
ollama pull llama3.2
python app.py
```

## ⚠️ Disclaimer

For clinical decision support only. All outputs must be reviewed by a licensed healthcare professional.
