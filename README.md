---
title: "Healthcare AI Clinical Decision Support"
emoji: "🏥"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: true
license: mit
short_description: "Multi-Agent Clinical Decision Support: Triage to Treatment"
---
# 🏥 Healthcare AI Clinical Decision Support

A **multi-agent AI system** for clinical decision support using LangGraph, Ollama (llama3.2), FastAPI, and Gradio.

## 🤖 Agent Architecture

```
Patient Input
      │
      ▼
[Supervisor Agent]  ← LangGraph StateGraph with dynamic routing
      │
      ├──▶ [Triage Agent]     → Urgency: CRITICAL / HIGH / MEDIUM / LOW
      ├──▶ [Diagnosis Agent]  → Differential diagnoses with evidence
      └──▶ [Treatment Agent]  → Evidence-based treatment plan
              │
              ▼
       [Final Report Compiler]
```

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM | Ollama llama3.2 (3B) |
| Backend API | FastAPI |
| Frontend UI | Gradio 5 |
| Device | MPS / CUDA / CPU auto-detect |

## 🚀 Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull model
ollama serve
ollama pull llama3.2

# 3. Launch everything
python app.py
```

Access:
- **UI**: http://localhost:7860
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ⚠️ Disclaimer

This system is for **clinical decision support only**. All outputs must be reviewed and approved by a licensed healthcare professional. This tool does not replace clinical judgment.

## 📁 Project Structure

```
healthcare/
├── app.py           # Orchestrator (kills ports, starts backend + frontend)
├── backend.py       # FastAPI API server
├── frontend.py      # Gradio UI
├── pipeline.py      # LangGraph multi-agent pipeline
├── agents/
│   ├── triage.py    # Triage Agent
│   ├── diagnosis.py # Diagnosis Agent
│   └── treatment.py # Treatment Agent
├── utils/
│   ├── device.py    # MPS/CUDA/CPU detection
│   └── ollama_client.py # Ollama wrapper
└── requirements.txt
```
