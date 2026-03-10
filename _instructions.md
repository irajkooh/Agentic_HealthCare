# Healthcare AI — Installation & Deployment Guide

## Table of Contents
1. [How It Works](#1-how-it-works)
2. [Prerequisites](#2-prerequisites)
3. [Local Installation](#3-local-installation)
4. [Running Locally](#4-running-locally)
5. [Testing the System](#5-testing-the-system)
6. [Deploy to HuggingFace Spaces](#6-deploy-to-huggingface-spaces)
7. [Troubleshooting](#7-troubleshooting)
8. [File Reference](#8-file-reference)

---

## 1. How It Works

### LLM Backend — Local vs HuggingFace

The system automatically detects which environment it is running in and routes
all LLM calls accordingly. You do not need to change any code.

```
LOCAL  (python app.py on your machine)
─────────────────────────────────────────────────────
  SYSTEM env var      : not set
  is_hf_space()       : False
  LLM backend         : Ollama  →  http://localhost:11434
  Model               : llama3.2 (3B) running on YOUR machine
  Acceleration        : MPS (Apple Silicon) / CUDA / CPU  ← auto-detected
  Cost                : free, fully private, offline-capable

HUGGINGFACE SPACES  (after git push to your Space)
─────────────────────────────────────────────────────
  SYSTEM env var      : "spaces"  ← set automatically by HuggingFace
  is_hf_space()       : True
  LLM backend         : HuggingFace Inference API
  Model               : Qwen/Qwen2.5-72B-Instruct  (HF's GPU servers)
  Cost                : free tier included, rate-limited
  Note                : Ollama does NOT run anywhere in this mode.
                        No model runs on your machine.
                        All inference happens on HF infrastructure.
```

### Why Qwen2.5-72B on HF and not llama3.2?

Ollama cannot run on HuggingFace Spaces (no persistent process allowed).
Qwen2.5-72B-Instruct is available free on HF's serverless inference endpoint
and is actually a significantly larger and more capable model than the local
llama3.2 3B. Clinical output quality on HF Spaces will generally be better
than local, at the cost of a possible cold-start wait of 20-30 seconds on
the first request if the model has not been recently used.

### Multi-Agent Pipeline

```
Patient Input
      │
      ▼
[Supervisor]  ←  pure Python routing, no LLM calls
      │
      ├──▶ [Triage Agent]     →  CRITICAL / HIGH / MEDIUM / LOW
      │          │
      ├──▶ [Diagnosis Agent]  →  Differential diagnoses + evidence
      │          │
      └──▶ [Treatment Agent]  →  Treatment plan + pharmacology
                 │
                 ▼
          [Report Compiler]   →  Full structured clinical report
```

Framework: LangGraph StateGraph with deterministic supervisor routing.

---

## 2. Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.10 or higher | https://python.org |
| Ollama | Latest | https://ollama.com |
| Git | Any | https://git-scm.com |

**System requirements:**
- RAM: 8 GB minimum (16 GB recommended for smooth llama3.2 inference)
- Disk: ~3 GB for the llama3.2 model download
- macOS Apple Silicon → MPS auto-detected and used
- NVIDIA GPU → CUDA auto-detected and used
- Everything else → CPU (slower but works)

---

## 3. Local Installation

### Step 1 — Unzip the project

```bash
unzip healthcare.zip
cd hc
```

### Step 2 — Create a virtual environment (strongly recommended)

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Install Ollama

Download and install from https://ollama.com for your operating system.

Then start the Ollama server. Keep this terminal open and running:

```bash
ollama serve
```

### Step 5 — Pull the llama3.2 model

Open a new terminal (keep `ollama serve` running) and run:

```bash
ollama pull llama3.2
ollama list   # confirm: shows llama3.2:latest
```

---

## 4. Running Locally

```bash
python app.py
```

`app.py` will:
1. Detect that `SYSTEM` env var is not set → run in local mode
2. Kill any existing process on ports 8000 and 7860
3. Start `backend.py` (FastAPI) on port 8000
4. Start `frontend.py` (Gradio) on port 7860
5. Stream color-coded logs from both services in the terminal
6. Auto-restart either service if it crashes

### Access URLs

| Service | URL |
|---|---|
| Gradio UI | http://localhost:7860 |
| FastAPI backend | http://localhost:8000 |
| API Swagger docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

**Stop:** Press `Ctrl+C`

---

## 5. Testing the System

### Via the UI

1. Open http://localhost:7860
2. Confirm the status bar shows **🟢 Backend: Online | Device: MPS**
3. Click one of the example cases at the bottom of the left panel
4. Click **🔬 Run Clinical Analysis**
5. Watch the tabs populate: Triage → Diagnosis → Treatment → Full Report

### Via curl

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Patient",
    "age": 52,
    "gender": "Male",
    "symptoms": "Crushing chest pain radiating to left arm, onset 30 minutes ago, diaphoresis",
    "vitals": "BP 162/98, HR 104, SpO2 93%",
    "history": "Hypertension, Type 2 Diabetes, smoker",
    "medications": "Metformin, Lisinopril",
    "allergies": "Penicillin"
  }'
```

### Health check

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

Expected when healthy:
```json
{
  "status": "healthy",
  "device_info": {
    "device": "mps",
    "note": "Apple Silicon GPU via Metal Performance Shaders"
  },
  "llm_backend": {
    "running": true,
    "model_ready": true,
    "backend": "ollama",
    "model": "llama3.2"
  }
}
```

### Expected response times (local)

| Device | Per agent call | Full pipeline (3 agents) |
|---|---|---|
| MPS (Apple Silicon) | 15–30 s | ~90 s |
| CUDA (NVIDIA GPU) | 5–15 s | ~40 s |
| CPU only | 60–120 s | ~5 min |

---

## 6. Deploy to HuggingFace Spaces

### What changes on HF

When deployed, `app.py` detects `SYSTEM=spaces` (set automatically by
HuggingFace) and switches to HF mode:

- FastAPI starts in a **background thread** (required by HF)
- Gradio starts as the **main process** (required by HF)
- All LLM calls go to **HuggingFace Inference API** using Qwen2.5-72B-Instruct
- Ollama is **not involved at all** — nothing runs on your local machine

### Step 1 — Create a HuggingFace account

Sign up at https://huggingface.co if you do not have one.

### Step 2 — Create a new Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Fill in:
   - **Space name**: `Agentic_HealthCare`
   - **SDK**: Gradio
   - **SDK Version**: 5.9.1
   - **Hardware**: CPU Basic (free tier is sufficient)
4. Click **Create Space**

### Step 3 — Install HuggingFace CLI and log in

```bash
pip install huggingface_hub
huggingface-cli login
```

Paste your token from https://huggingface.co/settings/tokens
(create a token with **write** permission if needed).

### Step 4 — Push the project

```bash
# Clone your empty Space
git clone https://huggingface.co/spaces/irajkoohi/Agentic_HealthCare
cd Agentic_HealthCare

# Copy all project files into the cloned folder
cp -r /path/to/hc/* .

# Commit and push
git add .
git commit -m "Initial Healthcare AI deployment"
git push
```

HuggingFace builds automatically. Watch the build log in the **App** tab.
Build typically takes 3–5 minutes.

### Step 5 — Verify deployment

Once the build completes your Space is live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/HealthCare
```

The status bar in the UI will show:
```
🟢 Backend: Online | Device: CPU | LLM: huggingface (Qwen/Qwen2.5-72B-Instruct)
```

### Step 6 — (Optional) Add HF_TOKEN for better rate limits

In your Space → **Settings** → **Repository secrets** → add:

| Secret name | Value |
|---|---|
| `HF_TOKEN` | your HuggingFace access token |

Without it the system still works but may hit rate limits under heavy use.

### Cold start note

The Qwen2.5-72B model loads on demand on HF's serverless infrastructure.
If the model has not been used recently, the first request may wait 20–30
seconds while HF warms it up. This is handled automatically with a retry.
Subsequent requests in the same session are fast.

---

## 7. Troubleshooting

### Ollama not running (local)
```bash
ollama serve
curl http://localhost:11434/api/tags   # should return JSON with models list
```

### llama3.2 model not found (local)
```bash
ollama pull llama3.2
ollama list   # confirm it appears
```

### Port still busy after app.py tries to kill it
```bash
# macOS / Linux
lsof -ti:8000 | xargs kill -9
lsof -ti:7860 | xargs kill -9
python app.py
```

### Status bar shows 🔴 Backend: Offline
- Make sure you ran `python app.py`, not `python frontend.py` alone
- Check your terminal for error messages from the backend subprocess
- Visit http://localhost:8000/health directly in your browser

### Responses very slow (CPU only)
```bash
# Use the smaller 1B variant for faster local inference
ollama pull llama3.2:1b
```
Then edit `utils/llm_client.py` line:
```python
OLLAMA_MODEL = "llama3.2:1b"
```

### HuggingFace build fails
- Check the build log in your Space's App tab for the exact error
- Confirm `requirements.txt` is in the repo root
- Confirm `README.md` frontmatter contains exactly: `app_file: app.py`

### HF Inference API returns 503 on first request
Cold start — the model is warming up on HF's servers. The system retries
automatically. Just wait 20–30 seconds.

### HF rate limit errors
Add `HF_TOKEN` as a Space secret (Step 6 above).

---

## 8. File Reference

```
hc/
├── app.py              ← START HERE — single entry point for both environments
│                          Local    : kills ports 8000/7860, spawns backend + frontend
│                          HF Spaces: background FastAPI thread + main Gradio process
│
├── backend.py          ← FastAPI server (port 8000)
│                          GET  /health   LLM + device status
│                          GET  /agents   agent descriptions
│                          POST /analyze  runs the full pipeline
│
├── frontend.py         ← Gradio UI (port 7860)
│                          Patient input form, status bar, 4 output tabs
│                          Talks to backend via HTTP POST /analyze
│
├── pipeline.py         ← LangGraph StateGraph
│                          Supervisor routes deterministically (pure Python, no LLM)
│                          triage → diagnosis → treatment → report
│
├── agents/
│   ├── triage.py       ← Agent 1: urgency classification (CRITICAL/HIGH/MEDIUM/LOW)
│   ├── diagnosis.py    ← Agent 2: differential diagnoses with supporting evidence
│   └── treatment.py    ← Agent 3: treatment plan, medications, follow-up
│
├── utils/
│   ├── llm_client.py   ← THE unified LLM router — all agents import from here
│   │                      Local  → Ollama @ localhost:11434  using llama3.2
│   │                      HF     → HF Inference API          using Qwen2.5-72B
│   └── device.py       ← MPS / CUDA / CPU auto-detection
│
├── requirements.txt    ← Python dependencies
├── README.md           ← HuggingFace Space config  (app_file: app.py)
└── _instructions.md    ← This file
```
