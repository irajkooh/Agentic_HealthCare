"""
Healthcare AI Backend — FastAPI
================================
Endpoints:
  GET  /health    system health + LLM backend status + device info
  GET  /agents    list of agents in the pipeline
  POST /analyze   run the full multi-agent pipeline

NOTE: No Ollama gate. Pipeline runs and llm_client.py handles
routing to Ollama (local) or HF Inference API (HF Spaces) automatically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.device     import get_device_info
from utils.llm_client import check_llm_health, is_hf_space
from pipeline         import run_pipeline


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare Multi-Agent AI",
    description="LangGraph multi-agent clinical decision support",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PatientRequest(BaseModel):
    name:        str           = Field(default="Anonymous")
    age:         int           = Field(default=0, ge=0, le=150)
    gender:      str           = Field(default="Unknown")
    symptoms:    str           = Field(..., min_length=5)
    vitals:      Optional[str] = Field(default="Not recorded")
    history:     Optional[str] = Field(default="None reported")
    medications: Optional[str] = Field(default="None")
    allergies:   Optional[str] = Field(default="NKDA")


class AnalysisResponse(BaseModel):
    success:          bool
    patient_name:     str
    triage_output:    str
    diagnosis_output: str
    treatment_output: str
    final_report:     str
    agents_completed: list
    status:           str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    device_info = get_device_info()
    llm         = check_llm_health()
    status      = "healthy" if llm["running"] and llm["model_ready"] else "degraded"
    return {
        "status":      status,
        "device_info": device_info,
        "llm_backend": llm,
        "ollama":      llm,   # kept for frontend compat
        "agents":      ["supervisor", "triage", "diagnosis", "treatment", "report"],
    }


@app.get("/agents", tags=["Agents"])
async def list_agents():
    backend = "HF Inference API" if is_hf_space() else "Ollama (llama3.2)"
    return {
        "agents": [
            {"id": "supervisor", "role": "Dynamic routing between agents",               "model": backend},
            {"id": "triage",     "role": "Urgency classification: CRITICAL/HIGH/MED/LOW","model": backend},
            {"id": "diagnosis",  "role": "Differential diagnoses with evidence",          "model": backend},
            {"id": "treatment",  "role": "Evidence-based treatment plan",                 "model": backend},
        ],
        "pipeline":    "supervisor -> triage -> diagnosis -> treatment -> report",
        "framework":   "LangGraph StateGraph",
        "llm_backend": backend,
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_patient(request: PatientRequest):
    """
    Run the full multi-agent pipeline.
    llm_client.py automatically routes to Ollama (local) or HF API (HF Spaces).
    No Ollama pre-check — the pipeline handles connection errors internally.
    """
    try:
        result = run_pipeline(request.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return {
        "success":          True,
        "patient_name":     request.name,
        "triage_output":    result.get("triage_output",    ""),
        "diagnosis_output": result.get("diagnosis_output", ""),
        "treatment_output": result.get("treatment_output", ""),
        "final_report":     result.get("final_report",     ""),
        "agents_completed": result.get("completed_agents", []),
        "status":           result.get("status",           "complete"),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    info = get_device_info()
    print("🏥 Starting Healthcare AI Backend...")
    print(f"   Device : {info['device'].upper()} - {info.get('note','')}")
    print(f"   URL    : http://localhost:8000")
    print(f"   Docs   : http://localhost:8000/docs")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
