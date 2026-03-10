"""
Healthcare AI Backend — FastAPI
Port: 8000

Endpoints:
  GET  /health          → system health, Ollama status, device info
  POST /analyze         → run full multi-agent pipeline
  GET  /agents          → list available agents
"""

##
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

from utils.device import get_device_info
from utils.ollama_client import check_ollama_health
from pipeline import run_pipeline


# ─── App Setup ────────────────────────────────────────────────────────────────

# In healthcare, triage is the process of determining the priority of patients’ treatments based on the severity of their condition. It helps ensure that those who need urgent care receive it first, especially in situations where resources are limited, such as emergency rooms or disaster scenarios. Triage can be performed by medical professionals using established protocols to quickly assess and categorize patients.
app = FastAPI(
    title="Healthcare Multi-Agent AI",
    description="LangGraph-powered clinical decision support with Triage, Diagnosis, and Treatment agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PatientRequest(BaseModel):
    name: str = Field(default="Anonymous", description="Patient name")
    age: int = Field(default=0, ge=0, le=150, description="Patient age")
    gender: str = Field(default="Unknown", description="Patient gender")
    symptoms: str = Field(..., min_length=5, description="Chief complaint and symptoms")
    vitals: Optional[str] = Field(default="Not recorded", description="Vital signs if available")
    history: Optional[str] = Field(default="None reported", description="Past medical history")
    medications: Optional[str] = Field(default="None", description="Current medications")
    allergies: Optional[str] = Field(default="NKDA", description="Known allergies")


class HealthResponse(BaseModel):
    status: str
    device_info: dict
    ollama: dict
    agents: list


class AnalysisResponse(BaseModel):
    success: bool
    patient_name: str
    triage_output: str
    diagnosis_output: str
    treatment_output: str
    final_report: str
    agents_completed: list
    status: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health, device, and Ollama status."""
    device_info = get_device_info()
    ollama_status = check_ollama_health()

    overall = "healthy" if ollama_status["running"] and ollama_status["model_ready"] else "degraded"

    return {
        "status": overall,
        "device_info": device_info,
        "ollama": ollama_status,
        "agents": ["supervisor", "triage", "diagnosis", "treatment", "report"]
    }


@app.get("/agents", tags=["Agents"])
async def list_agents():
    """Return information about all available agents."""
    return {
        "agents": [
            {
                "id": "supervisor",
                "name": "Supervisor Agent",
                "role": "Dynamic routing and orchestration of agent pipeline",
                "model": "llama3.2"
            },
            {
                "id": "triage",
                "name": "Triage Agent",
                "role": "Assess patient urgency: CRITICAL / HIGH / MEDIUM / LOW",
                "model": "llama3.2"
            },
            {
                "id": "diagnosis",
                "name": "Diagnosis Agent",
                "role": "Generate differential diagnoses with supporting evidence",
                "model": "llama3.2"
            },
            {
                "id": "treatment",
                "name": "Treatment Agent",
                "role": "Create evidence-based treatment plan with pharmacology",
                "model": "llama3.2"
            }
        ],
        "pipeline": "supervisor → triage → supervisor → diagnosis → supervisor → treatment → report",
        "framework": "LangGraph StateGraph"
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_patient(request: PatientRequest):
    """
    Run the full multi-agent healthcare pipeline on patient data.
    Returns triage, diagnosis, treatment outputs and compiled final report.
    """
    # Check Ollama before running
    ollama = check_ollama_health()
    if not ollama["running"]:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Please start it with: ollama serve"
        )
    if not ollama["model_ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model llama3.2 not available. Run: ollama pull llama3.2"
        )

    patient_data = request.model_dump()

    try:
        result = run_pipeline(patient_data)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return {
        "success": True,
        "patient_name": request.name,
        "triage_output": result.get("triage_output", ""),
        "diagnosis_output": result.get("diagnosis_output", ""),
        "treatment_output": result.get("treatment_output", ""),
        "final_report": result.get("final_report", ""),
        "agents_completed": result.get("completed_agents", []),
        "status": result.get("status", "complete"),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏥 Starting Healthcare AI Backend...")
    device = get_device_info()
    print(f"   Device  : {device['device'].upper()} — {device.get('note', '')}")
    print(f"   Ollama  : {device.get('ollama_accel', 'CPU')}")
    print(f"   API URL : http://localhost:8000")
    print(f"   Docs    : http://localhost:8000/docs")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
