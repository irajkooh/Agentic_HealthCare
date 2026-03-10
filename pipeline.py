"""
Healthcare Multi-Agent Pipeline using LangGraph.

Architecture:
    Patient Input
        │
        ▼
  [Supervisor Agent]  ──── decides routing based on urgency & completeness
        │
        ├──▶ [Triage Agent]    → classifies urgency
        │
        ├──▶ [Diagnosis Agent] → differential diagnoses
        │
        └──▶ [Treatment Agent] → treatment plan
                │
                ▼
         [Report Compiler] → structured final report
"""

import json
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from utils.ollama_client import invoke
from agents.triage import run_triage
from agents.diagnosis import run_diagnosis
from agents.treatment import run_treatment


# ─── State Schema ────────────────────────────────────────────────────────────

class PatientState(TypedDict, total=False):
    # Input fields
    name: str
    age: int
    gender: str
    symptoms: str
    vitals: str
    history: str
    medications: str
    allergies: str

    # Routing
    next_agent: str
    completed_agents: list

    # Agent outputs
    triage_output: str
    diagnosis_output: str
    treatment_output: str

    # Supervisor notes
    supervisor_notes: str

    # Final
    final_report: str
    status: str
    error: str


# ─── Supervisor Agent ─────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Chief Medical Officer supervising a team of AI medical agents.
Your job is to:
1. Review the current state of a patient case
2. Decide which agent should act next
3. Ensure all agents complete their tasks before final report

Available agents: triage, diagnosis, treatment, report
Rules:
- Always start with 'triage' if not done
- Move to 'diagnosis' only after triage is complete
- Move to 'treatment' only after diagnosis is complete
- Move to 'report' only after all three agents are done
- If any agent output is missing or incomplete, re-route to that agent

Respond with ONLY a JSON object:
{"next": "triage|diagnosis|treatment|report", "reason": "brief reason"}"""


def supervisor_node(state: PatientState) -> PatientState:
    """Supervisor decides which agent runs next."""
    completed = state.get("completed_agents", [])

    # Deterministic routing for reliability
    if "triage" not in completed:
        next_agent = "triage"
        reason = "Triage not yet performed — must assess urgency first"
    elif "diagnosis" not in completed:
        next_agent = "diagnosis"
        reason = "Triage complete — proceeding to differential diagnosis"
    elif "treatment" not in completed:
        next_agent = "treatment"
        reason = "Diagnosis complete — generating treatment plan"
    else:
        next_agent = "report"
        reason = "All agents complete — compiling final report"

    # Optionally ask LLM to validate routing (adds intelligence for edge cases)
    try:
        summary = f"""
Patient: {state.get('name', 'Unknown')}, {state.get('age', '?')} y/o {state.get('gender', '?')}
Symptoms: {state.get('symptoms', 'N/A')}
Completed agents: {completed}
Triage done: {'Yes' if 'triage' in completed else 'No'}
Diagnosis done: {'Yes' if 'diagnosis' in completed else 'No'}
Treatment done: {'Yes' if 'treatment' in completed else 'No'}
"""
        llm_response = invoke(
            f"Given this case state, confirm routing:\n{summary}\nShould next be: {next_agent}?",
            system=SUPERVISOR_SYSTEM,
            temperature=0.1,
            max_tokens=100
        )
        # Try to parse LLM override
        clean = llm_response.strip()
        if clean.startswith("{"):
            parsed = json.loads(clean)
            if parsed.get("next") in ["triage", "diagnosis", "treatment", "report"]:
                next_agent = parsed["next"]
                reason = parsed.get("reason", reason)
    except Exception:
        pass  # Fall back to deterministic routing

    return {
        **state,
        "next_agent": next_agent,
        "supervisor_notes": reason,
        "status": f"Routing to: {next_agent}"
    }


# ─── Agent Wrapper Nodes ──────────────────────────────────────────────────────

def triage_node(state: PatientState) -> PatientState:
    result = run_triage(state)
    completed = state.get("completed_agents", [])
    if "triage" not in completed:
        completed = completed + ["triage"]
    return {**result, "completed_agents": completed, "status": "Triage complete"}


def diagnosis_node(state: PatientState) -> PatientState:
    result = run_diagnosis(state)
    completed = state.get("completed_agents", [])
    if "diagnosis" not in completed:
        completed = completed + ["diagnosis"]
    return {**result, "completed_agents": completed, "status": "Diagnosis complete"}


def treatment_node(state: PatientState) -> PatientState:
    result = run_treatment(state)
    completed = state.get("completed_agents", [])
    if "treatment" not in completed:
        completed = completed + ["treatment"]
    return {**result, "completed_agents": completed, "status": "Treatment plan complete"}


# ─── Report Compiler Node ─────────────────────────────────────────────────────

def report_node(state: PatientState) -> PatientState:
    """Compile all agent outputs into a structured final report."""
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              HEALTHCARE AI CLINICAL REPORT                       ║
╚══════════════════════════════════════════════════════════════════╝

PATIENT INFORMATION
───────────────────
Name        : {state.get('name', 'Unknown')}
Age / Gender: {state.get('age', '?')} / {state.get('gender', '?')}
Presenting  : {state.get('symptoms', 'N/A')}
Vitals      : {state.get('vitals', 'Not recorded')}
History     : {state.get('history', 'None reported')}
Medications : {state.get('medications', 'None')}
Allergies   : {state.get('allergies', 'NKDA')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 TRIAGE ASSESSMENT  [Agent 1/3]
──────────────────────────────────
{state.get('triage_output', 'Not available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔬 DIFFERENTIAL DIAGNOSIS  [Agent 2/3]
────────────────────────────────────────
{state.get('diagnosis_output', 'Not available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💊 TREATMENT PLAN  [Agent 3/3]
───────────────────────────────
{state.get('treatment_output', 'Not available')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  DISCLAIMER
──────────────
This report is generated by an AI clinical decision support system.
All recommendations MUST be reviewed and approved by a licensed
healthcare professional before any clinical action is taken.
This tool does not replace clinical judgment.

Agents Used  : Triage → Diagnosis → Treatment (Supervisor-routed)
Model        : llama3.2 via Ollama
Routing      : Dynamic Supervisor (LangGraph)
"""
    return {
        **state,
        "final_report": report,
        "status": "Report complete",
        "completed_agents": state.get("completed_agents", []) + ["report"]
    }


# ─── Routing Logic ────────────────────────────────────────────────────────────

def route_from_supervisor(state: PatientState) -> Literal["triage", "diagnosis", "treatment", "report"]:
    return state.get("next_agent", "triage")


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PatientState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("triage", triage_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("treatment", treatment_node)
    graph.add_node("report", report_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes dynamically
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "triage": "triage",
            "diagnosis": "diagnosis",
            "treatment": "treatment",
            "report": "report",
        }
    )

    # After each agent → back to supervisor (except report → END)
    graph.add_edge("triage", "supervisor")
    graph.add_edge("diagnosis", "supervisor")
    graph.add_edge("treatment", "supervisor")
    graph.add_edge("report", END)

    return graph.compile()


# Singleton compiled graph
healthcare_graph = build_graph()


def run_pipeline(patient_data: dict) -> dict:
    """
    Run the full healthcare multi-agent pipeline.

    Args:
        patient_data: dict with patient information

    Returns:
        Final state dict with all agent outputs and final_report
    """
    initial_state: PatientState = {
        "name": patient_data.get("name", "Unknown Patient"),
        "age": patient_data.get("age", 0),
        "gender": patient_data.get("gender", "Unknown"),
        "symptoms": patient_data.get("symptoms", ""),
        "vitals": patient_data.get("vitals", "Not recorded"),
        "history": patient_data.get("history", "None"),
        "medications": patient_data.get("medications", "None"),
        "allergies": patient_data.get("allergies", "NKDA"),
        "completed_agents": [],
        "status": "Starting pipeline",
    }

    final_state = healthcare_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_patient = {
        "name": "Sarah Chen",
        "age": 45,
        "gender": "Female",
        "symptoms": "Severe headache, vision changes, nausea for 2 hours. History of migraines but says this feels different.",
        "vitals": "BP 180/115, HR 88, Temp 37.1°C, SpO2 98%",
        "history": "Hypertension, Migraines, Type 2 Diabetes",
        "medications": "Amlodipine, Metformin, Sumatriptan PRN",
        "allergies": "Sulfa drugs"
    }
    result = run_pipeline(test_patient)
    print(result["final_report"])
