# How It Works — Healthcare AI Clinical Decision Support

## Overview

This application is a **multi-agent AI system** for clinical decision support.
When you submit a patient case, it passes through three specialized AI agents
in sequence, each building on the previous agent's output. A supervisor
orchestrates the flow using LangGraph, and a final report compiler assembles
all outputs into a structured clinical document.

```
Patient Input
     │
     ▼
[Supervisor]  ──  routes deterministically through the pipeline
     │
     ├──▶ [Triage Agent]      → urgency classification
     │           │
     ├──▶ [Diagnosis Agent]   → differential diagnoses  (uses triage output)
     │           │
     └──▶ [Treatment Agent]   → treatment plan          (uses triage + diagnosis)
                 │
                 ▼
        [Report Compiler]     → assembles full clinical report
                 │
                 ▼
        Final Clinical Report
```

---

## Architecture

### Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph StateGraph |
| Backend API | FastAPI (port 8000) |
| Frontend UI | Gradio 5 (port 7860) |
| LLM — Local | Ollama → llama3.2 (MPS / CUDA / CPU) |
| LLM — Cloud | HuggingFace InferenceClient → Llama-3.1-8B-Instruct |
| State | LangGraph TypedDict (PatientState) |

### LLM Routing

```
Local (python app.py)
  └── is_hf_space() = False
  └── llm_client.py → Ollama @ localhost:11434
  └── model: llama3.2

HuggingFace Spaces (SPACE_ID env var set by HF)
  └── is_hf_space() = True
  └── llm_client.py → HuggingFace InferenceClient
  └── model: meta-llama/Llama-3.1-8B-Instruct
  └── requires: HF_TOKEN Space secret
```

---

## Shared State — PatientState

All agents communicate through a shared `TypedDict` called `PatientState`.
Each agent reads from it and writes its output back to it.

```python
class PatientState(TypedDict):
    # Patient inputs (set once at the start)
    name, age, gender, symptoms, vitals, history, medications, allergies

    # Routing (managed by supervisor)
    next_agent        # which agent runs next
    completed_agents  # list of agents that have finished

    # Agent outputs (filled in as pipeline progresses)
    triage_output     # filled by Triage Agent
    diagnosis_output  # filled by Diagnosis Agent
    treatment_output  # filled by Treatment Agent

    # Final
    final_report      # filled by Report Compiler
    status            # current pipeline status
```

---

## Agents — Input & Output

---

### 🧠 Supervisor Agent

**Role:** Routes the pipeline. Decides which agent runs next.

**Input:** Current `PatientState` — specifically the `completed_agents` list.

**Logic (pure Python — no LLM call):**
```
if "triage"    not in completed → route to triage
if "diagnosis" not in completed → route to diagnosis
if "treatment" not in completed → route to treatment
else                            → route to report
```

**Output:** Updates `next_agent` in state.

**Key design decision:** The supervisor is intentionally deterministic — no LLM
call is made here. This makes routing fast, reliable, and free of parsing errors.

---

### 🚨 Triage Agent (`agents/triage.py`)

**Role:** First responder. Assesses urgency before any diagnosis or treatment.

**Input from PatientState:**
- `symptoms` — chief complaint and all symptoms
- `vitals` — BP, HR, temperature, SpO2, RR
- `history` — past medical conditions
- `medications` — current drugs
- `allergies` — known allergies
- `age`, `gender`, `name`

**System prompt persona:** Emergency Triage Nurse with 20 years experience.

**LLM prompt:** Structured patient presentation requesting a specific output format.

**Output written to PatientState (`triage_output`):**
```
URGENCY_LEVEL:       CRITICAL / HIGH / MEDIUM / LOW
URGENCY_RATIONALE:   Why this urgency level was assigned
RED_FLAGS:           Dangerous symptoms identified
VITAL_CONCERNS:      Abnormal vital signs flagged
RECOMMENDED_SETTING: Emergency Room / Urgent Care / Primary Care / Telehealth
TRIAGE_NOTES:        Additional observations for the treating team
```

**Example output for STEMI patient:**
```
URGENCY_LEVEL: CRITICAL
URGENCY_RATIONALE: Crushing chest pain with radiation to left arm and jaw,
  diaphoresis, and hypoxia are classic STEMI presentation requiring immediate
  cardiac catheterisation.
RED_FLAGS: Chest pain radiating to arm/jaw, diaphoresis, SpO2 93%, HR 102
VITAL_CONCERNS: SpO2 critically low at 93%, hypertensive at 165/105
RECOMMENDED_SETTING: Emergency Room — activate STEMI protocol immediately
TRIAGE_NOTES: Aspirin already on board. Avoid NSAIDs. Prepare for cath lab.
```

---

### 🔬 Diagnosis Agent (`agents/diagnosis.py`)

**Role:** Generates a ranked differential diagnosis list with clinical reasoning.

**Input from PatientState:**
- All patient fields (same as triage)
- `triage_output` — critically, the urgency level and red flags inform which
  diagnoses to prioritise and which must-not-miss conditions to include

**System prompt persona:** Board-certified Internal Medicine physician with
Emergency Medicine subspecialty training.

**LLM prompt:** Full patient data plus triage findings, requesting structured
differential diagnoses.

**Output written to PatientState (`diagnosis_output`):**
```
PRIMARY_DIAGNOSIS:
  Condition:           Most likely diagnosis
  Likelihood:          High / Medium / Low
  Supporting Evidence: Clinical findings that support this diagnosis
  Against:             Findings that argue against it

DIFFERENTIAL_2:        Second most likely diagnosis (same format)
DIFFERENTIAL_3:        Third diagnosis (same format)

MUST_NOT_MISS:         Dangerous diagnoses to rule out even if less likely

RECOMMENDED_INVESTIGATIONS:
  Immediate:           Urgent tests (ECG, troponin, CT, blood cultures...)
  Routine:             Tests that can wait

CLINICAL_REASONING:    Summary of the diagnostic thought process
```

**Why triage output matters here:** If triage flagged CRITICAL urgency with
chest pain, the diagnosis agent will prioritise ACS/STEMI over anxiety or
musculoskeletal causes. The agents are not independent — each one is informed
by the previous.

---

### 💊 Treatment Agent (`agents/treatment.py`)

**Role:** Generates a comprehensive, evidence-based treatment plan. This is the
most detailed output — it covers immediate actions, medications with dosing,
monitoring, patient education, follow-up, and safety flags.

**Input from PatientState:**
- All patient fields
- `triage_output` — urgency drives the immediacy of interventions
- `diagnosis_output` — primary diagnosis drives drug selection and treatment
  pathway; differentials inform what to monitor for

**System prompt persona:** Senior Attending Physician and Clinical Pharmacologist.

**LLM prompt:** Full patient profile, triage findings, and differential diagnoses,
requesting a structured treatment plan. The prompt explicitly instructs:
- Check `allergies` before recommending any drug
- Check `medications` for interactions
- Follow standard guidelines (ACC/AHA, WHO, NICE)

**Output written to PatientState (`treatment_output`):**
```
IMMEDIATE_INTERVENTIONS:
  Actions in the next 0-30 minutes

PHARMACOLOGICAL_TREATMENT:
  Medication 1:
    Drug, Dose & Route, Frequency, Duration, Rationale, Interaction Check
  Medication 2: ...

NON_PHARMACOLOGICAL:
  Positioning, lifestyle, supportive measures, monitoring

INVESTIGATIONS_TO_ORDER:
  Specific tests with clinical reasoning

MONITORING_PARAMETERS:
  What to monitor, frequency, target values

PATIENT_EDUCATION:
  Key information for patient and family

FOLLOW_UP_PLAN:
  Immediate / Short-term / Long-term

RED_FLAG_SYMPTOMS:
  Return-to-ED criteria, warning signs

REFERRALS:
  Specialist consultations needed

SAFETY_NOTES:
  Allergy considerations, high-risk medications, critical monitoring
```

---

### 📄 Report Compiler (`pipeline.py` — `report_node`)

**Role:** Assembles all three agent outputs into a single formatted clinical
document. This is pure Python — no LLM call.

**Input from PatientState:** Everything — patient fields, triage, diagnosis,
and treatment outputs.

**Output written to PatientState (`final_report`):**
A structured text document with:
- Patient information header
- Triage Assessment section (Agent 1/3)
- Differential Diagnosis section (Agent 2/3)
- Treatment Plan section (Agent 3/3)
- Disclaimer

---

## Data Flow — Step by Step

```
1. User fills patient form in Gradio UI

2. Gradio calls POST /analyze on FastAPI backend

3. FastAPI calls run_pipeline(patient_data)

4. LangGraph initialises PatientState with patient data

5. Supervisor checks completed_agents = []
   → routes to "triage"

6. Triage Agent:
   - receives PatientState
   - calls llm_client.invoke(prompt, system)
     → Ollama (local) or HF InferenceClient (cloud)
   - writes triage_output to state
   - adds "triage" to completed_agents

7. Supervisor checks completed_agents = ["triage"]
   → routes to "diagnosis"

8. Diagnosis Agent:
   - receives PatientState including triage_output
   - calls llm_client.invoke(prompt, system)
   - writes diagnosis_output to state
   - adds "diagnosis" to completed_agents

9. Supervisor checks completed_agents = ["triage", "diagnosis"]
   → routes to "treatment"

10. Treatment Agent:
    - receives PatientState including triage + diagnosis outputs
    - calls llm_client.invoke(prompt, system)
    - writes treatment_output to state
    - adds "treatment" to completed_agents

11. Supervisor checks completed_agents = ["triage", "diagnosis", "treatment"]
    → routes to "report"

12. Report Compiler:
    - assembles all outputs into final_report (pure Python, no LLM)
    - adds "report" to completed_agents
    → LangGraph reaches END node

13. run_pipeline() returns final PatientState

14. FastAPI returns all outputs as JSON

15. Gradio displays results in 4 tabs:
    - Triage tab     ← triage_output
    - Diagnosis tab  ← diagnosis_output
    - Treatment tab  ← treatment_output
    - Full Report    ← final_report
```

---

## LLM Call Count

| Step | Agent | LLM Calls |
|---|---|---|
| Routing | Supervisor | 0 (pure Python) |
| Step 1 | Triage | 1 |
| Step 2 | Diagnosis | 1 |
| Step 3 | Treatment | 1 |
| Final | Report Compiler | 0 (pure Python) |
| **Total** | | **3 LLM calls per analysis** |

---

## Key Design Decisions

**Deterministic supervisor:** Early versions called the LLM to decide routing,
wasting 4 extra LLM calls and risking JSON parse failures. The supervisor now
uses pure Python logic — fast, zero-cost, and never fails.

**Cumulative context:** Each agent receives all previous outputs. The treatment
agent has full visibility of both urgency (triage) and likely diagnoses before
recommending any medications. This mirrors how a clinical team communicates.

**Allergy-aware pharmacology:** The treatment agent's system prompt explicitly
instructs it to check the allergies field before recommending any drug, and to
check current medications for interactions.

**No streaming:** All three agents complete before results appear in the UI.
This is intentional — partial clinical information could be misleading.

---

## ⚠️ Disclaimer

This system is for **clinical decision support only**. All outputs are generated
by an AI language model and **must be reviewed and approved by a licensed
healthcare professional** before any clinical action is taken. The system does
not replace clinical judgment, examination, or established diagnostic protocols.
