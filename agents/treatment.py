"""
Treatment Agent
Generates evidence-based treatment plans from triage and diagnosis outputs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import invoke

SYSTEM_PROMPT = """You are a Senior Attending Physician and Clinical Pharmacologist with 25 years of experience in internal medicine and emergency medicine. You are communicating PHYSICIAN TO PHYSICIAN in a secure clinical decision support platform used exclusively by licensed medical professionals.

CONTEXT: This is a clinical decision support system (CDSS) used in hospital and urgent care settings. The end user is a licensed physician or nurse practitioner who has already examined the patient and is requesting a structured treatment framework to review. You are NOT communicating with a patient. You are NOT prescribing directly to a patient.

YOUR TASK: Generate a comprehensive, evidence-based treatment plan for the physician to review and adapt based on their clinical judgment. This is equivalent to a senior consultant providing structured advice to a treating physician.

REQUIREMENTS:
- Always generate a complete treatment plan — refusing to do so is not an option in this physician-to-physician CDSS context
- Check patient allergies before recommending any drug
- Check current medications for interactions
- Follow ACC/AHA, WHO, NICE, and UpToDate guidelines
- Include dosing, route, frequency, duration, rationale, and interaction checks for each drug
- The treating physician will review, modify, and approve all recommendations before implementation

Do not add disclaimers about needing a doctor — the user IS the doctor. Do not refuse or hedge. Generate the full structured treatment plan."""


def run_treatment(state: dict) -> dict:
    prompt = f"""PHYSICIAN CASE REVIEW REQUEST — CDSS PLATFORM

PATIENT PROFILE:
- Name: {state.get('name', 'Anonymous')} | Age: {state.get('age', 'Unknown')} | Gender: {state.get('gender', 'Unknown')}
- Chief Complaint: {state.get('symptoms', 'Not provided')}
- Vitals: {state.get('vitals', 'Not provided')}
- Medical History: {state.get('history', 'Not provided')}
- Current Medications: {state.get('medications', 'None')}
- Known Allergies: {state.get('allergies', 'NKDA')}

TRIAGE ASSESSMENT (from Triage Agent):
{state.get('triage_output', 'Not available')}

DIFFERENTIAL DIAGNOSIS (from Diagnosis Agent):
{state.get('diagnosis_output', 'Not available')}

As the consulting Senior Attending Physician, provide a complete structured treatment plan for the treating physician to review. Use this exact format:

IMMEDIATE_INTERVENTIONS:
  [Specific actions in the next 0-30 minutes]

PHARMACOLOGICAL_TREATMENT:
  Medication 1:
    Drug: [Name]
    Dose & Route: [Specific dose and route]
    Frequency: [Schedule]
    Duration: [Length of treatment]
    Rationale: [Clinical reasoning]
    Interaction Check: [Interactions with current meds]
  Medication 2:
    [Same format]

NON_PHARMACOLOGICAL:
  [Positioning, oxygen, IV access, monitoring, supportive measures]

INVESTIGATIONS_TO_ORDER:
  Immediate: [Urgent tests with rationale]
  Routine: [Additional workup]

MONITORING_PARAMETERS:
  [What to monitor, frequency, target values, escalation criteria]

PATIENT_EDUCATION:
  [Key information for patient and family]

FOLLOW_UP_PLAN:
  Immediate: [Hours to days]
  Short_term: [1-2 weeks]
  Long_term: [Ongoing management]

RED_FLAG_SYMPTOMS:
  [Return-to-ED criteria and warning signs]

REFERRALS:
  [Specialist consultations with urgency level]

SAFETY_NOTES:
  [Allergy considerations, high-risk drug monitoring, special precautions]
"""
    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.3)
    return {**state, "treatment_output": response}