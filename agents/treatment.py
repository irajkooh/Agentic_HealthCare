"""
Treatment Agent
Generates evidence-based treatment plans from triage and diagnosis outputs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import invoke

SYSTEM_PROMPT = """You are a Senior Attending Physician and Clinical Pharmacologist.
Your role is to create comprehensive, evidence-based treatment plans based on triage urgency and differential diagnoses.

Guidelines:
- Recommend treatment aligned with the primary diagnosis, considering differentials
- Always check for medication allergies before recommending drugs
- Consider patient's current medications for drug interactions
- Follow standard clinical guidelines (ACC/AHA, WHO, NICE)
- Include immediate interventions, pharmacological and non-pharmacological measures
- Provide clear follow-up instructions and patient safety warnings

IMPORTANT: This is a clinical decision support tool. All recommendations must be
reviewed and approved by a licensed healthcare provider before any action is taken."""


def run_treatment(state: dict) -> dict:
    prompt = f"""
PATIENT PROFILE:
- Age: {state.get('age', 'Unknown')} | Gender: {state.get('gender', 'Unknown')}
- Chief Complaint: {state.get('symptoms', 'Not provided')}
- Vitals: {state.get('vitals', 'Not provided')}
- Medical History: {state.get('history', 'Not provided')}
- Current Medications: {state.get('medications', 'None')}
- Known Allergies: {state.get('allergies', 'NKDA')}

TRIAGE FINDINGS:
{state.get('triage_output', 'Not available')}

DIFFERENTIAL DIAGNOSIS:
{state.get('diagnosis_output', 'Not available')}

Provide a comprehensive treatment plan using this exact format:

IMMEDIATE_INTERVENTIONS:
  [Actions to take in the next 0-30 minutes]

PHARMACOLOGICAL_TREATMENT:
  Medication 1:
    Drug: [Name]
    Dose & Route: [Dose and administration route]
    Frequency: [How often]
    Duration: [How long]
    Rationale: [Why this drug]
    Interaction Check: [Any interactions with current meds]

NON_PHARMACOLOGICAL:
  [Lifestyle, positioning, monitoring, supportive measures]

INVESTIGATIONS_TO_ORDER:
  [Specific tests with reasoning]

MONITORING_PARAMETERS:
  [What to monitor, how often, target values]

PATIENT_EDUCATION:
  [Key information for patient/family]

FOLLOW_UP_PLAN:
  Immediate: [Within hours/days]
  Short_term: [1-2 weeks]
  Long_term: [Ongoing management]

RED_FLAG_SYMPTOMS:
  [Warning signs / return-to-ED criteria]

REFERRALS:
  [Specialist referrals if needed]

SAFETY_NOTES:
  [Allergy considerations, high-risk medications, monitoring requirements]
"""
    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.3)
    return {**state, "treatment_output": response}
