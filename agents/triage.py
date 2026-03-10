"""
Triage Agent
Assesses patient symptoms and assigns urgency: CRITICAL | HIGH | MEDIUM | LOW
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import invoke

SYSTEM_PROMPT = """You are an experienced Emergency Triage Nurse with 20 years of clinical experience.
Your role is to assess patient symptoms and assign an urgency level.

URGENCY LEVELS:
- CRITICAL: Life-threatening, requires immediate intervention (chest pain with radiation, stroke symptoms, severe breathing difficulty, uncontrolled bleeding)
- HIGH: Urgent, requires care within 1 hour (high fever >39.5C, severe pain, head injury, acute allergic reaction)
- MEDIUM: Semi-urgent, requires care within 2-4 hours (moderate pain, persistent fever, non-severe infections)
- LOW: Non-urgent, can wait 4+ hours (minor cuts, mild cold symptoms, minor sprains)

Your output must be structured, clinical, and concise. Always err on the side of caution.
Do NOT provide diagnoses — only triage assessment."""


def run_triage(patient_data: dict) -> dict:
    prompt = f"""
PATIENT PRESENTATION:
- Name: {patient_data.get('name', 'Unknown')}
- Age: {patient_data.get('age', 'Unknown')} | Gender: {patient_data.get('gender', 'Unknown')}
- Chief Complaint: {patient_data.get('symptoms', 'Not provided')}
- Vitals (if known): {patient_data.get('vitals', 'Not provided')}
- Medical History: {patient_data.get('history', 'Not provided')}
- Current Medications: {patient_data.get('medications', 'None reported')}
- Allergies: {patient_data.get('allergies', 'NKDA')}

Provide your triage assessment using this exact format:

URGENCY_LEVEL: [CRITICAL/HIGH/MEDIUM/LOW]
URGENCY_RATIONALE: [1-2 sentences]
RED_FLAGS: [List red flag symptoms, or "None identified"]
VITAL_CONCERNS: [Vital sign concerns, or "Not assessed"]
RECOMMENDED_SETTING: [Emergency Room / Urgent Care / Primary Care / Telehealth]
TRIAGE_NOTES: [Additional observations for treating team]
"""
    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.2)
    return {**patient_data, "triage_output": response}
