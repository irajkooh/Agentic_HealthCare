"""
Triage Agent
Analyzes patient symptoms and classifies urgency level.
Urgency levels: CRITICAL | HIGH | MEDIUM | LOW
"""

from utils.ollama_client import invoke

SYSTEM_PROMPT = """You are an experienced Emergency Triage Nurse with 20 years of clinical experience.
Your role is to assess patient symptoms and assign an urgency level.

URGENCY LEVELS:
- CRITICAL: Life-threatening, requires immediate intervention (chest pain with radiation, stroke symptoms, severe breathing difficulty, uncontrolled bleeding)
- HIGH: Urgent, requires care within 1 hour (high fever >39.5°C, severe pain, head injury, acute allergic reaction)
- MEDIUM: Semi-urgent, requires care within 2-4 hours (moderate pain, persistent fever, non-severe infections)
- LOW: Non-urgent, can wait 4+ hours (minor cuts, mild cold symptoms, minor sprains)

Your output must be structured, clinical, and concise. Always err on the side of caution.
Do NOT provide diagnoses — only triage assessment."""


def run_triage(patient_data: dict) -> dict:
    """
    Run the triage assessment on patient data.

    Args:
        patient_data: dict with keys: name, age, gender, symptoms, vitals, history

    Returns:
        dict with triage result added
    """
    prompt = f"""
PATIENT PRESENTATION:
- Name: {patient_data.get('name', 'Unknown')}
- Age: {patient_data.get('age', 'Unknown')} | Gender: {patient_data.get('gender', 'Unknown')}
- Chief Complaint: {patient_data.get('symptoms', 'Not provided')}
- Vitals (if known): {patient_data.get('vitals', 'Not provided')}
- Medical History: {patient_data.get('history', 'Not provided')}
- Current Medications: {patient_data.get('medications', 'None reported')}
- Allergies: {patient_data.get('allergies', 'NKDA')}

Please provide your triage assessment in the following format:

URGENCY_LEVEL: [CRITICAL/HIGH/MEDIUM/LOW]
URGENCY_RATIONALE: [1-2 sentences explaining why this urgency level]
RED_FLAGS: [List any red flag symptoms identified, or "None identified"]
VITAL_CONCERNS: [Any vital sign concerns, or "Not assessed"]
RECOMMENDED_SETTING: [Emergency Room / Urgent Care / Primary Care / Telehealth]
TRIAGE_NOTES: [Any additional clinical observations for the treating team]
"""

    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.2)
    return {**patient_data, "triage_output": response, "agent": "triage"}


if __name__ == "__main__":
    test = {
        "name": "John Doe",
        "age": 52,
        "gender": "Male",
        "symptoms": "Crushing chest pain radiating to left arm, started 30 min ago. Sweating profusely.",
        "vitals": "BP 160/100, HR 98, SpO2 94%",
        "history": "Hypertension, Type 2 Diabetes",
        "medications": "Metformin, Lisinopril",
        "allergies": "Penicillin"
    }
    result = run_triage(test)
    print(result["triage_output"])
