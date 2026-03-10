"""
Diagnosis Agent
Generates differential diagnoses based on triage assessment and patient data.
"""

from utils.ollama_client import invoke

SYSTEM_PROMPT = """You are a board-certified Internal Medicine physician with subspecialty training in Emergency Medicine.
Your role is to generate a structured differential diagnosis based on patient presentation and triage findings.

Guidelines:
- Provide 3-5 differential diagnoses ranked by likelihood
- Include supporting and opposing evidence for each
- Flag any must-not-miss diagnoses even if less likely
- Suggest key investigations to confirm/rule out diagnoses
- Use standard medical terminology but be clear
- Do NOT prescribe treatment — that is handled by the Treatment Agent

Your output must be evidence-based, structured, and clinically sound."""


def run_diagnosis(state: dict) -> dict:
    """
    Run differential diagnosis based on triage output and patient data.

    Args:
        state: patient data dict with triage_output included

    Returns:
        state dict with diagnosis_output added
    """
    prompt = f"""
PATIENT DATA:
- Age: {state.get('age', 'Unknown')} | Gender: {state.get('gender', 'Unknown')}
- Chief Complaint: {state.get('symptoms', 'Not provided')}
- Vitals: {state.get('vitals', 'Not provided')}
- Medical History: {state.get('history', 'Not provided')}
- Current Medications: {state.get('medications', 'None')}
- Allergies: {state.get('allergies', 'NKDA')}

TRIAGE ASSESSMENT:
{state.get('triage_output', 'Not available')}

Based on the above, provide your differential diagnosis in the following format:

PRIMARY_DIAGNOSIS:
  Condition: [Most likely diagnosis]
  Likelihood: [High/Medium/Low]
  Supporting Evidence: [Key findings supporting this]
  Against: [Any findings that argue against]

DIFFERENTIAL_2:
  Condition: [Second diagnosis]
  Likelihood: [High/Medium/Low]
  Supporting Evidence: [Key findings]
  Against: [Findings against]

DIFFERENTIAL_3:
  Condition: [Third diagnosis]
  Likelihood: [High/Medium/Low]
  Supporting Evidence: [Key findings]
  Against: [Findings against]

MUST_NOT_MISS: [Any dangerous diagnoses that must be excluded, even if less likely]

RECOMMENDED_INVESTIGATIONS:
  Immediate: [Tests needed urgently]
  Routine: [Tests that can wait]

CLINICAL_REASONING: [Brief summary of your diagnostic reasoning process]
"""

    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.3)
    return {**state, "diagnosis_output": response, "agent": "diagnosis"}


if __name__ == "__main__":
    test = {
        "name": "John Doe",
        "age": 52,
        "gender": "Male",
        "symptoms": "Crushing chest pain radiating to left arm, started 30 min ago. Sweating profusely.",
        "vitals": "BP 160/100, HR 98, SpO2 94%",
        "history": "Hypertension, Type 2 Diabetes",
        "medications": "Metformin, Lisinopril",
        "allergies": "Penicillin",
        "triage_output": "URGENCY_LEVEL: CRITICAL\nUrgency due to classic ACS presentation."
    }
    result = run_diagnosis(test)
    print(result["diagnosis_output"])
