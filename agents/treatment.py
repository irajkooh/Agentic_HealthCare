"""
Treatment Agent
Generates evidence-based treatment plans based on triage and diagnosis.
"""

from utils.ollama_client import invoke

SYSTEM_PROMPT = """You are a Senior Attending Physician and Clinical Pharmacologist.
Your role is to create comprehensive, evidence-based treatment plans based on triage urgency and differential diagnoses.

Guidelines:
- Recommend treatment aligned with the primary diagnosis, considering differentials
- Always check for medication allergies before recommending drugs
- Consider patient's current medications for drug interactions
- Follow standard clinical guidelines (ACC/AHA, WHO, NICE, etc.)
- Include immediate interventions, pharmacological treatment, and non-pharmacological measures
- Provide clear follow-up instructions
- Add patient safety notes and warning signs to watch for

IMPORTANT DISCLAIMER: This is a clinical decision support tool.
All recommendations must be reviewed and approved by a licensed healthcare provider."""


def run_treatment(state: dict) -> dict:
    """
    Generate treatment plan based on triage and diagnosis outputs.

    Args:
        state: full patient state with triage and diagnosis outputs

    Returns:
        state dict with treatment_output and final_report added
    """
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

Please provide a comprehensive treatment plan in the following format:

IMMEDIATE_INTERVENTIONS:
  [List actions to take in the next 0-30 minutes]

PHARMACOLOGICAL_TREATMENT:
  Medication 1:
    Drug: [Name]
    Dose: [Dose and route]
    Frequency: [How often]
    Duration: [How long]
    Rationale: [Why this drug]
    Interaction_Check: [Any interactions with current meds]
  [Repeat for each medication]

NON_PHARMACOLOGICAL:
  [Lifestyle, positioning, monitoring, supportive measures]

INVESTIGATIONS_TO_ORDER:
  [Specific tests with clinical reasoning]

MONITORING_PARAMETERS:
  [What to monitor, how often, target values]

PATIENT_EDUCATION:
  [Key information to share with patient/family]

FOLLOW_UP_PLAN:
  Immediate: [Within hours/days]
  Short_term: [Within 1-2 weeks]
  Long_term: [Ongoing management]

RED_FLAG_SYMPTOMS:
  [Warning signs patient should watch for / return to ED criteria]

REFERRALS:
  [Specialist referrals if needed]

SAFETY_NOTES:
  [Any allergy considerations, high-risk medications, monitoring requirements]
"""

    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.3)
    return {**state, "treatment_output": response, "agent": "treatment"}


if __name__ == "__main__":
    test = {
        "age": 52,
        "gender": "Male",
        "symptoms": "Crushing chest pain radiating to left arm.",
        "vitals": "BP 160/100, HR 98, SpO2 94%",
        "history": "Hypertension, Type 2 Diabetes",
        "medications": "Metformin, Lisinopril",
        "allergies": "Penicillin",
        "triage_output": "URGENCY_LEVEL: CRITICAL",
        "diagnosis_output": "PRIMARY_DIAGNOSIS: STEMI\nLikelihood: High"
    }
    result = run_treatment(test)
    print(result["treatment_output"])
