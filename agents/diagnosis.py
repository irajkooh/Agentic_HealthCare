"""
Diagnosis Agent
Generates differential diagnoses based on triage assessment and patient data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import invoke

SYSTEM_PROMPT = """You are a board-certified Internal Medicine physician with subspecialty training in Emergency Medicine.
Your role is to generate a structured differential diagnosis based on patient presentation and triage findings.

Guidelines:
- Provide 3-5 differential diagnoses ranked by likelihood
- Include supporting and opposing evidence for each
- Flag any must-not-miss diagnoses even if less likely
- Suggest key investigations to confirm/rule out diagnoses
- Use standard medical terminology
- Do NOT prescribe treatment — that is the Treatment Agent's role"""


def run_diagnosis(state: dict) -> dict:
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

Provide differential diagnosis using this exact format:

PRIMARY_DIAGNOSIS:
  Condition: [Most likely diagnosis]
  Likelihood: [High/Medium/Low]
  Supporting Evidence: [Key findings]
  Against: [Findings arguing against]

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

MUST_NOT_MISS: [Dangerous diagnoses to exclude even if less likely]

RECOMMENDED_INVESTIGATIONS:
  Immediate: [Urgent tests]
  Routine: [Tests that can wait]

CLINICAL_REASONING: [Brief summary of diagnostic reasoning]
"""
    response = invoke(prompt, system=SYSTEM_PROMPT, temperature=0.3)
    return {**state, "diagnosis_output": response}
