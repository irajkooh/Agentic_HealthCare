"""
Healthcare AI Frontend — Gradio UI
Port: 7860

Connects to FastAPI backend on port 8000.
Displays structured clinical reports with urgency badges.
"""

##

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import gradio as gr
from datetime import datetime

BACKEND_URL = "http://localhost:8000"


# ─── Backend Helpers ──────────────────────────────────────────────────────────

def check_backend() -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "ollama": {"running": False}}


def run_analysis(name, age, gender, symptoms, vitals, history, medications, allergies):
    """Call the backend /analyze endpoint and return formatted sections."""
    if not symptoms.strip():
        return (
            "⚠️ Please enter the patient's symptoms.",
            "", "", "", ""
        )

    payload = {
        "name": name or "Anonymous",
        "age": int(age) if age else 0,
        "gender": gender,
        "symptoms": symptoms,
        "vitals": vitals or "Not recorded",
        "history": history or "None reported",
        "medications": medications or "None",
        "allergies": allergies or "NKDA",
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=180)
        if resp.status_code == 503:
            return (
                f"❌ Backend unavailable: {resp.json().get('detail', 'Unknown error')}",
                "", "", "", ""
            )
        if resp.status_code != 200:
            return (f"❌ Error {resp.status_code}: {resp.text}", "", "", "", "")

        data = resp.json()

        agents_done = " → ".join(
            [f"✅ {a.capitalize()}" for a in data.get("agents_completed", [])]
        )
        status_line = f"**Pipeline:** {agents_done}"

        return (
            status_line,
            data.get("triage_output", ""),
            data.get("diagnosis_output", ""),
            data.get("treatment_output", ""),
            data.get("final_report", ""),
        )

    except requests.exceptions.ConnectionError:
        return (
            "❌ Cannot connect to backend at http://localhost:8000.\nMake sure app.py is running.",
            "", "", "", ""
        )
    except Exception as e:
        return (f"❌ Unexpected error: {e}", "", "", "", "")


def get_system_status():
    """Return system status string for the status bar."""
    health = check_backend()
    if health.get("status") == "unreachable":
        return "🔴 Backend: Offline"

    ollama = health.get("ollama", {})
    device = health.get("device_info", {})
    device_name = device.get("device", "?").upper()
    accel = device.get("ollama_accel", "?")

    if health.get("status") == "healthy":
        return f"🟢 Backend: Online | Device: {device_name} ({accel}) | Ollama: Ready | Model: llama3.2"
    return f"🟡 Backend: Degraded | Ollama: {ollama.get('running', False)} | Model: {ollama.get('model_ready', False)}"


# ─── Example Cases ────────────────────────────────────────────────────────────

EXAMPLES = [
    [
        "Sarah Chen", 45, "Female",
        "Severe headache, worst of her life, sudden onset. Visual changes, neck stiffness, photophobia.",
        "BP 150/95, HR 92, Temp 38.2°C, SpO2 99%",
        "Hypertension, Migraines",
        "Amlodipine 5mg",
        "Sulfa drugs"
    ],
    [
        "John Doe", 58, "Male",
        "Crushing chest pain radiating to left arm and jaw. Diaphoresis, nausea. Onset 45 minutes ago.",
        "BP 165/105, HR 102, SpO2 93%, RR 22",
        "Hypertension, Type 2 Diabetes, Smoker",
        "Metformin, Lisinopril, Aspirin 81mg",
        "Penicillin"
    ],
    [
        "Emma Wilson", 28, "Female",
        "Fever for 3 days (38.8°C), painful urination, lower back pain, urinary frequency.",
        "BP 118/75, HR 88, Temp 38.8°C, SpO2 99%",
        "No significant history",
        "Oral contraceptive pill",
        "None known"
    ],
]


# ─── UI Layout ────────────────────────────────────────────────────────────────

CSS = """
.report-box textarea {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.urgency-critical { border-left: 4px solid #dc2626 !important; }
.urgency-high     { border-left: 4px solid #ea580c !important; }
.urgency-medium   { border-left: 4px solid #ca8a04 !important; }
.urgency-low      { border-left: 4px solid #16a34a !important; }
.status-bar { 
    font-size: 13px; 
    color: #6b7280; 
    padding: 6px 12px;
    background: #f9fafb;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}
.agent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}
footer { display: none !important; }
"""

TITLE = """
<div style="text-align:center; padding: 24px 0 8px 0;">
  <h1 style="font-size:2rem; font-weight:700; color:#1e3a5f; margin:0;">
    🏥 Healthcare AI Clinical Decision Support
  </h1>
  <p style="color:#6b7280; margin-top:6px; font-size:0.95rem;">
    Multi-Agent System: Triage → Diagnosis → Treatment &nbsp;|&nbsp; 
    Powered by LangGraph + Ollama (llama3.2) + LangChain
  </p>
  <p style="color:#ef4444; font-size:0.8rem; margin-top:4px;">
    ⚠️ For clinical decision support only. All outputs must be reviewed by a licensed healthcare professional.
  </p>
</div>
"""


def build_ui():
    with gr.Blocks(
        title="Healthcare AI Clinical Decision Support",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="gray",
        )
    ) as demo:

        gr.HTML(TITLE)

        # Status Bar
        with gr.Row():
            status_display = gr.Markdown(
                value=get_system_status(),
                elem_classes=["status-bar"]
            )
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm", scale=0)

        refresh_btn.click(fn=get_system_status, outputs=status_display)

        with gr.Row(equal_height=False):

            # ── LEFT: Patient Input ────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Patient Information")

                with gr.Row():
                    name_in = gr.Textbox(label="Patient Name", placeholder="John Doe", scale=2)
                    age_in = gr.Number(label="Age", minimum=0, maximum=150, value=0, scale=1)
                    gender_in = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Non-binary", "Unknown"],
                        value="Unknown",
                        scale=1
                    )

                symptoms_in = gr.Textbox(
                    label="🩺 Chief Complaint & Symptoms *",
                    placeholder="Describe the patient's primary complaint and all symptoms...",
                    lines=4
                )

                vitals_in = gr.Textbox(
                    label="📊 Vital Signs",
                    placeholder="e.g. BP 120/80, HR 72, Temp 37°C, SpO2 98%, RR 16",
                    lines=2
                )

                history_in = gr.Textbox(
                    label="📋 Medical History",
                    placeholder="Past diagnoses, surgeries, chronic conditions...",
                    lines=2
                )

                with gr.Row():
                    meds_in = gr.Textbox(
                        label="💊 Current Medications",
                        placeholder="Drug name, dose, frequency...",
                        lines=2,
                        scale=1
                    )
                    allergies_in = gr.Textbox(
                        label="⚠️ Allergies",
                        placeholder="Drug/food allergies",
                        lines=2,
                        scale=1
                    )

                analyze_btn = gr.Button(
                    "🔬 Run Clinical Analysis",
                    variant="primary",
                    size="lg"
                )

                pipeline_status = gr.Markdown("*Pipeline status will appear here after analysis.*")

                gr.Markdown("### 📚 Example Cases")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[name_in, age_in, gender_in, symptoms_in,
                            vitals_in, history_in, meds_in, allergies_in],
                    label="Click to load example patient"
                )

            # ── RIGHT: Agent Outputs ───────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Clinical Report")

                with gr.Tab("🚨 Triage Assessment"):
                    triage_out = gr.Textbox(
                        label="Triage Agent Output",
                        lines=16,
                        interactive=False,
                        elem_classes=["report-box"]
                    )

                with gr.Tab("🔬 Differential Diagnosis"):
                    diagnosis_out = gr.Textbox(
                        label="Diagnosis Agent Output",
                        lines=16,
                        interactive=False,
                        elem_classes=["report-box"]
                    )

                with gr.Tab("💊 Treatment Plan"):
                    treatment_out = gr.Textbox(
                        label="Treatment Agent Output",
                        lines=16,
                        interactive=False,
                        elem_classes=["report-box"]
                    )

                with gr.Tab("📋 Full Clinical Report"):
                    final_report_out = gr.Textbox(
                        label="Complete Report (All Agents)",
                        lines=22,
                        interactive=False,
                        elem_classes=["report-box"]
                    )

        # ── Wiring ────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=run_analysis,
            inputs=[name_in, age_in, gender_in, symptoms_in,
                    vitals_in, history_in, meds_in, allergies_in],
            outputs=[pipeline_status, triage_out, diagnosis_out,
                     treatment_out, final_report_out],
            show_progress="full"
        )

        gr.Markdown(
            """---
            <div style="text-align:center; color:#9ca3af; font-size:12px; padding:8px 0;">
            Healthcare AI Clinical Decision Support &nbsp;|&nbsp; 
            LangGraph + FastAPI + Gradio + Ollama (llama3.2) &nbsp;|&nbsp;
            Not a replacement for clinical judgment
            </div>""",
        )

    return demo


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🖥️  Starting Healthcare AI Frontend...")
    print("   UI URL  : http://localhost:7860")
    print("   Backend : http://localhost:8000")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )
