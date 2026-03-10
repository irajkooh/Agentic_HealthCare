"""
Healthcare AI Frontend — Gradio UI
Gradio 6 compatible. CSS injected via gr.Blocks so it works
identically on local and HuggingFace Spaces.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import gradio as gr

BACKEND_URL = "http://localhost:8000"

EXAMPLES = [
    [
        "Sarah Chen", 45, "Female",
        "Severe headache, worst of her life, sudden onset. Visual changes, neck stiffness, photophobia.",
        "BP 150/95, HR 92, Temp 38.2C, SpO2 99%",
        "Hypertension, Migraines",
        "Amlodipine 5mg",
        "Sulfa drugs",
    ],
    [
        "John Doe", 58, "Male",
        "Crushing chest pain radiating to left arm and jaw. Diaphoresis, nausea. Onset 45 min ago.",
        "BP 165/105, HR 102, SpO2 93%, RR 22",
        "Hypertension, Type 2 Diabetes, Smoker",
        "Metformin, Lisinopril, Aspirin 81mg",
        "Penicillin",
    ],
    [
        "Emma Wilson", 28, "Female",
        "Fever for 3 days (38.8C), painful urination, lower back pain, urinary frequency.",
        "BP 118/75, HR 88, Temp 38.8C, SpO2 99%",
        "No significant history",
        "Oral contraceptive pill",
        "None known",
    ],
]

# ── Workflow diagram ───────────────────────────────────────────────────────────

WORKFLOW_HTML = """
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:28px 20px;margin:8px 0;font-family:'Segoe UI',sans-serif;">
  <h3 style="text-align:center;color:#1e3a5f;margin:0 0 24px 0;font-size:1.1rem;">
    🔀 LangGraph Multi-Agent Pipeline
  </h3>
  <div style="display:flex;justify-content:center;margin-bottom:6px;">
    <div style="background:#1e3a5f;color:white;padding:10px 28px;border-radius:8px;font-weight:600;font-size:0.9rem;">
      📋 Patient Input
    </div>
  </div>
  <div style="text-align:center;font-size:1.4rem;color:#64748b;line-height:1.2;">↓</div>
  <div style="display:flex;justify-content:center;margin-bottom:4px;">
    <div style="background:#7c3aed;color:white;padding:10px 28px;border-radius:8px;font-weight:600;font-size:0.9rem;text-align:center;">
      🧠 Supervisor Agent<br>
      <span style="font-size:0.75rem;font-weight:400;opacity:0.9;">Deterministic routing — pure Python, no LLM calls</span>
    </div>
  </div>
  <div style="display:flex;justify-content:center;gap:0;margin:8px 0 4px 0;">
    <div style="flex:1;border-top:2px solid #94a3b8;border-left:2px solid #94a3b8;height:20px;border-radius:4px 0 0 0;margin-top:10px;"></div>
    <div style="width:2px;background:#94a3b8;height:30px;"></div>
    <div style="flex:1;border-top:2px solid #94a3b8;border-right:2px solid #94a3b8;height:20px;border-radius:0 4px 0 0;margin-top:10px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;gap:12px;margin-bottom:4px;">
    <div style="flex:1;background:#dc2626;color:white;padding:12px 10px;border-radius:8px;text-align:center;">
      <div style="font-size:1.3rem;">🚨</div>
      <div style="font-weight:700;font-size:0.85rem;margin:4px 0;">Triage Agent</div>
      <div style="font-size:0.72rem;opacity:0.9;line-height:1.5;">
        <b>IN:</b> symptoms, vitals,<br>history, demographics<br>
        <b>OUT:</b> urgency level,<br>red flags, care setting
      </div>
    </div>
    <div style="flex:1;background:#0369a1;color:white;padding:12px 10px;border-radius:8px;text-align:center;">
      <div style="font-size:1.3rem;">🔬</div>
      <div style="font-weight:700;font-size:0.85rem;margin:4px 0;">Diagnosis Agent</div>
      <div style="font-size:0.72rem;opacity:0.9;line-height:1.5;">
        <b>IN:</b> patient data +<br>triage output<br>
        <b>OUT:</b> differentials,<br>evidence, investigations
      </div>
    </div>
    <div style="flex:1;background:#059669;color:white;padding:12px 10px;border-radius:8px;text-align:center;">
      <div style="font-size:1.3rem;">💊</div>
      <div style="font-weight:700;font-size:0.85rem;margin:4px 0;">Treatment Agent</div>
      <div style="font-size:0.72rem;opacity:0.9;line-height:1.5;">
        <b>IN:</b> patient data +<br>triage + diagnosis<br>
        <b>OUT:</b> medications,<br>plan, follow-up
      </div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;gap:0;margin:4px 0;">
    <div style="flex:1;border-bottom:2px solid #94a3b8;border-left:2px solid #94a3b8;height:20px;border-radius:0 0 0 4px;"></div>
    <div style="width:2px;background:#94a3b8;height:30px;"></div>
    <div style="flex:1;border-bottom:2px solid #94a3b8;border-right:2px solid #94a3b8;height:20px;border-radius:0 0 4px 0;"></div>
  </div>
  <div style="text-align:center;font-size:1.4rem;color:#64748b;line-height:1.2;">↓</div>
  <div style="display:flex;justify-content:center;margin-bottom:6px;">
    <div style="background:#b45309;color:white;padding:10px 28px;border-radius:8px;font-weight:600;font-size:0.9rem;text-align:center;">
      📄 Report Compiler<br>
      <span style="font-size:0.75rem;font-weight:400;opacity:0.9;">Assembles all outputs — pure Python, no LLM</span>
    </div>
  </div>
  <div style="text-align:center;font-size:1.4rem;color:#64748b;line-height:1.2;">↓</div>
  <div style="display:flex;justify-content:center;">
    <div style="background:#1e3a5f;color:white;padding:10px 28px;border-radius:8px;font-weight:600;font-size:0.9rem;">
      ✅ Clinical Report Output
    </div>
  </div>
  <div style="margin-top:20px;padding:10px 16px;background:#eff6ff;border-radius:8px;border-left:4px solid #3b82f6;">
    <div style="font-size:0.78rem;color:#1e40af;line-height:1.6;">
      <b>🖥️ Local:</b> Ollama → llama3.2 (MPS/CUDA/CPU) &nbsp;|&nbsp;
      <b>☁️ HF Spaces:</b> HuggingFace InferenceClient → Llama-3.1-8B<br>
      <b>🔀 Framework:</b> LangGraph StateGraph · 3 LLM calls per analysis
    </div>
  </div>
</div>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
# Injected via gr.Blocks(css=...) — applies identically on local AND HF Spaces.
# Do NOT pass css to .launch() — HF Spaces ignores launch() kwargs.

CSS = """
/* ── Agent report boxes: fixed height, always scrollable ── */
.report-box textarea,
.report-box .scroll-hide {
    font-family: 'Courier New', 'Consolas', monospace !important;
    font-size:   13px   !important;
    line-height: 1.65   !important;
    height:      420px  !important;
    min-height:  420px  !important;
    max-height:  420px  !important;
    overflow-y:  auto   !important;
    resize:      none   !important;
    white-space: pre-wrap !important;
}

/* Full Report tab — taller */
.full-report textarea,
.full-report .scroll-hide {
    height:     560px !important;
    min-height: 560px !important;
    max-height: 560px !important;
    overflow-y: auto  !important;
}

/* Status bar */
.status-bar > div {
    font-size:     13px;
    color:         #4b5563;
    padding:       6px 10px;
    background:    #f9fafb;
    border:        1px solid #e5e7eb;
    border-radius: 6px;
}

/* Hide Gradio footer */
footer { display: none !important; }
"""

TITLE_HTML = """
<div style="text-align:center;padding:20px 0 6px 0;">
  <h1 style="font-size:1.9rem;font-weight:700;color:#1e3a5f;margin:0;">
    🏥 Healthcare AI Clinical Decision Support
  </h1>
  <p style="color:#6b7280;margin-top:6px;font-size:0.9rem;">
    Multi-Agent System: Triage &rarr; Diagnosis &rarr; Treatment
    &nbsp;|&nbsp; LangGraph + Ollama (local) / HF Inference API (cloud)
  </p>
  <p style="color:#ef4444;font-size:0.78rem;margin-top:4px;">
    &#9888; For clinical decision support only.
    All outputs must be reviewed by a licensed healthcare professional.
  </p>
</div>
"""

# ── Backend helpers ────────────────────────────────────────────────────────────

def get_system_status() -> str:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=4)
        if resp.status_code != 200:
            return "🟡 Backend: Responding but unhealthy"
        h           = resp.json()
        device      = h.get("device_info", {})
        llm         = h.get("llm_backend") or h.get("ollama", {})
        device_name = device.get("device", "?").upper()
        backend_lbl = llm.get("backend", "?")
        model_lbl   = llm.get("model",   "?")
        if h.get("status") == "healthy":
            return f"🟢 Backend: Online | Device: {device_name} | LLM: {backend_lbl} ({model_lbl})"
        return f"🟡 Backend: Degraded | LLM ready: {llm.get('model_ready', False)}"
    except requests.exceptions.ConnectionError:
        return "🔴 Backend: Offline — make sure app.py is running"
    except Exception as e:
        return f"🟡 Backend: Error ({e})"


def run_analysis(name, age, gender, symptoms, vitals, history, medications, allergies):
    if not str(symptoms).strip():
        return ("⚠️ Please enter the patient's symptoms.", "", "", "", "")

    payload = {
        "name":        name        or "Anonymous",
        "age":         int(age)    if age else 0,
        "gender":      gender      or "Unknown",
        "symptoms":    symptoms,
        "vitals":      vitals      or "Not recorded",
        "history":     history     or "None reported",
        "medications": medications or "None",
        "allergies":   allergies   or "NKDA",
    }

    try:
        resp = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=300)

        if resp.status_code == 503:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            return (f"❌ Service unavailable: {detail}", "", "", "", "")

        if resp.status_code != 200:
            return (f"❌ Error {resp.status_code}: {resp.text}", "", "", "", "")

        data        = resp.json()
        agents_done = " → ".join(
            f"✅ {a.capitalize()}" for a in data.get("agents_completed", [])
        )
        return (
            f"**Pipeline:** {agents_done}",
            data.get("triage_output",    ""),
            data.get("diagnosis_output", ""),
            data.get("treatment_output", ""),
            data.get("final_report",     ""),
        )

    except requests.exceptions.ConnectionError:
        return ("❌ Cannot connect to backend. Make sure app.py is running.", "", "", "", "")
    except requests.exceptions.Timeout:
        return ("❌ Request timed out (>300s). The model may be overloaded.", "", "", "", "")
    except Exception as e:
        return (f"❌ Unexpected error: {e}", "", "", "", "")


# ── UI ─────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    # css here — works on BOTH local and HF Spaces
    with gr.Blocks(
        title="Healthcare AI",
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    ) as demo:

        gr.HTML(TITLE_HTML)

        # Status bar + buttons
        with gr.Row():
            status_md    = gr.Markdown(
                "*Checking backend status...*",
                elem_classes=["status-bar"],
            )
            refresh_btn  = gr.Button("🔄 Refresh",      size="sm", scale=0)
            workflow_btn = gr.Button("🔀 Show Workflow", size="sm", scale=0)

        demo.load(fn=get_system_status, outputs=status_md)
        refresh_btn.click(fn=get_system_status, outputs=status_md)

        # Workflow diagram — hidden by default
        workflow_html    = gr.HTML(value=WORKFLOW_HTML, visible=False)
        workflow_visible = gr.State(value=False)

        def _toggle(current):
            new_vis = not current
            btn_lbl = "🔀 Hide Workflow" if new_vis else "🔀 Show Workflow"
            return new_vis, gr.update(value=btn_lbl), gr.update(visible=new_vis)

        workflow_btn.click(
            fn=_toggle,
            inputs=[workflow_visible],
            outputs=[workflow_visible, workflow_btn, workflow_html],
        )

        with gr.Row(equal_height=False):

            # Left — patient inputs
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Patient Information")

                with gr.Row():
                    name_in   = gr.Textbox(label="Name",   placeholder="John Doe", scale=2)
                    age_in    = gr.Number( label="Age",    minimum=0, maximum=150, value=0, scale=1)
                    gender_in = gr.Dropdown(
                        label="Gender",
                        choices=["Male", "Female", "Non-binary", "Unknown"],
                        value="Unknown", scale=1,
                    )

                symptoms_in  = gr.Textbox(
                    label="🩺 Chief Complaint & Symptoms *",
                    placeholder="Describe all symptoms in detail...",
                    lines=4,
                )
                vitals_in    = gr.Textbox(
                    label="📊 Vital Signs",
                    placeholder="BP 120/80, HR 72, Temp 37C, SpO2 98%, RR 16",
                    lines=2,
                )
                history_in   = gr.Textbox(
                    label="📋 Medical History",
                    placeholder="Past diagnoses, surgeries, chronic conditions...",
                    lines=2,
                )
                with gr.Row():
                    meds_in      = gr.Textbox(
                        label="💊 Current Medications",
                        placeholder="Drug, dose, frequency...",
                        lines=2, scale=1,
                    )
                    allergies_in = gr.Textbox(
                        label="⚠️ Allergies",
                        placeholder="Drug/food allergies",
                        lines=2, scale=1,
                    )

                analyze_btn     = gr.Button("🔬 Run Clinical Analysis", variant="primary", size="lg")
                pipeline_status = gr.Markdown("*Pipeline status will appear here.*")

                gr.Markdown("### 📚 Example Cases")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[name_in, age_in, gender_in, symptoms_in,
                            vitals_in, history_in, meds_in, allergies_in],
                    label="Click to load an example patient",
                )

            # Right — agent outputs (all same fixed height, all scrollable)
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Clinical Report")

                with gr.Tab("🚨 Triage"):
                    triage_out = gr.Textbox(
                        label="Triage Agent",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["report-box"],
                    )
                with gr.Tab("🔬 Diagnosis"):
                    diagnosis_out = gr.Textbox(
                        label="Diagnosis Agent",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["report-box"],
                    )
                with gr.Tab("💊 Treatment"):
                    treatment_out = gr.Textbox(
                        label="Treatment Agent",
                        lines=20,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["report-box"],
                    )
                with gr.Tab("📋 Full Report"):
                    final_out = gr.Textbox(
                        label="Complete Clinical Report",
                        lines=26,
                        max_lines=26,
                        interactive=False,
                        elem_classes=["report-box", "full-report"],
                    )

        analyze_btn.click(
            fn=run_analysis,
            inputs=[name_in, age_in, gender_in, symptoms_in,
                    vitals_in, history_in, meds_in, allergies_in],
            outputs=[pipeline_status, triage_out, diagnosis_out, treatment_out, final_out],
            show_progress="full",
        )

        gr.Markdown(
            "<div style='text-align:center;color:#9ca3af;font-size:11px;padding:6px'>"
            "Healthcare AI | LangGraph + FastAPI + Gradio 6 | Not a replacement for clinical judgment"
            "</div>"
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Healthcare AI Frontend on :7860")
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True,
    )