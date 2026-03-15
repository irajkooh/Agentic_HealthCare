"""
Healthcare AI Frontend — Gradio UI
JS via gr.Button(js=...) — works in all Gradio versions without injection.
"""

import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from threading import Lock
from datetime import datetime
import gradio as gr

BACKEND_URL = "http://localhost:8000"
_HERE       = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(_HERE, "patients.db")

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

SEED_PATIENTS = [
    # (name, age, gender, symptoms, vitals, history, medications, allergies, days_ago)
    ("John Doe",        58, "Male",   "Crushing chest pain radiating to left arm, diaphoresis, nausea. Onset 45 min ago.",
     "BP 165/105, HR 102, SpO2 93%, RR 22, Temp 37.1°C",
     "Hypertension, Type 2 Diabetes, Smoker 20 pack-years",
     "Metformin 500mg BD, Lisinopril 10mg OD, Aspirin 75mg OD",
     "Penicillin", 480),

    ("Sarah Chen",      34, "Female", "Severe headache 10/10, photophobia, neck stiffness, fever 39.2°C, vomiting x3.",
     "BP 128/84, HR 112, SpO2 98%, RR 18, Temp 39.2°C",
     "No significant past history. Unvaccinated for meningococcal.",
     "Paracetamol 1g PRN",
     "None", 460),

    ("Michael Torres",  72, "Male",   "Progressive breathlessness over 3 days, orthopnoea, bilateral ankle swelling, reduced urine output.",
     "BP 148/92, HR 98, SpO2 88% on air, RR 28, Temp 36.8°C",
     "Ischaemic heart disease, AF, CKD stage 3, previous MI 2019",
     "Warfarin 3mg OD, Furosemide 40mg OD, Bisoprolol 5mg OD, Ramipril 5mg OD",
     "Aspirin (GI bleed)", 440),

    ("Emma Wilson",     28, "Female", "Fever 38.8°C for 3 days, painful urination, lower back pain, rigors.",
     "BP 110/70, HR 108, SpO2 99%, RR 16, Temp 38.8°C",
     "Recurrent UTIs. No other significant history.",
     "OCP",
     "Trimethoprim", 420),

    ("Robert Kim",      45, "Male",   "Sudden onset severe right-sided flank pain radiating to groin, nausea, haematuria.",
     "BP 142/88, HR 95, SpO2 99%, RR 18, Temp 37.0°C",
     "Previous kidney stones 2021. Gout.",
     "Allopurinol 100mg OD",
     "NSAIDs (severe gastritis)", 400),

    ("Aisha Patel",     52, "Female", "Sudden left-sided facial droop, slurred speech, right arm weakness. Onset 1 hour ago.",
     "BP 178/110, HR 88, SpO2 97%, RR 16, Temp 36.9°C",
     "Hypertension, Hypercholesterolaemia, Obesity BMI 33",
     "Amlodipine 10mg OD, Atorvastatin 40mg OD",
     "None", 380),

    ("David Nguyen",    67, "Male",   "Haematemesis x2 today (approximately 400ml), dizziness, black tarry stools for 2 days.",
     "BP 95/60, HR 118, SpO2 97%, RR 20, Temp 36.5°C",
     "Alcoholic liver disease, known varices, previous GI bleed 2022",
     "Propranolol 40mg BD, Lactulose 15ml TDS, Spironolactone 100mg OD",
     "None", 360),

    ("Lisa Park",       39, "Female", "Acute severe asthma, unable to complete sentences, accessory muscle use, silent chest on auscultation.",
     "BP 118/76, HR 128, SpO2 88%, RR 32, Peak flow 28% predicted",
     "Severe asthma since childhood, 2 previous ICU admissions",
     "Seretide 500/50 BD, Montelukast 10mg OD, Salbutamol PRN",
     "Aspirin (Samter's triad)", 340),

    ("James Okafor",    81, "Male",   "Confusion, reduced oral intake for 4 days, urinary incontinence, falls x2.",
     "BP 138/86, HR 90, SpO2 95%, RR 18, Temp 37.9°C",
     "Dementia, Hypertension, Osteoarthritis, Recurrent UTIs, Hearing loss",
     "Donepezil 10mg OD, Amlodipine 5mg OD, Paracetamol 500mg QDS",
     "Codeine (confusion)", 320),

    ("Maria Santos",    25, "Female", "Severe epigastric pain radiating to back, nausea, vomiting, pain worse after eating.",
     "BP 122/78, HR 105, SpO2 98%, RR 18, Temp 38.1°C",
     "Gallstones diagnosed 6 months ago. Social alcohol use.",
     "None",
     "None", 300),

    ("Thomas Brown",    55, "Male",   "Productive cough 10 days, fever, right-sided pleuritic chest pain, haemoptysis.",
     "BP 132/84, HR 102, SpO2 94%, RR 24, Temp 38.6°C",
     "COPD GOLD 2, Ex-smoker, Hypertension",
     "Tiotropium 18mcg OD, Salbutamol PRN, Amlodipine 5mg OD",
     "None", 280),

    ("Grace Lee",       19, "Female", "Severe lower abdominal pain, vaginal bleeding, last period 7 weeks ago, shoulder tip pain.",
     "BP 90/55, HR 124, SpO2 98%, RR 22, Temp 36.7°C",
     "No significant history. LMP 7 weeks ago.",
     "None",
     "None", 260),

    ("Henry Walsh",     63, "Male",   "Right leg swelling, erythema, warmth, calf tenderness. Long haul flight 48 hours ago.",
     "BP 136/82, HR 88, SpO2 96%, RR 18, Temp 37.3°C",
     "Prostate cancer on hormonal therapy, Obesity BMI 31",
     "Bicalutamide 150mg OD",
     "Heparin (HIT in 2020)", 240),

    ("Priya Sharma",    42, "Female", "Polyuria, polydipsia, weight loss 8kg over 3 months, fatigue, blurred vision.",
     "BP 118/72, HR 92, SpO2 99%, RR 16, Temp 36.6°C",
     "Family history of Type 1 Diabetes (mother).",
     "None",
     "None", 220),

    ("Carlos Mendez",   50, "Male",   "Acute confusion, rigidity, high fever 40.1°C, diaphoresis following recent antidepressant change.",
     "BP 155/95, HR 130, SpO2 96%, RR 24, Temp 40.1°C",
     "Depression, Hypertension. Recently started Linezolid and Sertraline.",
     "Sertraline 100mg OD (new), Linezolid 600mg BD (new), Ramipril 5mg OD",
     "None", 200),

    ("Eleanor Grant",   76, "Female", "Hip pain after fall, unable to weight bear, shortened and externally rotated left leg.",
     "BP 142/88, HR 84, SpO2 97%, RR 16, Temp 36.8°C",
     "Osteoporosis, Hypertension, Previous fragility fracture wrist 2021",
     "Alendronate 70mg weekly, Amlodipine 5mg OD, Calcium + Vit D",
     "None", 185),

    ("Samuel Adeyemi",  35, "Male",   "Generalised tonic-clonic seizure lasting 3 minutes, post-ictal confusion, no prior epilepsy.",
     "BP 148/92, HR 110, SpO2 95%, RR 20, Temp 37.2°C",
     "No known epilepsy. Heavy alcohol use. Sleep deprived.",
     "None",
     "None", 170),

    ("Rachel Cohen",    48, "Female", "Palpitations, heat intolerance, weight loss 5kg, tremor, exophthalmos, diarrhoea.",
     "BP 128/68, HR 118 irregular, SpO2 98%, RR 18, Temp 37.4°C",
     "No significant history.",
     "None",
     "Iodine contrast (anaphylaxis)", 155),

    ("William Foster",  69, "Male",   "Haematuria, dysuria, urinary frequency, weight loss, back pain for 6 weeks.",
     "BP 138/86, SpO2 98%, HR 78, RR 16, Temp 36.7°C",
     "Smoker 40 pack-years. Hypertension.",
     "Ramipril 5mg OD",
     "None", 140),

    ("Fatima Al-Rashid",31, "Female", "Severe joint pain and swelling both knees and wrists, butterfly rash, fever, hair loss, fatigue.",
     "BP 118/74, HR 96, SpO2 98%, RR 16, Temp 37.8°C",
     "No significant history. Family history of autoimmune disease.",
     "Ibuprofen 400mg TDS PRN",
     "Sulphonamides", 125),

    ("George Murphy",   57, "Male",   "Sudden painless loss of vision right eye, afferent pupillary defect, visual field defect.",
     "BP 158/96, HR 82, SpO2 98%, RR 16, Temp 36.8°C",
     "Hypertension, Hypercholesterolaemia, AF",
     "Warfarin 4mg OD, Atorvastatin 40mg OD, Bisoprolol 5mg OD",
     "None", 115),

    ("Nina Johansson",  22, "Female", "Severe anxiety, chest tightness, tingling hands and feet, hyperventilating, panic attack.",
     "BP 128/82, HR 130, SpO2 99%, RR 28, Temp 36.9°C",
     "Generalised anxiety disorder. Previous panic attacks.",
     "Sertraline 50mg OD",
     "None", 105),

    ("Patrick O'Brien", 74, "Male",   "Worsening breathlessness, cough, significant weight loss, haemoptysis, right-sided chest dullness.",
     "BP 128/78, SpO2 91% on air, HR 92, RR 26, Temp 37.1°C",
     "Ex-smoker 50 pack-years. Asbestos exposure (shipbuilder).",
     "Salbutamol PRN",
     "None", 95),

    ("Zoe Anderson",    29, "Female", "Thunderclap headache 10/10 sudden onset, worst headache of life, neck stiffness, photophobia.",
     "BP 152/96, HR 98, SpO2 99%, RR 18, Temp 37.0°C",
     "OCP use. Smoker.",
     "OCP",
     "None", 85),

    ("Hassan Ibrahim",  44, "Male",   "Jaundice, dark urine, pale stools, pruritus, RUQ pain, weight loss 7kg over 2 months.",
     "BP 118/76, HR 80, SpO2 98%, RR 16, Temp 36.9°C",
     "Type 2 Diabetes. Non-alcoholic fatty liver disease.",
     "Metformin 1g BD",
     "None", 75),

    ("Dorothy Hughes",  83, "Female", "Chest pain, vomiting, epigastric pain, new onset AF. Diabetic with atypical presentation.",
     "BP 138/88, HR 108 irregular, SpO2 94%, RR 22, Temp 37.0°C",
     "Type 2 Diabetes, Hypertension, Hypothyroidism, Previous NSTEMI",
     "Insulin glargine 20u ON, Metformin 500mg BD, Levothyroxine 100mcg OD, Amlodipine 5mg OD",
     "Statin (myopathy)", 65),

    ("Liam Carter",     38, "Male",   "Sudden severe back pain radiating to abdomen, pulsatile abdominal mass, hypotension.",
     "BP 85/50, HR 128, SpO2 96%, RR 24, Temp 36.6°C",
     "Hypertension, Smoker. Family history of AAA (father).",
     "Amlodipine 10mg OD",
     "None", 55),

    ("Amelia Jackson",  16, "Female", "Paracetamol overdose 30 tablets 4 hours ago, expressed suicidal ideation.",
     "BP 112/72, HR 88, SpO2 99%, RR 16, Temp 36.8°C",
     "Depression, previous self-harm.",
     "Fluoxetine 20mg OD",
     "None", 48),

    ("Frank Mueller",   61, "Male",   "Diabetic foot ulcer with spreading cellulitis, fever, crepitus on palpation, foul odour.",
     "BP 142/88, HR 104, SpO2 96%, RR 20, Temp 38.9°C",
     "Type 2 Diabetes poorly controlled, PVD, Obesity BMI 35",
     "Insulin aspart TDS, Insulin glargine 30u ON, Metformin 1g BD",
     "Metronidazole (disulfiram reaction)", 42),

    ("Isabelle Dupont", 33, "Female", "32 weeks pregnant, severe headache, visual disturbance, facial oedema, epigastric pain.",
     "BP 168/110, HR 94, SpO2 98%, RR 18, Temp 36.9°C",
     "G2P1. Previous pregnancy-induced hypertension.",
     "Folic acid, Iron supplementation, Low dose aspirin",
     "None", 36),

    ("Alan Richards",   70, "Male",   "Falls x3 past month, dizziness on standing, postural hypotension, increasing confusion.",
     "BP 158/90 lying → 118/70 standing, HR 76, SpO2 96%, Temp 37.0°C",
     "Parkinson's disease, Hypertension, Benign prostatic hypertrophy",
     "Levodopa/Carbidopa 250/25 TDS, Tamsulosin 400mcg OD, Ramipril 5mg OD, Temazepam 10mg ON",
     "None", 30),

    ("Sophie Turner",   27, "Female", "First episode psychosis, auditory hallucinations, paranoid delusions, disorganised behaviour, not sleeping for 5 days.",
     "BP 122/78, HR 102, SpO2 99%, RR 18, Temp 37.1°C",
     "Cannabis use. Family history of schizophrenia (brother).",
     "None",
     "None", 25),

    ("Kwame Asante",    46, "Male",   "Sickle cell crisis, severe bone pain both legs and back, fever, dehydration.",
     "BP 118/72, HR 112, SpO2 92%, RR 22, Temp 38.4°C",
     "Sickle cell disease HbSS. Recurrent vaso-occlusive crises. Splenectomy 2015.",
     "Hydroxyurea 500mg OD, Folic acid 5mg OD, Penicillin V 250mg BD",
     "None", 20),

    ("Mei Lin",         53, "Female", "Acute angle closure glaucoma, severe right eye pain, headache, nausea, vomiting, red eye, halos.",
     "BP 148/92, HR 96, SpO2 98%, RR 18, Temp 37.0°C",
     "Hypermetropia. Hypertension.",
     "Amlodipine 5mg OD",
     "Sulphonamides", 16),

    ("Oliver Stone",    40, "Male",   "Burns to face, neck, both arms after house fire. Singed nasal hairs, hoarse voice, carbonaceous sputum.",
     "BP 118/76, HR 118, SpO2 95%, RR 24, Temp 36.4°C",
     "No significant history.",
     "None",
     "None", 14),

    ("Chloe Martin",    24, "Female", "MDMA ingestion at music festival, hyperthermia, agitation, tachycardia, clenched jaw.",
     "BP 168/102, HR 148, SpO2 97%, RR 22, Temp 40.8°C",
     "No significant history.",
     "None (illicit MDMA)",
     "None", 12),

    ("Bernard Collins", 77, "Male",   "Ruptured Baker's cyst vs DVT, sudden onset posterior knee pain with swelling, cannot walk.",
     "BP 148/88, HR 82, SpO2 97%, RR 16, Temp 37.0°C",
     "Rheumatoid arthritis, Hypertension, Atrial fibrillation",
     "Methotrexate 15mg weekly, Folic acid 5mg weekly, Warfarin 3mg OD, Amlodipine 5mg OD",
     "None", 10),

    ("Diana Foster",    36, "Female", "Anaphylaxis post penicillin injection at GP, urticaria, angioedema, stridor, hypotension.",
     "BP 78/48, HR 142, SpO2 91%, RR 30, Temp 37.2°C",
     "Known penicillin allergy not documented correctly.",
     "OCP, Cetirizine PRN",
     "Penicillin (anaphylaxis — this admission)", 8),

    ("Marcus Webb",     49, "Male",   "Hyperglycaemic hyperosmolar state, BG 42 mmol/L, profoundly dehydrated, drowsy GCS 12.",
     "BP 102/64, HR 118, SpO2 97%, RR 22, Temp 37.6°C",
     "Type 2 Diabetes poorly controlled. Non-compliant with medications.",
     "Metformin 1g BD (stopped by patient), Gliclazide 80mg BD (stopped by patient)",
     "None", 7),

    ("Yuki Tanaka",     31, "Male",   "Spontaneous pneumothorax, sudden onset left chest pain and breathlessness, tall thin male.",
     "BP 122/76, HR 108, SpO2 93%, RR 26, Temp 36.8°C",
     "Marfan syndrome. Previous right spontaneous pneumothorax 2022.",
     "None",
     "None", 6),

    ("Rose Kelly",      68, "Female", "Bowel obstruction, absolute constipation 5 days, colicky central abdominal pain, vomiting.",
     "BP 132/82, HR 96, SpO2 97%, RR 18, Temp 37.3°C",
     "Previous colorectal cancer with hemi-colectomy 2019. Adhesions.",
     "Lactulose 15ml BD, Senna 2 tabs ON",
     "None", 5),

    ("Anthony Price",   55, "Male",   "Septic shock secondary to pneumonia, confusion, mottled skin, not passed urine for 12 hours.",
     "BP 78/44, HR 132, SpO2 88% on 15L O2, RR 34, Temp 39.4°C",
     "Type 2 Diabetes, Immunosuppressed post renal transplant",
     "Tacrolimus 3mg BD, Prednisolone 5mg OD, Metformin 500mg BD, Trimethoprim prophylaxis",
     "Penicillin", 4),

    ("Naomi Clarke",    43, "Female", "Acute liver failure, jaundice, coagulopathy, encephalopathy grade 2, paracetamol overdose 48h ago.",
     "BP 98/62, HR 108, SpO2 96%, RR 20, Temp 37.8°C",
     "Depression. Alcohol misuse.",
     "Fluoxetine 40mg OD",
     "None", 3),

    ("Peter Hammond",   79, "Male",   "COPD exacerbation triggered by viral URTI, increased sputum, wheeze, severe dyspnoea.",
     "BP 138/86, HR 104, SpO2 82% on air, RR 32, Temp 37.7°C",
     "COPD GOLD 3, cor pulmonale, OSA on home CPAP, Hypertension",
     "Tiotropium 18mcg OD, Seretide 500/50 BD, Salbutamol PRN, Furosemide 40mg OD, Home O2 1L/min",
     "Theophylline (previous toxicity)", 3),

    ("Ingrid Larsson",  38, "Female", "Pulmonary embolism, pleuritic chest pain, haemoptysis, right leg swelling, post-partum 6 weeks.",
     "BP 108/68, HR 122, SpO2 90%, RR 28, Temp 37.2°C",
     "6 weeks post Caesarean section. No prior VTE.",
     "Iron supplementation",
     "None", 2),

    ("Derek Osman",     60, "Male",   "Melaena, haematemesis, known cirrhosis, spider naevi, palmar erythema, caput medusae.",
     "BP 88/52, HR 136, SpO2 96%, RR 22, Temp 37.1°C",
     "Alcoholic cirrhosis Child-Pugh C, Previous variceal bleed 2023, Hepatic encephalopathy",
     "Propranolol 40mg BD, Lactulose TDS, Rifaximin 550mg BD, Spironolactone 100mg OD",
     "None", 2),

    ("Lynda Moore",     47, "Female", "Rhabdomyolysis post prolonged immobility after fall at home, severe muscle pain, dark urine, AKI.",
     "BP 118/74, HR 104, SpO2 98%, RR 18, Temp 37.4°C",
     "Alcohol dependence. Found on floor after 2 days.",
     "None",
     "None", 1),

    ("Victor Chen",     64, "Male",   "Acute aortic dissection type A, tearing chest pain radiating to back, BP differential both arms > 30mmHg.",
     "BP right arm 188/110, left arm 142/88, HR 108, SpO2 95%, RR 22, Temp 36.9°C",
     "Hypertension uncontrolled, Bicuspid aortic valve, Marfan features",
     "Amlodipine 10mg OD (non-compliant)",
     "None", 1),

    ("Sandra Mitchell", 58, "Female", "Cellulitis right leg with lymphangitis, spreading erythema, red streaking up thigh, rigors, high fever.",
     "BP 118/74, HR 116, SpO2 97%, RR 20, Temp 39.1°C",
     "Lymphoedema, Obesity BMI 38, Type 2 Diabetes",
     "Metformin 1g BD, Compression hosiery",
     "Flucloxacillin (rash)", 1),

    ("Raj Patel",       52, "Male",   "Acute gout attack right big toe, exquisitely tender, swollen, erythematous, cannot weight bear.",
     "BP 148/92, HR 84, SpO2 99%, RR 16, Temp 37.6°C",
     "Hypertension, Chronic kidney disease stage 2, Obesity",
     "Amlodipine 5mg OD, Allopurinol 300mg OD",
     "NSAIDs (CKD)", 1),
]


def init_db():
    """
    Create DB schema.
    Seeding rule:
      - If the DB file does not exist            → seed 50 demo patients.
      - If the DB file exists but has 0 rows     → seed 50 demo patients
        (handles blank placeholder files committed to git / HF Spaces repos).
      - If the DB file exists and has rows       → load as-is, do NOT modify.
    """
    from datetime import timedelta

    # Must check existence BEFORE sqlite3.connect, which creates the file.
    file_existed = os.path.exists(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, age INTEGER, gender TEXT,
            symptoms TEXT, vitals TEXT, history TEXT,
            medications TEXT, allergies TEXT,
            created_at TEXT DEFAULT (strftime('%Y-%m-%d %H:%M:%S','now','localtime'))
        )
    """)
    try:
        con.execute("ALTER TABLE patients ADD COLUMN created_at TEXT")
    except Exception:
        pass
    con.commit()

    count = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    if count == 0:
        # Either brand-new file or an empty placeholder — seed either way
        reason = "existing empty file" if file_existed else "new file"
        print(f"[DB] {reason.capitalize()} — seeding {len(SEED_PATIENTS)} demo patients into {DB_PATH}")
        # Reset AUTOINCREMENT counter so ids always start from 1
        try:
            con.execute("DELETE FROM sqlite_sequence WHERE name='patients'")
            con.commit()
        except Exception:
            pass
        base = datetime.now()
        for p in SEED_PATIENTS:
            name, age, gender, symptoms, vitals, history, meds, allergies, days_ago = p
            ts = (base - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
            con.execute(
                "INSERT INTO patients "
                "(name,age,gender,symptoms,vitals,history,medications,allergies,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (name, age, gender, symptoms, vitals, history, meds, allergies, ts)
            )
        con.commit()
        print(f"[DB] Seeded {len(SEED_PATIENTS)} patients OK.")
    else:
        print(f"[DB] Existing database — {count} patients loaded from {DB_PATH}")

    con.close()

def db_load_all():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM patients ORDER BY id").fetchall()]
    con.close(); return rows

def db_add_patient(name, age, gender, symptoms, vitals, history, medications, allergies):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO patients (name,age,gender,symptoms,vitals,history,medications,allergies,created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (name, int(age) if age else 0, gender, symptoms, vitals, history, medications, allergies,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    con.commit()
    _db_reindex(con)
    con.close()

def _db_reindex(con):
    """Renumber all patient IDs to 1..N in current order, reset AUTOINCREMENT."""
    rows = con.execute("SELECT id FROM patients ORDER BY id").fetchall()
    for new_id, (old_id,) in enumerate(rows, start=1):
        if new_id != old_id:
            con.execute("UPDATE patients SET id=? WHERE id=?", (new_id, old_id))
    con.execute("DELETE FROM sqlite_sequence WHERE name='patients'")
    remaining = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if remaining > 0:
        con.execute(
            "INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES ('patients', ?)",
            (remaining,)
        )
    con.commit()

def db_delete_patient(pid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM patients WHERE id=?", (pid,))
    con.commit()
    _db_reindex(con)
    con.close()
    con.close()

def db_delete_all():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM patients")
    con.execute("DELETE FROM sqlite_sequence WHERE name='patients'")
    con.commit(); con.close()

_db_lock = Lock()

# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────



TITLE_HTML = """
<div style="text-align:center;padding:4px 0 0 0;">
  <h1 style="font-size:1.35rem;font-weight:700;color:#1e3a5f;margin:0;">🏥 Healthcare AI Clinical Decision Support</h1>
</div>
"""

def _render_text(text: str) -> str:
    """
    Shared renderer: converts LLM markdown-ish output into clean HTML.
    Handles every pattern the LLM produces:
      ===TITLE=== / === TITLE      → dark blue header bar
      ---                          → thin divider
      ## Heading / # Heading       → bold section heading
      **LABEL**: rest              → blue left-bar label
      **bold** inline              → <strong>
      *italic* inline              → <em>
      - bullet / • bullet          → indented bullet
      1. numbered item             → numbered list
      blank line                   → spacer
      plain prose                  → paragraph
    """
    import html as _html, re

    def inline(raw):
        s = _html.escape(raw)
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', s)
        return s

    out = []
    for line in text.split('\n'):
        s = line.strip()

        # Pure divider: --- or ===
        if re.match(r'^-{3,}$', s) or re.match(r'^={3,}$', s):
            out.append('<hr style="border:none;border-top:1px solid #d1d5db;margin:8px 0 6px;">')
            continue

        # === TITLE === or === TITLE
        if s.startswith('='):
            title = re.sub(r'^=+\s*', '', s).rstrip('= ').strip()
            if title:
                out.append(
                    '<div style="margin:16px 0 6px;padding:7px 14px;background:#1e3a5f;'
                    'color:#fff;border-radius:6px;font-weight:700;font-size:0.88rem;'
                    'letter-spacing:0.04em;text-transform:uppercase;">'
                    + _html.escape(title) + '</div>'
                )
            continue

        # ## Heading or # Heading
        m = re.match(r'^(#{1,3})\s+(.+)', s)
        if m:
            lvl = len(m.group(1))
            sizes = {1: '1.0rem', 2: '0.92rem', 3: '0.86rem'}
            mg    = {1: '14px',  2: '12px',     3: '10px'}
            out.append(
                f'<div style="margin:{mg[lvl]} 0 4px;font-weight:700;'
                f'font-size:{sizes[lvl]};color:#1e3a5f;">'
                + inline(m.group(2)) + '</div>'
            )
            continue

        # **LABEL**: rest  (line starts with a bold label)
        m = re.match(r'^\*\*([^*]+)\*\*\s*:?\s*(.*)', s)
        if m:
            label = m.group(1).strip()
            rest  = m.group(2).strip()
            out.append(
                '<div style="margin:10px 0 3px;padding:5px 10px;background:#eef2f9;'
                'border-left:3px solid #1e3a5f;border-radius:0 4px 4px 0;'
                'font-weight:700;font-size:0.86rem;color:#1e3a5f;">'
                + _html.escape(label)
                + (': <span style="font-weight:400;color:#111827;">' + inline(rest) + '</span>' if rest else '')
                + '</div>'
            )
            continue

        # Bullet: -, •, *  (not followed by bold)
        m = re.match(r'^[-•*]\s+(.+)', s)
        if m and not s.startswith('**'):
            out.append(
                '<div style="display:flex;gap:8px;margin:3px 0 3px 12px;align-items:baseline;">'
                '<span style="color:#1e3a5f;font-weight:700;flex-shrink:0;font-size:0.9rem;">•</span>'
                '<span style="flex:1;">' + inline(m.group(1)) + '</span></div>'
            )
            continue

        # Numbered list: 1. item
        m = re.match(r'^(\d+)\.\s+(.+)', s)
        if m:
            out.append(
                '<div style="display:flex;gap:8px;margin:3px 0 3px 12px;align-items:baseline;">'
                '<span style="color:#1e3a5f;font-weight:700;flex-shrink:0;min-width:20px;">'
                + m.group(1) + '.</span>'
                '<span style="flex:1;">' + inline(m.group(2)) + '</span></div>'
            )
            continue

        # Empty line → small spacer
        if not s:
            out.append('<div style="height:6px;"></div>')
            continue

        # Plain prose
        out.append('<div style="margin:2px 0 2px 0;">' + inline(s) + '</div>')

    return '\n'.join(out)


def make_report_html(text: str, full: bool = False) -> str:
    def make_report_markdown(text: str) -> str:
        """
        Generate and normalize clinical report as Markdown for easy copy/paste.
        Uses the same normalization as chat answers.
        """
        return clean_for_chat(text)
    """Wrap _render_text in a scrollable report container."""
    height = '900px' if full else '700px'
    wrap = (
        f'width:100%;height:{height};overflow-y:auto;overflow-x:hidden;'
        'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
        'font-size:13.5px;line-height:1.75;background:#f9fafb;'
        'border:1px solid #e5e7eb;border-radius:8px;'
        'padding:16px 18px;box-sizing:border-box;color:#111827;'
    )
    if not text or not text.strip():
        return (f'<div style="{wrap}"><span style="color:#9ca3af;font-style:italic;">'
                'Output will appear here after analysis.</span></div>')
    # Apply the same normalization as chat answers
    text = clean_for_chat(text)
    return f'<div style="{wrap}">{_render_text(text)}</div>'



def clean_for_chat(text: str) -> str:
    """
    Normalise LLM output into clean Markdown for gr.Chatbot (which renders MD natively).
    - Converts ===TITLE=== → ## TITLE  (so Gradio renders a heading)
    - Strips raw === / --- divider lines (Gradio renders --- as <hr> already)
    - Leaves **bold**, *italic*, bullets, numbered lists, and prose untouched
    - Collapses 3+ consecutive blank lines to 2
    """
    import re
    lines = text.split('\n')
    out = []
    for line in lines:
        s = line.strip()
        # Pure divider lines — keep as markdown hr
        if re.match(r'^-{3,}$', s) or re.match(r'^={3,}$', s):
            out.append('---')
            continue
        # === TITLE === or === TITLE  → ## TITLE
        if s.startswith('='):
            title = re.sub(r'^=+\s*', '', s).rstrip('= ').strip()
            if title:
                out.append(f'## {title}')
            continue
        # Everything else unchanged
        out.append(line)
    # Collapse 3+ blank lines → 1 blank line
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_system_status() -> str:
    import os as _os
    env = "🌐 HF Space" if _os.environ.get("SYSTEM") == "spaces" or _os.environ.get("SPACE_ID") else "💻 Local"
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=4)
        if resp.status_code != 200:
            return f'<div style="text-align:center;font-size:0.78rem;margin:0;padding:0 0 3px 0;">{env} | 🟡 Backend unhealthy</div>'
        h      = resp.json()
        device = h.get("device_info", {}).get("device", "?").upper()
        llm    = h.get("llm_backend") or h.get("ollama", {})
        model  = llm.get("model", "")
        llm_str = f"{llm.get('backend','?')} ({model})" if model else llm.get("backend", "?")
        status = "🟢" if h.get("status") == "healthy" else "🟡"
        return f'<div style="text-align:center;font-size:0.78rem;margin:0;padding:0 0 3px 0;">{env} | {status} Device: {device} | LLM: <span style="color:#1d4ed8;font-weight:600;">{llm_str}</span></div>'
    except requests.exceptions.ConnectionError:
        return f'<div style="text-align:center;font-size:0.78rem;margin:0;padding:0 0 3px 0;">{env} | 🔴 Backend offline — run app.py</div>'
    except Exception as e:
        return f'<div style="text-align:center;font-size:0.78rem;margin:0;padding:0 0 3px 0;">{env} | 🟡 Error ({e})</div>'

# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

CHAT_SYSTEM_WITH_ANALYSIS = (
    "You are a clinical decision support assistant.\n"
    "Answer concisely and clinically.\n"
    "Format all answers and clinical reports in readable, visually appealing forms.\n"
    "Use clear section headings, bullet points, concise paragraphs, and consistent formatting.\n"
    "Make reports easy to copy and review.\n"
    "CURRENT PATIENT:\n"
    "  Name       : {analyzed_name}\n"
    "  Age        : {analyzed_age}\n"
    "  Gender     : {analyzed_gender}\n"
    "  Symptoms   : {analyzed_symptoms}\n"
    "  Vitals     : {analyzed_vitals}\n"
    "  History    : {analyzed_history}\n"
    "  Medications: {analyzed_medications}\n"
    "  Allergies  : {analyzed_allergies}\n\n"
    "{analysis_context}\n"
    "Use the CURRENT PATIENT details and PROCESSED ANALYSIS above.\n"
    "Never say a field is Unknown if it is listed above.\n"
)

CHAT_SYSTEM_NO_ANALYSIS = (
    "You are a clinical decision support assistant.\n"
    "Answer concisely and clinically. Plain text ONLY.\n\n"
    "{patients_context}"
    "PROCESSED ANALYSIS: None yet.\n\n"
    "RULES:\n"
    "- No patient has been analysed yet.\n"
    "- For general questions (how can you help, patient count, system info) answer freely.\n"
    "- For any patient-specific or clinical question, reply exactly: "
    "'Please select a patient and click Load & Analyse first.'\n"
)

# Clinical keywords — if any appear in the message and no analysis exists,
# short-circuit in Python before the LLM is even called.
_CLINICAL_KEYWORDS = (
    "diagnos", "treatment", "triage", "symptom", "vital", "medication", "drug",
    "allerg", "medical history", "prognos", "urgent", "condition", "disease",
    "clinical report", "finding", "prescription", "dose", "interact",
    "lab result", "scan", "xray", "x-ray", "ecg", "ekg",
    "blood pressure", "heart rate", "chest pain", "fever",
    "tell me about this patient", "what is wrong with",
    "full report", "triage report", "diagnosis report", "treatment plan",
    "analyse", "analyze",
)

# General questions that must always pass through regardless of keywords
_GENERAL_PATTERNS = (
    "how many", "how can you", "what can you", "what is this", "who are you",
    "list patient", "list all", "show patient", "show all",
)

def build_chat_system(patient_state, all_patients):
    has_analysis = bool(
        patient_state
        and patient_state.get("triage_output")
        and patient_state.get("analyzed_name")
    )

    if has_analysis:
        analyzed_name = patient_state.get("analyzed_name", "")
        analysis_ctx = (
            f"PROCESSED ANALYSIS — Patient: {analyzed_name.upper()}\n\n"
            f"TRIAGE:\n{patient_state.get('triage_output','')}\n\n"
            f"DIAGNOSIS:\n{patient_state.get('diagnosis_output','')}\n\n"
            f"TREATMENT:\n{patient_state.get('treatment_output','')}\n"
        )
        return CHAT_SYSTEM_WITH_ANALYSIS.format(
            analyzed_name       = analyzed_name,
            analyzed_age        = patient_state.get("analyzed_age",        "Unknown"),
            analyzed_gender     = patient_state.get("analyzed_gender",     "Unknown"),
            analyzed_symptoms   = patient_state.get("analyzed_symptoms",   ""),
            analyzed_vitals     = patient_state.get("analyzed_vitals",     ""),
            analyzed_history    = patient_state.get("analyzed_history",    ""),
            analyzed_medications= patient_state.get("analyzed_medications",""),
            analyzed_allergies  = patient_state.get("analyzed_allergies",  ""),
            analysis_context    = analysis_ctx,
        )
    else:
        if all_patients:
            lines = [f"PATIENT LIST ({len(all_patients)} total):\n"]
            for i, p in enumerate(all_patients, 1):
                lines.append(
                    f"  {i}. {p.get('name','?')}, Age {p.get('age','?')}, "
                    f"{p.get('gender','?')}\n"
                )
            patients_ctx = "\n".join(lines) + "\n\n"
        else:
            patients_ctx = "No patients in database.\n\n"
        return CHAT_SYSTEM_NO_ANALYSIS.format(patients_context=patients_ctx)

def chat_respond(message, history, patient_state, all_patients):
    if not isinstance(history, list):
        history = []
    if not message or not message.strip():
        return history, ""

    # Python-level guard: if no processed analysis and message is clinical → short-circuit
    has_analysis = bool(
        patient_state
        and patient_state.get("triage_output")
        and patient_state.get("analyzed_name")
    )
    if not has_analysis:
        msg_lower = message.lower()
        is_general = any(p in msg_lower for p in _GENERAL_PATTERNS)
        is_clinical = not is_general and any(kw in msg_lower for kw in _CLINICAL_KEYWORDS)
        if is_clinical:
            answer = "Please select a patient and click Load & Analyse first."
            return history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": answer},
            ], ""

    system_prompt = build_chat_system(patient_state, all_patients or [])
    def flatten_content(content):
        # If content is a dict or list, extract 'text' or 'content' recursively
        if isinstance(content, dict):
            return content.get("text") or content.get("content") or str(content)
        elif isinstance(content, list):
            return " ".join(flatten_content(item) for item in content)
        return str(content)

    # Build clean history:
    # 1. Drop redirect messages (and the user turn that triggered them)
    # 2. Compact old turns into a summary when history grows large,
    #    to avoid exceeding the model context window.
    _REDIRECT = "Please select a patient and click Load & Analyse first."
    _MAX_RECENT = 8   # keep last 8 turns (4 exchanges) verbatim

    clean_history = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in ("user", "assistant"):
            continue
        txt = flatten_content(msg.get("content", ""))
        if txt.strip() == _REDIRECT:
            if clean_history and clean_history[-1]["role"] == "user":
                clean_history.pop()
            continue
        clean_history.append({"role": msg["role"], "content": txt})

    # Summarize previous conversation after every answer
    if clean_history:
        summary_lines = []
        for m in clean_history:
            role = "Physician" if m["role"] == "user" else "Assistant"
            summary_lines.append(f"{role}: {m['content'][:200]}")
        summary = "Previous conversation summary:\n" + "\n".join(summary_lines)
        clean_history = [{"role": "assistant", "content": summary}]

    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in clean_history:
        llm_messages.append(msg)
    llm_messages.append({"role": "user", "content": flatten_content(message)})
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json={"messages": llm_messages}, timeout=120)
        if resp.status_code == 200:
            raw = resp.json().get("response", "No response.")
            # Defensively extract plain text regardless of what the backend returns:
            # - plain string  → use directly
            # - content block dict  e.g. {'text': '...', 'type': 'text'}  → extract 'text'
            # - list of content blocks  → join all text blocks
            if isinstance(raw, str):
                answer = raw
            elif isinstance(raw, dict):
                answer = raw.get("text") or raw.get("content") or str(raw)
            elif isinstance(raw, list):
                answer = " ".join(
                    (item.get("text") or item.get("content") or "")
                    for item in raw if isinstance(item, dict)
                ).strip() or "No response."
            else:
                answer = str(raw)
            # Last-resort: if answer still looks like a raw dict/list repr, try to parse it
            if isinstance(answer, str) and answer.startswith("{") and "'text'" in answer:
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(answer)
                    if isinstance(parsed, dict):
                        answer = parsed.get("text") or parsed.get("content") or answer
                except Exception:
                    pass
        else:
            answer = f"Backend error {resp.status_code}"
    except requests.exceptions.ConnectionError:
        answer = "Cannot connect to backend."
    except Exception as e:
        answer = f"Error: {e}"
    # Normalise to clean markdown so gr.Chatbot renders headings/bold/bullets properly
    if isinstance(answer, str) and not answer.startswith("Error") and not answer.startswith("Cannot"):
        answer = clean_for_chat(answer)

    # FINAL DEFENSIVE FLATTENING: ensure assistant reply is always a string
    def flatten_final(content):
        if isinstance(content, dict):
            return content.get("text") or content.get("content") or str(content)
        elif isinstance(content, list):
            return " ".join(flatten_final(item) for item in content)
        return str(content)

    return history + [
        {"role": "user",      "content": flatten_final(message)},
        {"role": "assistant", "content": flatten_final(answer)},
    ], ""

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(name, age, gender, symptoms, vitals, history, medications, allergies):
    if not str(symptoms or "").strip():
        e = make_report_html("Please load a patient first.")
        return ("No patient loaded.", e, e, e, e, {})
    payload = {
        "name": name or "Anonymous", "age": int(age) if age else 0,
        "gender": gender or "Unknown", "symptoms": symptoms,
        "vitals": vitals or "Not recorded", "history": history or "None reported",
        "medications": medications or "None", "allergies": allergies or "NKDA",
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=300)
        if resp.status_code != 200:
            e = make_report_html(f"Error {resp.status_code}")
            return (f"Error {resp.status_code}", e, e, e, e, {})
        data = resp.json()
        done = " → ".join(f"✅ {a.capitalize()}" for a in data.get("agents_completed", []))
        state = {
            "analyzed_name":    name or "Anonymous",
            "analyzed_age":     age  or "Unknown",
            "analyzed_gender":  gender or "Unknown",
            "analyzed_symptoms":    symptoms    or "",
            "analyzed_vitals":      vitals      or "",
            "analyzed_history":     history     or "",
            "analyzed_medications": medications or "",
            "analyzed_allergies":   allergies   or "",
            "triage_output":    data.get("triage_output",    ""),
            "diagnosis_output": data.get("diagnosis_output", ""),
            "treatment_output": data.get("treatment_output", ""),
            "final_report":     data.get("final_report",     ""),
        }
        return (
            f"**Pipeline:** {done}",
            make_report_html(state["triage_output"]),
            make_report_html(state["diagnosis_output"]),
            make_report_html(state["treatment_output"]),
            make_report_html(state["final_report"], full=True),
            state,
        )
    except requests.exceptions.ConnectionError:
        e = make_report_html("Cannot connect to backend.")
        return ("Cannot connect to backend.", e, e, e, e, {})
    except Exception as ex:
        e = make_report_html(f"Error: {ex}")
        return (f"Error: {ex}", e, e, e, e, {})

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI extra route
# ─────────────────────────────────────────────────────────────────────────────

def register_routes(fastapi_app):
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @fastapi_app.post("/add-patient")
    async def add_patient_route(request: Request):
        try:
            data = await request.json()
            name = (data.get("name") or "").strip()
            if not name:
                return JSONResponse({"ok": False, "error": "Name required"}, status_code=400)
            with _db_lock:
                db_add_patient(
                    name, int(data.get("age") or 0),
                    (data.get("gender") or "Unknown").strip(),
                    (data.get("symptoms") or "").strip(),
                    (data.get("vitals") or "").strip(),
                    (data.get("history") or "").strip(),
                    (data.get("medications") or "").strip(),
                    (data.get("allergies") or "").strip(),
                )
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ─────────────────────────────────────────────────────────────────────────────
# Sample questions
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    "How can you help me?",
    "Tell me about this patient.",
    "What doctor this patient should see?",
    "Address of doctors the patient can refer in Chicago?",
    "Tell me about the patient's triage report.",
    "Tell me about the patient's diagnosis report.",
    "Tell me about the patient's treatment report.",
    "What are the side effects of the patient's treatement?",
    "Summarize the patient's full report in max 10 bullet points.",
    "What is the main clinical concern of this patient?",
    "List the required Lab works for this patient, and why?",
    "What is the recommended next steps for this patient?",
]

# ─────────────────────────────────────────────────────────────────────────────
# JS helpers — all used via gr.Button(fn=None, js=...)
# Each is a self-contained JS function string "() => { ... return []; }"
# This is the ONLY reliable method — no injection, no sanitisation, Gradio 3/4/5/6.
# ─────────────────────────────────────────────────────────────────────────────

# Shared voice-selection JS snippets (inlined into each function)
# American female: Samantha (macOS) > Zira (Windows) > any en-US non-male > any en-US
_VOICE_FEMALE = """
    var _vs = window.speechSynthesis.getVoices();
    var _v = _vs.find(function(x){ return x.name === 'Samantha'; })
          || _vs.find(function(x){ return /Zira/.test(x.name) && /US/.test(x.lang); })
          || _vs.find(function(x){ return x.lang === 'en-US' && /female/i.test(x.name); })
          || _vs.find(function(x){ return x.lang === 'en-US' && !/male/i.test(x.name); })
          || _vs.find(function(x){ return x.lang === 'en-US'; })
          || _vs.find(function(x){ return /en/i.test(x.lang); });
    if (_v) u.voice = _v;
"""

# American male: Alex (macOS) > David (Windows) > any en-US male > any en-US
_VOICE_MALE = """
    var _vs = window.speechSynthesis.getVoices();
    var _v = _vs.find(function(x){ return x.name === 'Alex'; })
          || _vs.find(function(x){ return /David/.test(x.name) && /US/.test(x.lang); })
          || _vs.find(function(x){ return x.lang === 'en-US' && /male/i.test(x.name) && !/female/i.test(x.name); })
          || _vs.find(function(x){ return x.lang === 'en-US'; })
          || _vs.find(function(x){ return /en/i.test(x.lang); });
    if (_v) u.voice = _v;
"""

# TreeWalker text extractor — skips buttons/scripts so only report text is read
_EXTRACT_TEXT = """
  function extractText(el) {
    var skip = new Set(['SCRIPT','STYLE','BUTTON','NOSCRIPT','SVG','PATH']);
    var tw = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
      acceptNode: function(node) {
        var p = node.parentElement;
        while (p) { if (skip.has(p.tagName)) return NodeFilter.FILTER_REJECT; p = p.parentElement; }
        return NodeFilter.FILTER_ACCEPT;
      }
    }, false);
    var parts = []; var n;
    while ((n = tw.nextNode())) { var t = n.nodeValue.trim(); if (t) parts.push(t); }
    return parts.join(' ');
  }
"""


def _tts_report_js(elem_id: str) -> str:
    """
    Read a report panel aloud using an American female voice.
    elem_id must match the Gradio elem_id of the gr.HTML component.
    Gradio renders: <div id="<elem_id>">...<div class="svelte...">...HTML content...</div></div>
    We search for the elem_id div, then look inside it for the actual content div.
    """
    return (
        "() => {\n"
        "  var EID = '" + elem_id + "';\n"
        "  var PLAY = '🔊 Read Aloud';\n"
        "  var STOP = '⏹ Stop';\n"
        "\n"
        "  // Toggle off\n"
        "  if (window._hcai_rpt_speaking === EID) {\n"
        "    window.speechSynthesis && window.speechSynthesis.cancel();\n"
        "    window._hcai_rpt_speaking = null;\n"
        "    document.querySelectorAll('button').forEach(function(b){\n"
        "      if (b.innerText.trim() === STOP) b.innerText = PLAY;\n"
        "    });\n"
        "    return [];\n"
        "  }\n"
        "  window.speechSynthesis && window.speechSynthesis.cancel();\n"
        "\n"
        "  // Find container — try exact id, then prefix match (Gradio sometimes appends -0)\n"
        "  var el = document.getElementById(EID);\n"
        "  if (!el) {\n"
        "    var candidates = document.querySelectorAll('[id^=\"' + EID + '\"]');\n"
        "    if (candidates.length) el = candidates[0];\n"
        "  }\n"
        "  if (!el) { alert('Container not found: ' + EID); return []; }\n"
        "\n"
        + _EXTRACT_TEXT +
        "\n"
        "  var text = extractText(el);\n"
        "  // Strip markdown dividers and box-drawing chars before speaking\n"
        "  text = text.replace(/[=\\-]{3,}/g,' ').replace(/[#*|>~`]+/g,' ')\n"
        "           .replace(/[\\u2500-\\u259F]+/g,' ').replace(/\\s{2,}/g,' ').trim();\n"
        "  if (!text || text.indexOf('Output will appear here') >= 0) {\n"
        "    alert('No report yet. Load & Analyse a patient first.'); return [];\n"
        "  }\n"
        "\n"
        "  window._hcai_rpt_speaking = EID;\n"
        "  document.querySelectorAll('button').forEach(function(b){\n"
        "    if (b.innerText.trim() === PLAY) b.innerText = STOP;\n"
        "  });\n"
        "\n"
        "  var u = new SpeechSynthesisUtterance(text);\n"
        "  u.lang = 'en-US'; u.rate = 0.88; u.pitch = 1.05;\n"
        "\n"
        "  function speak() {\n"
        + _VOICE_FEMALE +
        "    u.onend = u.onerror = function() {\n"
        "      window._hcai_rpt_speaking = null;\n"
        "      document.querySelectorAll('button').forEach(function(b){\n"
        "        if (b.innerText.trim() === STOP) b.innerText = PLAY;\n"
        "      });\n"
        "    };\n"
        "    window.speechSynthesis.speak(u);\n"
        "  }\n"
        "  // Voices may not be loaded yet on first call\n"
        "  if (window.speechSynthesis.getVoices().length > 0) { speak(); }\n"
        "  else { var _sp=false; window.speechSynthesis.onvoiceschanged=function(){if(!_sp){_sp=true;speak();}}; setTimeout(function(){if(!_sp){_sp=true;speak();}},250); }\n"
        "  return [];\n"
        "}"
    )


def _copy_report_js(elem_id: str) -> str:
    """Copy report text to clipboard."""
    return (
        "() => {\n"
        "  var EID = '" + elem_id + "';\n"
        "  var el = document.getElementById(EID);\n"
        "  if (!el) {\n"
        "    var c = document.querySelectorAll('[id^=\"' + EID + '\"]');\n"
        "    if (c.length) el = c[0];\n"
        "  }\n"
        "  if (!el) { alert('Container not found: ' + EID); return []; }\n"
        + _EXTRACT_TEXT +
        "  var text = extractText(el);\n"
        "  // Strip markdown dividers and box-drawing chars before speaking\n"
        "  text = text.replace(/[=\\-]{3,}/g,' ').replace(/[#*|>~`]+/g,' ')\n"
        "           .replace(/[\\u2500-\\u259F]+/g,' ').replace(/\\s{2,}/g,' ').trim();\n"
        "  if (!text || text.indexOf('Output will appear here') >= 0) {\n"
        "    alert('Nothing to copy yet.'); return [];\n"
        "  }\n"
        "  navigator.clipboard.writeText(text).then(\n"
        "    function(){ alert('Report copied!'); },\n"
        "    function(err){ alert('Copy failed: ' + err); }\n"
        "  );\n"
        "  return [];\n"
        "}"
    )



# ─────────────────────────────────────────────────────────────────────────────
# Chat JS helpers
# All use gr.Button(fn=None, js=...) — the only reliable Gradio JS method.
# Chatbot is given elem_id="hcai-chatbot" so we anchor to it precisely.
# ─────────────────────────────────────────────────────────────────────────────

# Inline JS to strip markdown/divider noise before TTS reads it
_JS_CLEAN = (
    "  function cleanTTS(t) {\n"
    "    return t.replace(/={3,}/g,' ').replace(/-{3,}/g,' ')\n"
    "            .replace(/[#*|>~`]+/g,' ').replace(/[\\u2500-\\u259F]+/g,' ')\n"
    "            .replace(/\\s{2,}/g,' ').trim();\n"
    "  }\n"
)

# Inline JS: resolve chatbot wrapper + extract only message bubbles (not footer)
# Gradio 4/5/6 marks messages with data-testid="bot" and data-testid="user".
# If those are absent we use the scrollable message list, NOT wrap.children.
_JS_FIND_CHATBOT = (
    "  var wrap = document.getElementById('hcai-chatbot');\n"
    "  // Gradio may nest the actual scroll container one level deep\n"
    "  var scope = wrap || document.querySelector('.chatbot, [data-testid=\"chatbot\"]');\n"
)


def _chat_tts_js() -> str:
    return (
        "() => {\n"
        "  var PLAY = '\U0001f50a Read Last'; var STOP = '\u23f9 Stop';\n"
        "  if (window._hcai_chat_speaking) {\n"
        "    speechSynthesis.cancel(); window._hcai_chat_speaking = false;\n"
        "    document.querySelectorAll('button').forEach(function(b){if(b.innerText.trim()===STOP)b.innerText=PLAY;});\n"
        "    return [];\n"
        "  }\n"
        + _JS_CLEAN
        + _JS_FIND_CHATBOT
        + "  var el = null;\n"
        "  if (scope) {\n"
        "    var bots = scope.querySelectorAll('[data-testid=\"bot\"]');\n"
        "    if (bots.length) el = bots[bots.length-1];\n"
        "    if (!el) {\n"
        "      // Fallback: find the Gradio message list (ul or div with role=list or class containing 'messages')\n"
        "      var msgList = scope.querySelector('[class*=\"messages\"], [class*=\"message-wrap\"], ul');\n"
        "      if (!msgList) msgList = scope;\n"
        "      var all = Array.from(msgList.querySelectorAll('[class*=\"message\"],[class*=\"bubble\"]'))\n"
        "                  .filter(function(e){ return (e.innerText||'').trim(); });\n"
        "      if (all.length) el = all[all.length-1];\n"
        "    }\n"
        "  }\n"
        "  if (!el) { alert('No messages yet. Send a message first.'); return []; }\n"
        "  var text = cleanTTS((el.innerText||el.textContent||'').trim());\n"
        "  if (!text) { alert('Empty message.'); return []; }\n"
        "  window._hcai_chat_speaking = true;\n"
        "  document.querySelectorAll('button').forEach(function(b){if(b.innerText.trim()===PLAY)b.innerText=STOP;});\n"
        "  var u = new SpeechSynthesisUtterance(text);\n"
        "  u.lang='en-US'; u.rate=0.88; u.pitch=0.95;\n"
        "  function speak(){\n"
        + _VOICE_MALE
        + "    u.onend=u.onerror=function(){window._hcai_chat_speaking=false;\n"
        "      document.querySelectorAll('button').forEach(function(b){if(b.innerText.trim()===STOP)b.innerText=PLAY;});};\n"
        "    speechSynthesis.speak(u);\n"
        "  }\n"
        "  if (speechSynthesis.getVoices().length>0){speak();}\n"
        "  else{var _sp=false; speechSynthesis.onvoiceschanged=function(){if(!_sp){_sp=true;speak();}}; setTimeout(function(){if(!_sp){_sp=true;speak();}},250);}\n"
        "  return [];\n"
        "}"
    )


def _chat_copy_last_js() -> str:
    return (
        "() => {\n"
        + _JS_FIND_CHATBOT
        + "  var el = null;\n"
        "  if (scope) {\n"
        "    var bots = scope.querySelectorAll('[data-testid=\"bot\"]');\n"
        "    if (bots.length) el = bots[bots.length-1];\n"
        "    if (!el) {\n"
        "      var msgList = scope.querySelector('[class*=\"messages\"],[class*=\"message-wrap\"],ul');\n"
        "      if (!msgList) msgList = scope;\n"
        "      var all = Array.from(msgList.querySelectorAll('[class*=\"message\"],[class*=\"bubble\"]'))\n"
        "                  .filter(function(e){ return (e.innerText||'').trim(); });\n"
        "      if (all.length) el = all[all.length-1];\n"
        "    }\n"
        "  }\n"
        "  if (!el) { alert('No message found.'); return []; }\n"
        "  var text = (el.innerText||el.textContent||'').trim();\n"
        "  if (!text) { alert('Nothing to copy.'); return []; }\n"
        "  navigator.clipboard.writeText(text).then(\n"
        "    function(){alert('Last reply copied!');}, function(){alert('Copy failed.');});\n"
        "  return [];\n"
        "}"
    )


def _chat_copy_all_js() -> str:
    return (
        "() => {\n"
        + _JS_FIND_CHATBOT
        + "  if (!scope) { alert('Chat not found.'); return []; }\n"
        "  var lines = [];\n"
        "  var uEls = Array.from(scope.querySelectorAll('[data-testid=\"user\"]'));\n"
        "  var bEls = Array.from(scope.querySelectorAll('[data-testid=\"bot\"]'));\n"
        "  if (uEls.length || bEls.length) {\n"
        "    var n = Math.max(uEls.length, bEls.length);\n"
        "    for (var i=0;i<n;i++) {\n"
        "      if (uEls[i]){var t=(uEls[i].innerText||'').trim(); if(t) lines.push('You: '+t);}\n"
        "      if (bEls[i]){var t=(bEls[i].innerText||'').trim(); if(t) lines.push('AI:  '+t);}\n"
        "    }\n"
        "  } else {\n"
        "    // Fallback: message list children only (not outer wrapper children)\n"
        "    var msgList = scope.querySelector('[class*=\"messages\"],[class*=\"message-wrap\"],ul');\n"
        "    if (!msgList) msgList = scope;\n"
        "    var all = Array.from(msgList.querySelectorAll('[class*=\"message\"],[class*=\"bubble\"]'))\n"
        "                .filter(function(e){ return (e.innerText||'').trim(); });\n"
        "    all.forEach(function(e,i){\n"
        "      var t=e.innerText.trim(); if(t) lines.push((i%2===0?'You: ':'AI:  ')+t);\n"
        "    });\n"
        "  }\n"
        "  if (!lines.length) { alert('No messages to copy.'); return []; }\n"
        "  navigator.clipboard.writeText(lines.join('\\n\\n')).then(\n"
        "    function(){alert('Copied '+lines.length+' messages!');}, function(){alert('Copy failed.');});\n"
        "  return [];\n"
        "}"
    )



# ─────────────────────────────────────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    init_db()
    startup_rows    = db_load_all()
    startup_choices = [f"{r['id']}. {r['name']} (age {r['age']})" for r in startup_rows]
    print(f"[DB] Loaded {len(startup_rows)} patients")



    tab_css = """
        /* ── Compact layout, natural page scroll ── */
        .gradio-container { padding: 4px 8px !important; }
        .gap { gap: 4px !important; }
        .form { gap: 4px !important; }
        .block { padding: 4px !important; }
        footer { display: none !important; }

        /* Fixed header button size: small height, wide enough for label */
        #hcai-header-row button {
            min-width: 140px !important;
            max-width: 180px !important;
            width: 160px !important;
            font-size: 0.92rem !important;
            padding: 4px 10px !important;
            height: 32px !important;
        }

        /* Patients tab — navy */
        button[id$="-0"] { background: #1e3a5f !important; color: white !important;
                           border-radius: 8px 8px 0 0 !important; font-weight: 600 !important; }
        button[id$="-0"]:not(.selected) { background: #e8edf5 !important; color: #1e3a5f !important; }
        /* Chat tab — green */
        button[id$="-1"] { background: #059669 !important; color: white !important;
                           border-radius: 8px 8px 0 0 !important; font-weight: 600 !important; }
        button[id$="-1"]:not(.selected) { background: #d1fae5 !important; color: #065f46 !important; }
        /* Status md — no gap */
        #hcai-status-md { margin: 0 !important; padding: 0 !important; }
        /* Sample question buttons — fixed width, white background */
        .sq-btn { width: 420px !important; max-width: 100% !important; }
        .sq-btn button {
            width: 420px !important;
            max-width: 100% !important;
            text-align: left !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: initial !important;
            display: inline-block !important;
            margin-bottom: 3px !important;
            font-size: 0.78rem !important;
            padding: 4px 10px !important;
            background: white !important;
            border: 1px solid #d1d5db !important;
            color: #374151 !important;
            box-shadow: none !important;
        }
        .sq-btn button:hover {
            background: #f3f4f6 !important;
            border-color: #1e3a5f !important;
            color: #1e3a5f !important;
        }
    """
    with gr.Blocks(title="Healthcare AI") as demo:
        gr.HTML(f"<style>{tab_css}</style>")

        gr.HTML(TITLE_HTML)

        # ── Env/device/LLM line — centered under title ────────────────────────
        status_md = gr.HTML(
            f'<div style="text-align:center;font-size:0.78rem;margin:0;padding:0 0 3px 0;">{get_system_status()}</div>',
            elem_id="hcai-status-md"
        )

        # ── Header row: Workflow + Refresh side by side ───────────────────────
        with gr.Row(elem_id="hcai-header-row"):
            workflow_toggle_btn = gr.Button("🧩 Show Workflow", size="sm", variant="secondary", scale=1)
            refresh_btn         = gr.Button("🔄 Refresh",      size="sm", scale=1)

        refresh_btn.click(fn=get_system_status, outputs=status_md)

        # ── Shared state ──────────────────────────────────────────────────────
        patient_state      = gr.State(value={})
        all_patients       = gr.State(value=startup_rows)
        delete_all_confirm = gr.State(value=False)
        workflow_visible   = gr.State(value=False)
        workflow_html = gr.HTML("", visible=False, elem_id="hcai-workflow")

        # ── Two top-level tabs: Patients | Chat ───────────────────────────────
        with gr.Tabs(selected=0) as main_tabs:

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — PATIENTS
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("🏥 Patients", id=0):

                # ── Patient Database + New Patient side by side ───────────
                with gr.Row(equal_height=False):
                    # Left: selector + action buttons
                    with gr.Column(scale=3):
                        gr.Markdown("#### 🗄️ Patient Database")
                        patient_selector = gr.Dropdown(
                            label="Select Patient",
                            choices=startup_choices,
                            value=startup_choices[0] if startup_choices else None,
                            interactive=True,
                        )
                        with gr.Row():
                            load_btn           = gr.Button("📋 Load & Analyse",   variant="primary", size="sm", scale=3)
                            delete_btn         = gr.Button("🗑️ Delete",            variant="stop",    size="sm", scale=1)
                            delete_all_btn     = gr.Button("🗑️ Delete All",        variant="stop",    size="sm", scale=2)
                        with gr.Row():
                            delete_all_confirm_btn = gr.Button("⚠️ Confirm Delete All", variant="stop",      size="sm", scale=2, visible=False)
                            delete_all_cancel_btn  = gr.Button("✕ Cancel",              variant="secondary", size="sm", scale=1, visible=False)
                        delete_all_status = gr.Markdown("")

                    # Right: New Patient form
                    with gr.Column(scale=2):
                        with gr.Accordion("➕ New Patient", open=False):
                            with gr.Row():
                                np_name   = gr.Textbox(label="Name *", scale=2)
                                np_age    = gr.Number(label="Age", minimum=0, maximum=150, value=None, scale=1)
                                np_gender = gr.Dropdown(label="Gender",
                                                        choices=["Male","Female","Non-binary","Unknown"],
                                                        value=None, scale=1)
                            np_symptoms  = gr.Textbox(label="🩺 Symptoms *",     lines=1)
                            np_vitals    = gr.Textbox(label="📊 Vital Signs",     lines=1)
                            np_history   = gr.Textbox(label="📋 Medical History", lines=1)
                            with gr.Row():
                                np_meds      = gr.Textbox(label="💊 Medications", lines=1, scale=1)
                                np_allergies = gr.Textbox(label="⚠️ Allergies",   lines=1, scale=1)
                            np_save_btn = gr.Button("💾 Save Patient", variant="primary")
                            np_status   = gr.Markdown("")

                # ── Patient info + Reports side by side ───────────────────────
                with gr.Row(equal_height=False):

                    # LEFT: Patient details (compact)
                    with gr.Column(scale=1):
                        gr.Markdown("#### 👤 Patient Information")
                        created_at_box = gr.Textbox(
                            label="🗓️ Date Registered",
                            value="",
                            interactive=False, lines=1,
                        )
                        with gr.Row():
                            name_in   = gr.Textbox(label="Name",   placeholder="Patient name", scale=2)
                            age_in    = gr.Number( label="Age",    minimum=0, maximum=150, value=None, scale=1)
                            gender_in = gr.Dropdown(label="Gender",
                                                    choices=["Male","Female","Non-binary","Unknown"],
                                                    value=None, scale=1)
                        symptoms_in = gr.Textbox(label="🩺 Symptoms",        placeholder="Chief complaint...", lines=1)
                        vitals_in   = gr.Textbox(label="📊 Vital Signs",     placeholder="BP 120/80, HR 72...", lines=1)
                        history_in  = gr.Textbox(label="📋 Medical History", placeholder="Past diagnoses...", lines=1)
                        with gr.Row():
                            meds_in      = gr.Textbox(label="💊 Medications", placeholder="Drug, dose...", lines=1, scale=1)
                            allergies_in = gr.Textbox(label="⚠️ Allergies",   placeholder="Allergies",     lines=1, scale=1)

                    # RIGHT: Clinical Reports (compact)
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📄 Clinical Reports")

                        with gr.Tab("🚨 Triage"):
                            with gr.Row():
                                triage_tts_btn  = gr.Button("🔊 Read Aloud", size="sm", variant="secondary")
                                triage_copy_btn = gr.Button("📋 Copy",       size="sm", variant="secondary")
                            triage_out = gr.HTML(value=make_report_html(""), elem_id="hcai-triage")

                        with gr.Tab("🔬 Diagnosis"):
                            with gr.Row():
                                diag_tts_btn  = gr.Button("🔊 Read Aloud", size="sm", variant="secondary")
                                diag_copy_btn = gr.Button("📋 Copy",       size="sm", variant="secondary")
                            diagnosis_out = gr.HTML(value=make_report_html(""), elem_id="hcai-diag")

                        with gr.Tab("💊 Treatment"):
                            with gr.Row():
                                treat_tts_btn  = gr.Button("🔊 Read Aloud", size="sm", variant="secondary")
                                treat_copy_btn = gr.Button("📋 Copy",       size="sm", variant="secondary")
                            treatment_out = gr.HTML(value=make_report_html(""), elem_id="hcai-treat")

                        with gr.Tab("📋 Full Report"):
                            with gr.Row():
                                full_tts_btn  = gr.Button("🔊 Read Aloud", size="sm", variant="secondary")
                                full_copy_btn = gr.Button("📋 Copy",       size="sm", variant="secondary")
                            final_out = gr.HTML(value=make_report_html("", full=True), elem_id="hcai-full")

                gr.HTML("<div id='hcai-footer-p' style='text-align:center;color:#9ca3af;font-size:11px;padding:12px'>"
                        "Healthcare AI | LangGraph + FastAPI + Gradio | Not a replacement for clinical judgment</div>")

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — CHAT
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("💬 Chat", id=1):

                # Header row: subtitle left, "Sample Questions" label right
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 💬 Clinical Chat")
                        gr.Markdown("<small style='color:#6b7280'>Load & Analyze a patient first for full clinical answers.</small>")
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div style="font-size:0.95rem;font-weight:700;color:#1e3a5f;'
                            'padding:22px 0 0 4px;">Sample Questions</div>'
                        )

                # Chat + Sample Questions side by side
                with gr.Row(equal_height=False):

                    # ── LEFT: chat window ─────────────────────────────────────
                    with gr.Column(scale=4):
                        try:
                            chatbot = gr.Chatbot(label="Clinical Chat", height=450, elem_id="hcai-chatbot")
                        except TypeError:
                            chatbot = gr.Chatbot(label="Clinical Chat", height=450, elem_id="hcai-chatbot")

                        # Toolbar under chatbot
                        with gr.Row():
                            chat_tts_btn       = gr.Button("🔊 Read Last", size="sm", variant="secondary")
                            chat_copy_last_btn = gr.Button("📋 Copy Last", size="sm", variant="secondary")
                            chat_copy_all_btn  = gr.Button("📋 Copy All",  size="sm", variant="secondary")

                        chat_input = gr.Textbox(
                            placeholder="Type a question and press Enter or click Send...",
                            label="", lines=1, show_label=False, elem_id="hcai-chat-input",
                        )
                        with gr.Row():
                            chat_btn  = gr.Button("Send 💬",  variant="primary",   scale=1)
                            clear_btn = gr.Button("Clear 🗑️", variant="secondary", scale=1)

                    # ── RIGHT: sample questions, width = longest question ──────
                    with gr.Column(scale=1):
                        sq_btns = []
                        for q in SAMPLE_QUESTIONS:
                            b = gr.Button(
                                q, size="sm", variant="secondary",
                                elem_classes=["sq-btn"],
                            )
                            sq_btns.append((b, q))

                gr.HTML("<div id='hcai-footer' style='text-align:center;color:#9ca3af;font-size:11px;padding:8px'>"
                        "Healthcare AI | LangGraph + FastAPI + Gradio | Not a replacement for clinical judgment</div>")

        # ── Workflow Toggle Logic ─────────────────────────────────────────────
        def toggle_workflow_diagram(current_visible):
            if current_visible:
                return False, gr.update(visible=False), gr.update(value="🧩 Show Workflow")
            try:
                import base64
                from pipeline import build_graph
                png = build_graph().get_graph().draw_mermaid_png()
                b64 = base64.b64encode(png).decode()
                html = (
                    '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                    'border-radius:12px;padding:6px;margin:4px 0;'
                    'display:flex;flex-direction:column;align-items:center;'
                    'overflow-x:auto;max-width:320px;">'
                    '<h3 style="color:#1e3a5f;margin:0 0 8px 0;font-size:0.82rem;">'
                    '🔀 LangGraph Multi-Agent Pipeline</h3>'
                    f'<img src="data:image/png;base64,{b64}" '
                    'style="width:100%;max-width:300px;height:auto;'
                    'object-fit:contain;border-radius:8px;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.1);">'
                    '</div>'
                )
            except Exception as e:
                html = f'<i style="color:red;">Could not render workflow: {e}</i>'
            return True, gr.update(value=html, visible=True), gr.update(value="🙈 Hide Workflow")

        workflow_toggle_btn.click(
            fn=toggle_workflow_diagram,
            inputs=[workflow_visible],
            outputs=[workflow_visible, workflow_html, workflow_toggle_btn],
        )

        # ── Python event functions ────────────────────────────────────────────

        def load_and_analyse(selection, rows):
            EMPTY = [
                "", None, None, "", "", "", "", "",
                "",
                make_report_html(""), make_report_html(""),
                make_report_html(""), make_report_html("", full=True),
                {}, [],
            ]
            if not selection:
                return EMPTY
            try:
                pid = int(str(selection).split(".")[0])
                row = next((r for r in db_load_all() if r["id"] == pid), None)
                if not row:
                    return EMPTY
            except Exception as e:
                # error handled — no pipeline_status to update
                return EMPTY

            name      = row.get("name", "")
            age       = row.get("age") or None
            gender    = row.get("gender") or None
            symptoms  = row.get("symptoms", "")
            vitals    = row.get("vitals", "")
            history   = row.get("history", "")
            meds      = row.get("medications", "")
            allergies = row.get("allergies", "")
            created   = row.get("created_at") or "Unknown"

            status, triage, diagnosis, treatment, full, state = run_analysis(
                name, age, gender, symptoms, vitals, history, meds, allergies)

            return (
                name, age, gender, symptoms, vitals, history, meds, allergies,
                created,
                triage, diagnosis, treatment, full,
                state, [],
            )

        def delete_patient(selection, rows):
            if not selection:
                return rows, gr.update()
            try:
                pid = int(str(selection).split(".")[0])
                db_delete_patient(pid)
                nr = db_load_all()
                nc = [f"{r['id']}. {r['name']} (age {r['age']})" for r in nr]
                return nr, gr.update(choices=nc, value=nc[0] if nc else None)
            except Exception:
                return rows, gr.update()

        def save_new_patient(name, age, gender, symptoms, vitals, history,
                             medications, allergies, rows):
            if not (name or "").strip():
                return "⚠️ Name required.", rows, gr.update()
            if not (symptoms or "").strip():
                return "⚠️ Symptoms required.", rows, gr.update()
            with _db_lock:
                db_add_patient(name, int(age) if age else 0, gender or "Unknown",
                               symptoms, vitals or "", history or "",
                               medications or "", allergies or "")
            nr = db_load_all()
            nc = [f"{r['id']}. {r['name']} (age {r['age']})" for r in nr]
            return f"✅ {name} saved.", nr, gr.update(choices=nc, value=nc[-1])

        # Delete All — two-step confirmation
        def on_delete_all_click():
            """First click: show confirm/cancel buttons."""
            return (
                gr.update(value="⚠️ This will permanently delete ALL patients. Click Confirm to proceed."),
                gr.update(visible=False),   # hide Delete All btn
                gr.update(visible=True),    # show Confirm btn
                gr.update(visible=True),    # show Cancel btn
            )

        def on_delete_all_confirm(rows):
            """Confirmed: delete everything, refresh dropdown."""
            with _db_lock:
                db_delete_all()
            return (
                [],                                                       # all_patients
                gr.update(choices=[], value=None),                        # patient_selector
                gr.update(value="✅ All patients deleted."),              # status
                gr.update(visible=True),                                  # show Delete All btn
                gr.update(visible=False),                                 # hide Confirm
                gr.update(visible=False),                                 # hide Cancel
            )

        def on_delete_all_cancel():
            """Cancelled: hide confirm/cancel, clear status."""
            return (
                gr.update(value=""),          # clear status
                gr.update(visible=True),      # show Delete All btn
                gr.update(visible=False),     # hide Confirm
                gr.update(visible=False),     # hide Cancel
            )

        # ── Wire Python events ────────────────────────────────────────────────

        load_btn.click(
            fn=load_and_analyse,
            inputs=[patient_selector, all_patients],
            outputs=[name_in, age_in, gender_in, symptoms_in,
                     vitals_in, history_in, meds_in, allergies_in,
                     created_at_box,
                     triage_out, diagnosis_out, treatment_out, final_out,
                     patient_state, chatbot],
            show_progress="full",
        )
        # After loading, switch to Chat tab and focus the question input
        load_btn.click(
            fn=None,
            js="""() => {
                // Switch to Chat tab immediately on click (before analysis completes)
                var tabs = document.querySelectorAll('.tab-nav button');
                if (tabs.length > 1) tabs[1].click();
                // Focus chat input after tab switch
                setTimeout(function() {
                    var inp = document.getElementById('hcai-chat-input');
                    if (inp) {
                        var ta = inp.querySelector('textarea') || inp.querySelector('input');
                        if (ta) { ta.focus(); }
                    }
                }, 300);
                return [];
            }"""
        )
        delete_btn.click(
            fn=delete_patient,
            inputs=[patient_selector, all_patients],
            outputs=[all_patients, patient_selector],
        )
        np_save_btn.click(
            fn=save_new_patient,
            inputs=[np_name, np_age, np_gender, np_symptoms, np_vitals,
                    np_history, np_meds, np_allergies, all_patients],
            outputs=[np_status, all_patients, patient_selector],
        )

        # Delete All — 2-step
        delete_all_btn.click(
            fn=on_delete_all_click,
            outputs=[delete_all_status, delete_all_btn, delete_all_confirm_btn, delete_all_cancel_btn],
        )
        delete_all_confirm_btn.click(
            fn=on_delete_all_confirm,
            inputs=[all_patients],
            outputs=[all_patients, patient_selector, delete_all_status,
                     delete_all_btn, delete_all_confirm_btn, delete_all_cancel_btn],
        )
        delete_all_cancel_btn.click(
            fn=on_delete_all_cancel,
            outputs=[delete_all_status, delete_all_btn, delete_all_confirm_btn, delete_all_cancel_btn],
        )

        # Sample questions → auto-send
        for btn, question in sq_btns:
            btn.click(
                fn=lambda h, ps, ap, q=question: chat_respond(q, h, ps, ap),
                inputs=[chatbot, patient_state, all_patients],
                outputs=[chatbot, chat_input],
            )

        chat_inputs = [chat_input, chatbot, patient_state, all_patients]
        chat_btn.click(fn=chat_respond, inputs=chat_inputs, outputs=[chatbot, chat_input])
        chat_input.submit(fn=chat_respond, inputs=chat_inputs, outputs=[chatbot, chat_input])
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

        # ── Refresh patients from DB on every page load ───────────────────────
        # gr.State(value=...) is frozen at server-start time.
        # demo.load() re-queries the DB for every new browser session,
        # so patients added/removed while the server is running are always visible.
        def _on_page_load():
            rows = db_load_all()
            choices = [f"{r['id']}. {r['name']} (age {r['age']})" for r in rows]
            print(f"[DB] Page load — {len(rows)} patients in DB at {DB_PATH}")
            return (
                rows,
                gr.update(choices=choices, value=choices[0] if choices else None),
            )

        demo.load(
            fn=_on_page_load,
            outputs=[all_patients, patient_selector],
        )

        # ── Wire JS buttons ───────────────────────────────────────────────────

        # Reports — American female voice
        triage_tts_btn.click( fn=None, js=_tts_report_js("hcai-triage"))
        diag_tts_btn.click(   fn=None, js=_tts_report_js("hcai-diag"))
        treat_tts_btn.click(  fn=None, js=_tts_report_js("hcai-treat"))
        full_tts_btn.click(   fn=None, js=_tts_report_js("hcai-full"))

        triage_copy_btn.click(fn=None, js=_copy_report_js("hcai-triage"))
        diag_copy_btn.click(  fn=None, js=_copy_report_js("hcai-diag"))
        treat_copy_btn.click( fn=None, js=_copy_report_js("hcai-treat"))
        full_copy_btn.click(  fn=None, js=_copy_report_js("hcai-full"))

        # Chat — American male voice; Copy All collects all messages
        chat_tts_btn.click(      fn=None, js=_chat_tts_js())
        chat_copy_last_btn.click(fn=None, js=_chat_copy_last_js())
        chat_copy_all_btn.click( fn=None, js=_chat_copy_all_js())

    demo.queue()
    register_routes(demo.app)
    return demo

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Healthcare AI Frontend on :7860")
    build_ui().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
