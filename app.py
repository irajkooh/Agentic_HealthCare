"""
Healthcare AI — Main Entry Point
==================================
Dual-mode launcher detected automatically:

  HuggingFace Spaces  (SYSTEM=spaces env var set by HF)
      FastAPI runs in a background thread.
      Gradio runs as the main process (required by HF).
      LLM = HuggingFace Inference API (no Ollama needed).

  Local development
      Creates patients.db automatically if missing (empty, no prompts).
      Kills ports 8000 and 7860 if occupied.
      Spawns backend.py (FastAPI) and frontend.py (Gradio) as
      subprocesses with color-coded log streaming.
      LLM = Ollama localhost:11434 with llama3.2.

Usage (both environments):
    python app.py
"""

import os
import sys
import time
import signal
import socket
import sqlite3
import platform
import subprocess
import threading
import webbrowser

PROJECT_DIR   = os.path.dirname(os.path.abspath(__file__))
BACKEND_PORT  = 8000
FRONTEND_PORT = 7860
PYTHON        = sys.executable
DB_PATH       = os.path.join(PROJECT_DIR, "patients.db")

sys.path.insert(0, PROJECT_DIR)


# ── Patient DB check ───────────────────────────────────────────────────────────
# Runs BEFORE backend/frontend start so the prompt reaches the terminal.

_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS patients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        age         INTEGER,
        gender      TEXT,
        symptoms    TEXT,
        vitals      TEXT,
        history     TEXT,
        medications TEXT,
        allergies   TEXT
    )
"""


def _create_empty_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(_CREATE_SQL)
    con.commit()
    con.close()




def check_patient_db():
    """
    Ensure patients.db exists with the correct schema.
    Creates an empty database silently if missing — no prompts.
    """
    existed = os.path.isfile(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.execute(_CREATE_SQL)
    count = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    con.commit()
    con.close()
    if existed:
        print(f"[DB] patients.db found — {count} patient(s)")
    else:
        print(f"[DB] patients.db not found — created empty database at {DB_PATH}")
        print("[DB] Add patients via the UI (➕ New Patient button)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_hf_space() -> bool:
    return os.environ.get("SYSTEM") == "spaces"


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(port: int, timeout_s: float = 30.0) -> bool:
    """Block until port is open or timeout. Returns True if open."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if port_is_open(port):
            return True
        time.sleep(0.5)
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  HF SPACES MODE
# ══════════════════════════════════════════════════════════════════════════════

def run_hf_mode():
    """
    Start FastAPI in a daemon thread, then launch Gradio as the main process.
    Required by HuggingFace Spaces (Gradio must be the main process).
    """
    import uvicorn
    from backend import app as fastapi_app

    def _backend():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=BACKEND_PORT, log_level="warning")

    t = threading.Thread(target=_backend, daemon=True, name="fastapi-backend")
    t.start()

    ready = wait_for_port(BACKEND_PORT, timeout_s=30)
    if ready:
        print(f"[HF] Backend ready on :{BACKEND_PORT}")
    else:
        print(f"[HF] WARNING: backend did not open port {BACKEND_PORT} in 30s — continuing anyway")

    from frontend import build_ui
    demo = build_ui()
    print(f"[HF] Launching Gradio on :{FRONTEND_PORT}")
    demo.launch(server_name="0.0.0.0", server_port=FRONTEND_PORT)


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL MODE
# ══════════════════════════════════════════════════════════════════════════════

_processes: list = []


def kill_port(port: int):
    """Kill whatever process is listening on port (macOS/Linux/Windows)."""
    system = platform.system()
    if system in ("Darwin", "Linux"):
        try:
            r = subprocess.run(["lsof", "-ti", f":{port}"],
                               capture_output=True, text=True)
            for pid in r.stdout.strip().splitlines():
                pid = pid.strip()
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                    print(f"   killed PID {pid} on port {port}")
        except FileNotFoundError:
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            except FileNotFoundError:
                pass
    elif system == "Windows":
        try:
            r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                    print(f"   killed PID {pid} on port {port}")
        except Exception as e:
            print(f"   WARNING: could not kill port {port}: {e}")
    time.sleep(0.5)


def free_ports():
    print("\nChecking ports...")
    for port in [BACKEND_PORT, FRONTEND_PORT]:
        if port_is_open(port):
            print(f"  Port {port} is busy — killing...")
            kill_port(port)
            time.sleep(0.3)
        label = "free" if not port_is_open(port) else "still busy (proceeding anyway)"
        print(f"  Port {port}: {label}")


def spawn(script: str, port: int, label: str) -> subprocess.Popen:
    print(f"\nStarting {label}...")
    proc = subprocess.Popen(
        [PYTHON, os.path.join(PROJECT_DIR, script)],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if wait_for_port(port, timeout_s=45):
        print(f"  {label} ready on :{port}")
    else:
        print(f"  {label} taking longer than expected (still starting...)")
    return proc


def stream(proc: subprocess.Popen, tag: str):
    BLUE, GREEN, RESET = "\033[94m", "\033[92m", "\033[0m"
    color = BLUE if "BACKEND" in tag else GREEN
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"{color}[{tag}]{RESET} {line}", flush=True)
    except Exception:
        pass


def shutdown(sig=None, frame=None):
    print("\nShutting down Healthcare AI...")
    for p in _processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("All processes stopped.")
    sys.exit(0)


def preflight():
    import requests
    print("\nPre-flight checks:")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models    = [m["name"] for m in r.json().get("models", [])] if r.ok else []
        has_model = any("llama3.2" in m for m in models)
        print(f"  Ollama  : running")
        print(f"  llama3.2: {'ready' if has_model else 'NOT FOUND — run: ollama pull llama3.2'}")
    except Exception:
        print("  Ollama  : NOT running — run: ollama serve")

    try:
        from utils.device import get_device_info
        d = get_device_info()
        print(f"  Device  : {d['device'].upper()} — {d.get('note','')}")
    except Exception as e:
        print(f"  Device  : check failed ({e})")


def run_local_mode():
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("=" * 58)
    print("  Healthcare AI — Clinical Decision Support")
    print("  Multi-Agent: Triage -> Diagnosis -> Treatment")
    print("=" * 58)
    print(f"  Python: {sys.version.split()[0]}  |  OS: {platform.system()} {platform.machine()}")

    preflight()
    free_ports()

    backend  = spawn("backend.py",  BACKEND_PORT,  "Backend  (FastAPI :8000)")
    _processes.append(backend)
    frontend = spawn("frontend.py", FRONTEND_PORT, "Frontend (Gradio  :7860)")
    _processes.append(frontend)

    print()
    print("=" * 58)
    print("  Healthcare AI is RUNNING")
    print(f"  UI       ->  http://localhost:{FRONTEND_PORT}")
    print(f"  API      ->  http://localhost:{BACKEND_PORT}")
    print(f"  API Docs ->  http://localhost:{BACKEND_PORT}/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 58)

    def _open_browser():
        if wait_for_port(FRONTEND_PORT, timeout_s=30):
            webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
    threading.Thread(target=_open_browser, daemon=True).start()

    threading.Thread(target=stream, args=(backend,  "BACKEND "), daemon=True).start()
    threading.Thread(target=stream, args=(frontend, "FRONTEND"), daemon=True).start()

    while True:
        time.sleep(3)
        if backend.poll() is not None:
            print("Backend crashed — restarting...")
            _processes.remove(backend)
            backend = spawn("backend.py", BACKEND_PORT, "Backend")
            _processes.append(backend)
            threading.Thread(target=stream, args=(backend, "BACKEND "), daemon=True).start()
        if frontend.poll() is not None:
            print("Frontend crashed — restarting...")
            _processes.remove(frontend)
            frontend = spawn("frontend.py", FRONTEND_PORT, "Frontend")
            _processes.append(frontend)
            threading.Thread(target=stream, args=(frontend, "FRONTEND"), daemon=True).start()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if is_hf_space():
        print("HuggingFace Spaces detected — HF mode")
        run_hf_mode()
    else:
        print("Local environment — local mode")
        check_patient_db()   # ← blocks here if no DB, prompts before anything starts
        run_local_mode()