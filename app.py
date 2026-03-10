"""
Healthcare AI — Main Orchestrator
===================================
Kills ports 8000 and 7860 if in use, then starts:
  • backend.py  → FastAPI on port 8000
  • frontend.py → Gradio UI on port 7860

Usage:
  python app.py

Stop:
  Ctrl+C
"""

import subprocess
import sys
import os
import time
import signal
import platform
import socket
import webbrowser

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_PORT = 8000
FRONTEND_PORT = 7860
PYTHON = sys.executable

processes = []


# ─── Port Management ──────────────────────────────────────────────────────────

def is_port_open(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


def kill_port(port: int):
    """Kill any process listening on the given port."""
    system = platform.system()
    killed = False

    if system in ("Linux", "Darwin"):
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            pids = result.stdout.strip().splitlines()
            for pid in pids:
                pid = pid.strip()
                if pid:
                    subprocess.run(["kill", "-9", pid], capture_output=True)
                    print(f"   ✅ Killed PID {pid} on port {port}")
                    killed = True
        except FileNotFoundError:
            # fallback: fuser
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
                killed = True
            except FileNotFoundError:
                pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = parts[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                    print(f"   ✅ Killed PID {pid} on port {port}")
                    killed = True
        except Exception as e:
            print(f"   ⚠️  Could not kill port {port}: {e}")

    if not killed and is_port_open(port):
        print(f"   ⚠️  Port {port} still in use — proceeding anyway")

    time.sleep(0.5)


def free_ports():
    """Ensure both ports are free before launching."""
    print("\n🔌 Checking ports...")

    for port in [BACKEND_PORT, FRONTEND_PORT]:
        if is_port_open(port):
            print(f"   Port {port} is in use → killing...")
            kill_port(port)
        else:
            print(f"   Port {port} is free ✅")

    # Final check
    time.sleep(1)
    for port in [BACKEND_PORT, FRONTEND_PORT]:
        if is_port_open(port):
            print(f"   ⚠️  Port {port} still occupied after kill attempt")
        else:
            print(f"   Port {port} confirmed free ✅")


# ─── Process Launchers ────────────────────────────────────────────────────────

def start_backend() -> subprocess.Popen:
    """Launch the FastAPI backend."""
    print("\n🚀 Starting Backend (FastAPI on :8000)...")
    proc = subprocess.Popen(
        [PYTHON, os.path.join(PROJECT_DIR, "backend.py")],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    # Give backend time to bind
    for _ in range(20):
        time.sleep(0.5)
        if is_port_open(BACKEND_PORT):
            print(f"   ✅ Backend ready at http://localhost:{BACKEND_PORT}")
            print(f"   📖 API docs   at http://localhost:{BACKEND_PORT}/docs")
            return proc
    print("   ⚠️  Backend taking longer than expected...")
    return proc


def start_frontend() -> subprocess.Popen:
    """Launch the Gradio frontend."""
    print("\n🖥️  Starting Frontend (Gradio on :7860)...")
    proc = subprocess.Popen(
        [PYTHON, os.path.join(PROJECT_DIR, "frontend.py")],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for _ in range(20):
        time.sleep(0.5)
        if is_port_open(FRONTEND_PORT):
            print(f"   ✅ Frontend ready at http://localhost:{FRONTEND_PORT}")
            return proc
    print("   ⚠️  Frontend taking longer than expected...")
    return proc


# ─── Log Streaming ────────────────────────────────────────────────────────────

import threading

def stream_logs(proc: subprocess.Popen, label: str):
    """Stream subprocess output with label prefix."""
    prefix_colors = {
        "BACKEND ": "\033[94m",   # blue
        "FRONTEND": "\033[92m",   # green
    }
    reset = "\033[0m"
    color = prefix_colors.get(label, "")

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"{color}[{label}]{reset} {line}")


# ─── Shutdown Handler ─────────────────────────────────────────────────────────

def shutdown(signum=None, frame=None):
    print("\n\n🛑 Shutting down Healthcare AI system...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    print("   All processes stopped. Goodbye! 👋")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ─── Pre-flight Checks ────────────────────────────────────────────────────────

def preflight_checks():
    """Run system preflight checks."""
    print("\n🔍 Pre-flight checks...")

    # Check Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            has_model = any("llama3.2" in m for m in models)
            print(f"   ✅ Ollama running | llama3.2: {'ready' if has_model else '❌ NOT FOUND'}")
            if not has_model:
                print("   💡 Run: ollama pull llama3.2")
        else:
            print("   ⚠️  Ollama returned unexpected status")
    except Exception:
        print("   ❌ Ollama not running — run: ollama serve")
        print("   ⚠️  Continuing anyway (Ollama required for analysis)")

    # Check device
    try:
        sys.path.insert(0, PROJECT_DIR)
        from utils.device import get_device_info
        info = get_device_info()
        print(f"   ✅ Device: {info['device'].upper()} — {info.get('note', '')}")
    except Exception as e:
        print(f"   ⚠️  Device check failed: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║       🏥 Healthcare AI Clinical Decision Support      ║")
    print("║   Multi-Agent: Triage → Diagnosis → Treatment        ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"   Python  : {sys.version.split()[0]}")
    print(f"   OS      : {platform.system()} {platform.machine()}")

    preflight_checks()
    free_ports()

    # Start backend first
    backend_proc = start_backend()
    processes.append(backend_proc)

    # Start frontend
    frontend_proc = start_frontend()
    processes.append(frontend_proc)

    print("\n" + "═" * 55)
    print("  ✅ Healthcare AI System is RUNNING")
    print(f"  🖥️  UI      → http://localhost:{FRONTEND_PORT}")
    print(f"  🔌 API     → http://localhost:{BACKEND_PORT}")
    print(f"  📖 API Docs→ http://localhost:{BACKEND_PORT}/docs")
    print("  Press Ctrl+C to stop all services")
    print("═" * 55 + "\n")

    # Automatically open the UI in the default web browser
    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    # Stream logs from both processes
    backend_thread = threading.Thread(
        target=stream_logs, args=(backend_proc, "BACKEND "), daemon=True
    )
    frontend_thread = threading.Thread(
        target=stream_logs, args=(frontend_proc, "FRONTEND"), daemon=True
    )
    backend_thread.start()
    frontend_thread.start()

    # Keep alive, restart if process dies
    while True:
        time.sleep(3)
        if backend_proc.poll() is not None:
            print("⚠️  Backend crashed — restarting...")
            processes.remove(backend_proc)
            backend_proc = start_backend()
            processes.append(backend_proc)
            threading.Thread(
                target=stream_logs, args=(backend_proc, "BACKEND "), daemon=True
            ).start()

        if frontend_proc.poll() is not None:
            print("⚠️  Frontend crashed — restarting...")
            processes.remove(frontend_proc)
            frontend_proc = start_frontend()
            processes.append(frontend_proc)
            threading.Thread(
                target=stream_logs, args=(frontend_proc, "FRONTEND"), daemon=True
            ).start()


if __name__ == "__main__":
    main()
