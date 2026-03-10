"""
Device detection utility.
Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
"""

import platform
import subprocess
import sys


def get_device() -> str:
    """Detect and return the best available compute device."""
    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            if platform.machine() == "arm64":
                return "mps"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu"


def get_device_info() -> dict:
    """Return full device info dictionary."""
    device = get_device()
    info = {
        "device": device,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if device == "mps":
        info["note"] = "Apple Silicon GPU via Metal Performance Shaders"
        info["ollama_accel"] = "Metal (automatic)"
    elif device == "cuda":
        info["note"] = "NVIDIA GPU via CUDA"
        info["ollama_accel"] = "CUDA (automatic)"
    else:
        info["note"] = "CPU inference"
        info["ollama_accel"] = "CPU"
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(get_device_info(), indent=2))
