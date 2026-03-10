"""
Device detection utility.
Automatically selects MPS (Apple Silicon), CUDA, or CPU.
"""

import platform
import subprocess
import sys


def get_device() -> str:
    """
    Detect and return the best available compute device.
    Returns: 'mps', 'cuda', or 'cpu'
    """
    # Check for Apple Silicon MPS
    if platform.system() == "Darwin":
        try:
            import torch
            if torch.backends.mps.is_available():
                print("✅ Device: MPS (Apple Silicon GPU)")
                return "mps"
        except ImportError:
            # torch not installed, check for Apple Silicon via platform
            if platform.machine() == "arm64":
                print("✅ Device: MPS (Apple Silicon - torch not installed, Ollama will use Metal)")
                return "mps"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ Device: CUDA ({device_name})")
            return "cuda"
    except ImportError:
        pass

    # Check for NVIDIA via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"✅ Device: CUDA ({result.stdout.strip().splitlines()[0]})")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("✅ Device: CPU")
    return "cpu"


def get_device_info() -> dict:
    """Return full device info dictionary for API and UI display."""
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
