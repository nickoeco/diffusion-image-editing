"""Best-effort diagnostics, including missing and broken installations."""
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

PACKAGES = ("torch", "torchvision", "diffusers", "transformers", "accelerate",
            "safetensors", "huggingface-hub", "Pillow", "numpy", "opencv-python",
            "matplotlib", "scikit-image", "gradio", "pytest", "PyYAML")


def inspect_environment() -> dict[str, Any]:
    """Return JSON-safe diagnostics; failure is recorded, never labelled success."""
    report: dict[str, Any] = {
        "os": platform.platform(), "python": platform.python_version(),
        "executable": sys.executable, "packages": {},
    }
    for name in PACKAGES:
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
            capture_output=True, text=True, timeout=10, check=True)
        report["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi_error"] = str(exc)
    try:
        import torch
        available = torch.cuda.is_available()
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": available, "gpus": []}
        if available:
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                report["torch"]["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                                "free_bytes": free, "total_bytes": total})
    except Exception as exc:
        # Diagnostics must remain usable even when a binary installation is broken.
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
