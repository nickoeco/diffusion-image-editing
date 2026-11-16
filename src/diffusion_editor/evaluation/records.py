"""Strict JSON and artifact integrity checks, independent of ML libraries."""
import json
import math
from pathlib import Path
from typing import Any

from diffusion_editor.evaluation.experiment_config import file_hash


def write_json(path: Path, value: Any) -> None:
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        return x
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_completed(directory: Path, expected_fingerprint: str) -> dict | None:
    path = directory / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or report.get("case_fingerprint") != expected_fingerprint:
        raise ValueError(f"Incompatible completed case: {directory}")
    for name, digest in report["artifacts"].items():
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError(f"Changed or missing artifact: {directory / name}")
    return report
