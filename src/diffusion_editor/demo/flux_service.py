"""Validated local FLUX reference jobs in an isolated, serialized runtime."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from uuid import uuid4
import zipfile

from PIL import Image

from diffusion_editor.demo.inputs import normalize_image
from diffusion_editor.demo.image_fit import ImageFit

ROOT = Path(__file__).resolve().parents[3]
Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class FluxRequest:
    operation: str
    prompt: str
    width: int = 512
    height: int = 512
    seed: int = 42
    fit_reference: bool = False

    def __post_init__(self) -> None:
        if self.operation not in ("Generate", "Edit image"):
            raise ValueError("Choose Generate or Edit image for FLUX.")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Enter a generation prompt or editing instruction.")
        if len(self.prompt) > 8000:
            raise ValueError("Keep the prompt below 8000 characters.")
        if any(type(n) is not int or not 64 <= n <= 1024 or n % 16 for n in (self.width, self.height)):
            raise ValueError("Output dimensions must be multiples of 16 in [64,1024].")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0,2**32).")
        if type(self.fit_reference) is not bool:
            raise ValueError("Reference fitting must be a boolean switch.")
        if self.operation == "Generate":
            object.__setattr__(self, "fit_reference", False)


def request_digest(request: FluxRequest, image: Image.Image | None) -> str:
    """Bind a completed result to the effective settings and reference pixels."""
    h = hashlib.sha256(json.dumps(asdict(request), sort_keys=True).encode())
    if request.operation == "Edit image" and image is not None:
        rgb = normalize_image(image)
        h.update(str(rgb.size).encode())
        h.update(rgb.tobytes())
    return h.hexdigest()


@dataclass
class FluxResult:
    image: Image.Image
    status: str
    report: dict[str, Any]
    result_file: str
    archive_file: str
    report_file: str
    digest: str
    run_id: str


def run_worker(command: list[str], log: Path, notify: Progress, timeout: float) -> None:
    """Monitor our sole child, forward structured progress and retain raw logs."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        started = time.monotonic()
        try:
            with log.open(encoding="utf-8", errors="replace") as reader:
                while True:
                    for line in reader:
                        if line.startswith("FLUX_PROGRESS "):
                            update = json.loads(line.removeprefix("FLUX_PROGRESS "))
                            notify(float(update["fraction"]), str(update["message"]))
                    if process.poll() is not None:
                        break
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f"FLUX exceeded {timeout:g} seconds; the worker will be stopped.")
                    time.sleep(0.2)
            if process.returncode:
                tail = log.read_text(encoding="utf-8", errors="replace")[-1800:]
                raise RuntimeError(f"FLUX worker exited with code {process.returncode}. Check runtime/model compatibility.\n{tail}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


class FluxDemoService:
    """No model loading on construction; each job releases its isolated worker."""

    def __init__(self, owner: Any, *, python: str | Path | None = None,
                 config_path: str | Path | None = None, worker_runner: Callable = run_worker,
                 timeout: float = 900) -> None:
        self.owner = owner
        self.python = Path(python or os.environ.get("DIFFUSION_EDITOR_FLUX_PYTHON") or
                           ROOT.parent / "FLUX/.venv/Scripts/python.exe").resolve()
        self.config_path = Path(config_path or ROOT / "configs/flux-klein-4b-int8.json").resolve()
        self.worker_runner, self.timeout = worker_runner, timeout

    def _configuration(self) -> dict:
        if not self.python.is_file():
            raise ValueError("FLUX runtime was not found. Launch with --flux-python PATH or set DIFFUSION_EDITOR_FLUX_PYTHON.")
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        if (cfg.get("model_id") != "black-forest-labs/FLUX.2-klein-4B" or cfg.get("is_distilled") is not True
                or cfg.get("quantization") != "bitsandbytes-int8" or cfg.get("num_inference_steps") != 4
                or cfg.get("guidance_scale") != 1.0 or not cfg.get("transformer_skip_modules")):
            raise ValueError("FLUX requires the verified distilled, protected INT8 configuration, four steps and guidance 1.")
        root = ROOT / cfg["quantized_dir"]
        if not (root / "model_index.json").is_file():
            raise ValueError("Local FLUX INT8 export is missing. Follow docs/flux-klein-int8.md before running.")
        if json.loads((root / "model_index.json").read_text()).get("is_distilled") is not True:
            raise ValueError("The local FLUX export is not marked as distilled.")
        for name in ("transformer", "text_encoder"):
            marker = root / name / "export_complete.json"
            if not marker.is_file():
                raise ValueError(f"Prepared FLUX INT8 {name} is missing.")
            meta = json.loads(marker.read_text())
            skip = cfg["transformer_skip_modules"] if name == "transformer" else []
            if (meta.get("revision") != cfg["revision"] or meta.get("bits") != 8
                    or meta.get("skip_modules", []) != skip):
                raise ValueError(f"FLUX {name} export does not match the pinned INT8 revision.")
        return cfg

    def run(self, request: FluxRequest, image: Image.Image | None = None,
            progress: Progress | None = None) -> FluxResult:
        notify = progress or (lambda *_: None)
        original = normalize_image(image) if request.operation == "Edit image" else None
        fit = ImageFit.plan(original.size, enabled=request.fit_reference, alignment=16) if original else None
        source = fit.image(original) if original else None
        if source is not None and (min(source.size) < 64 or max(source.size) > 1024
                                   or source.width % 16 or source.height % 16):
            raise ValueError("Reference dimensions must be multiples of 16 in [64,1024]. Enable reference fitting or resize explicitly.")
        cfg = self._configuration()
        fingerprint = request_digest(request, original)
        with self.owner.external_inference():
            directory = self.owner.output_dir / "flux" / uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            cfg.update(prompt=request.prompt, width=request.width, height=request.height, seed=request.seed)
            config = directory / "request.json"
            config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            command = [str(self.python), "-B", "-u", str(ROOT / "scripts/run_flux_klein_int8_reference.py"),
                       "--run-gpu", "--require-export", "--progress-json", "--config", str(config),
                       "--run-dir", str(directory / "run")]
            if source is not None:
                source.save(directory / "input.png")
                if fit.changed:
                    original.save(directory / "original-input.png")
                command += ["--image", str(directory / "input.png")]
            started = time.perf_counter()
            notify(0.01, "Starting FLUX; other inference jobs wait until this run finishes")
            try:
                self.worker_runner(command, directory / "worker.log", notify, self.timeout)
                run = directory / "run"
                report = json.loads((run / "run.json").read_text(encoding="utf-8"))
                output = run / "output.png"
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                if report.get("output_sha256") != digest or report.get("revision") != cfg["revision"]:
                    raise RuntimeError("FLUX output hash or model revision does not match its run report.")
                with Image.open(output) as opened:
                    result = opened.convert("RGB")
                if result.size != (request.width, request.height):
                    raise RuntimeError("FLUX returned unexpected output dimensions.")
                ui_report = {"backend": "FLUX.2 Klein 4B protected INT8 reference", "run_id": directory.name,
                    "request": asdict(request), "settings_digest": fingerprint, "worker": report,
                    "wall_seconds": time.perf_counter() - started, "output_sha256": digest,
                    "input_sha256": hashlib.sha256((directory / "input.png").read_bytes()).hexdigest() if source else None,
                    "reference_preprocessing": fit.to_dict() if fit else None,
                    "original_input_sha256": hashlib.sha256((directory / "original-input.png").read_bytes()).hexdigest() if fit and fit.changed else None,
                    "capabilities": ["text-to-image", "instruction image editing"],
                    "note": "Reference backend; does not implement custom DDIM or masked inpainting/outpainting."}
                report_file = directory / "ui_report.json"
                report_file.write_text(json.dumps(ui_report, indent=2) + "\n", encoding="utf-8")
                archive = directory / "run.zip"
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                    for file in directory.rglob("*"):
                        if file.is_file() and file != archive:
                            package.write(file, file.relative_to(directory))
                status = f"FLUX {request.operation.lower()} complete · 4 steps · seed {request.seed} · {ui_report['wall_seconds']:.1f}s."
                notify(1.0, "Complete")
                return FluxResult(result, status, ui_report, str(output), str(archive), str(report_file), fingerprint, directory.name)
            except Exception as exc:
                (directory / "failure.json").write_text(json.dumps({"request": asdict(request),
                    "error": f"{type(exc).__name__}: {exc}", "settings_digest": fingerprint}, indent=2), encoding="utf-8")
                raise
