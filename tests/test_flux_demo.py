"""Offline FLUX bridge safety, lifecycle, artifacts and opt-in UI integration."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import zipfile

from PIL import Image
import pytest

from diffusion_editor.demo.flux_service import FluxDemoService, FluxRequest, request_digest, run_worker
from diffusion_editor.demo.service import GenerationRequest
from test_generation import generation_service


@pytest.fixture
def flux_service(generation_service, tmp_path, monkeypatch):
    monkeypatch.setattr("diffusion_editor.demo.service.torch.cuda.is_initialized", lambda: False)
    cfg = json.loads(Path("configs/flux-klein-4b-int8.json").read_text())
    export = tmp_path / "quantized export with spaces"
    export.mkdir()
    (export / "model_index.json").write_text(json.dumps({"is_distilled": True}))
    for name in ("transformer", "text_encoder"):
        (export / name).mkdir()
        (export / name / "export_complete.json").write_text(json.dumps({"revision": cfg["revision"],
            "bits": 8, "skip_modules": cfg["transformer_skip_modules"] if name == "transformer" else []}))
    cfg["quantized_dir"] = str(export)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    calls = []

    def worker(command, log, notify, timeout):
        assert generation_service._lock.locked()
        assert generation_service._backend is generation_service._editor is generation_service._reconstructor is None
        assert "--require-export" in command and "--progress-json" in command
        local = json.loads(Path(command[command.index("--config") + 1]).read_text())
        out = Path(command[command.index("--run-dir") + 1])
        out.mkdir()
        Image.new("RGB", (local["width"], local["height"]), "red").save(out / "output.png")
        (out / "run.json").write_text(json.dumps({"revision": local["revision"],
            "output_sha256": hashlib.sha256((out / "output.png").read_bytes()).hexdigest(),
            "prompt": local["prompt"], "num_inference_steps": 4, "guidance_scale": 1.0}))
        log.write_text("synthetic offline worker; no real model loading")
        calls.append((command, local))
        notify(.7, "Synthetic sampling step")

    service = FluxDemoService(generation_service, python=sys.executable, config_path=config, worker_runner=worker)
    service.calls = calls
    return service


@pytest.mark.parametrize("field,value", [("operation", "inpaint"), ("prompt", ""), ("width", 65),
    ("height", True), ("height", 2048), ("seed", -1), ("seed", 2**32), ("seed", 1.5)])
def test_flux_request_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        FluxRequest(**(dict(operation="Generate", prompt="red mug") | {field: value}))


def test_flux_ignores_unused_image_and_preserves_literal_request(flux_service):
    before = flux_service.config_path.read_bytes()
    request = FluxRequest("Generate", 'red mug; $(literal) "quoted"\nsecond line', seed=17)
    progress = []
    result = flux_service.run(request, object(), progress=lambda *values: progress.append(values))
    command, cfg = flux_service.calls[0]
    assert "--image" not in command and cfg["prompt"] == request.prompt and cfg["seed"] == 17
    assert flux_service.config_path.read_bytes() == before
    assert result.image.size == (512, 512) and result.report["request"] == asdict(request)
    assert progress[0][0] == .01 and progress[-1] == (1., "Complete")
    with zipfile.ZipFile(result.archive_file) as package:
        assert {"request.json", "worker.log", "run/output.png", "run/run.json", "ui_report.json"} <= set(package.namelist())
        assert hashlib.sha256(package.read("run/output.png")).hexdigest() == result.report["output_sha256"]


def test_flux_upload_is_frozen_and_archived(flux_service):
    source = Image.new("RGBA", (64, 80), (20, 40, 70, 128))
    request = FluxRequest("Edit image", "Make the cup blue.", width=64, height=80)
    result = flux_service.run(request, source)
    command = flux_service.calls[0][0]
    with Image.open(command[command.index("--image") + 1]) as saved:
        assert saved.mode == "RGB" and saved.size == source.size
        assert saved.getpixel((0, 0)) == (20, 40, 70)
    assert result.report["input_sha256"] and result.digest == request_digest(request, source)
    assert request_digest(request, Image.new("RGB", source.size, "white")) != result.digest
    with zipfile.ZipFile(result.archive_file) as package:
        assert "input.png" in package.namelist()


def test_flux_explicit_fitting_keeps_original_pixels_and_provenance(flux_service):
    source = Image.new("RGB", (600, 400), (20, 40, 70))
    request = FluxRequest("Edit image", "blue cup", fit_reference=True)
    result = flux_service.run(request, source)
    command = flux_service.calls[0][0]
    with Image.open(command[command.index("--image") + 1]) as prepared:
        assert prepared.size == (608, 400)
        assert prepared.crop((0, 0, 600, 400)).tobytes() == source.tobytes()
        assert prepared.getpixel((607, 10)) == (20, 40, 70)
    assert result.report["reference_preprocessing"]["padding_right"] == 8
    assert result.report["original_input_sha256"] and result.digest == request_digest(request, source)
    with zipfile.ZipFile(result.archive_file) as package:
        assert "original-input.png" in package.namelist()


@pytest.mark.parametrize("image", [None, Image.new("RGB", (63, 64)), Image.new("RGB", (1056, 64))])
def test_flux_invalid_reference_fails_before_model_release(flux_service, image):
    sentinel = object()
    flux_service.owner._backend = sentinel
    with pytest.raises(ValueError):
        flux_service.run(FluxRequest("Edit image", "blue cup"), image)
    assert flux_service.owner._backend is sentinel and not flux_service.calls


@pytest.mark.parametrize("field,value", [("revision", "different-model"), ("bits", 4), ("skip_modules", [])])
def test_flux_rejects_incompatible_export_before_releasing_sd(flux_service, field, value):
    cfg = json.loads(flux_service.config_path.read_text())
    marker = Path(cfg["quantized_dir"]) / "transformer/export_complete.json"
    meta = json.loads(marker.read_text())
    meta[field] = value
    marker.write_text(json.dumps(meta))
    sentinel = object()
    flux_service.owner._backend = sentinel
    with pytest.raises(ValueError, match="pinned INT8 revision"):
        flux_service.run(FluxRequest("Generate", "red mug"))
    assert flux_service.owner._backend is sentinel and not flux_service.calls


def test_flux_releases_sd_cache_and_sd_reloads_afterwards(flux_service):
    owner = flux_service.owner
    owner._models()
    assert len(owner.load_calls) == 1
    flux_service.run(FluxRequest("Generate", "red mug"))
    assert owner._backend is None and not owner._lock.locked()
    owner.generate(GenerationRequest("apple", width=64, height=64, steps=3))
    assert len(owner.load_calls) == 2


def test_flux_worker_failure_is_retained_and_releases_lock(flux_service):
    def fail(*args):
        raise RuntimeError("synthetic worker failure")
    flux_service.worker_runner = fail
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        flux_service.run(FluxRequest("Generate", "red mug"))
    assert not flux_service.owner._lock.locked()
    failure = next((flux_service.owner.output_dir / "flux").glob("*/failure.json"))
    assert json.loads(failure.read_text())["request"]["prompt"] == "red mug"


def test_flux_rejects_mismatched_output_hash(flux_service):
    original = flux_service.worker_runner
    def corrupt(command, *args):
        original(command, *args)
        out = Path(command[command.index("--run-dir") + 1])
        (out / "output.png").write_bytes(b"changed")
    flux_service.worker_runner = corrupt
    with pytest.raises(RuntimeError, match="hash"):
        flux_service.run(FluxRequest("Generate", "red mug"))


def test_flux_waits_for_the_shared_sd_lock(flux_service):
    completed = threading.Event()
    errors = []
    def run():
        try:
            flux_service.run(FluxRequest("Generate", "red mug"))
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()
    with flux_service.owner._lock:
        thread = threading.Thread(target=run)
        thread.start()
        assert not completed.wait(.1) and not flux_service.calls
    thread.join(5)
    assert completed.is_set() and not errors


def test_worker_progress_protocol_and_failure_logs(tmp_path):
    progress = []
    code = "print('ordinary log',flush=True);print('FLUX_PROGRESS {\"fraction\":0.5,\"message\":\"sampling\"}',flush=True)"
    run_worker([sys.executable, "-u", "-c", code], tmp_path / "worker.log", lambda *v: progress.append(v), 10)
    assert progress == [(.5, "sampling")]
    assert "ordinary log" in (tmp_path / "worker.log").read_text()
    with pytest.raises(RuntimeError, match="code 3"):
        run_worker([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "failed.log", lambda *_: None, 10)


def test_worker_timeout_stops_only_its_child(tmp_path):
    with pytest.raises(TimeoutError):
        run_worker([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path / "timeout.log", lambda *_: None, .05)


@pytest.mark.demo
def test_flux_http_route_and_error_retains_result(flux_service):
    pytest.importorskip("gradio")
    from gradio_client import Client, handle_file
    from diffusion_editor.demo.app import create_app
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = create_app(service=flux_service.owner, flux_service=flux_service)
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, prevent_thread_lock=True,
                   allowed_paths=[str(flux_service.owner.output_dir)], quiet=True)
        client = Client(f"http://127.0.0.1:{port}", verbose=False)
        result = client.predict("Generate", "red mug", 64, 64, 42, None, False, api_name="/flux_run")
        assert result[2]["request"]["operation"] == "Generate" and Path(result[3]).is_file()
        assert Path(result[4]).is_file() and Path(result[5]).suffix == ".zip"
        source = flux_service.owner.output_dir / "test-input.png"
        Image.new("RGB", (64, 64), "white").save(source)
        edited = client.predict("Edit image", "blue cup", 64, 64, 42, handle_file(source), False, api_name="/flux_run")
        assert edited[2]["input_sha256"] and edited[2]["request"]["prompt"] == "blue cup"
        invalid = client.predict("Generate", "", 64, 64, 42, None, False, api_name="/flux_run")
        assert invalid[1].startswith("Failed:") and Path(edited[3]).is_file()
    finally:
        app.close()
