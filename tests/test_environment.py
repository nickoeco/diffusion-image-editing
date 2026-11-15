import json
from types import SimpleNamespace
import sys

from diffusion_editor.utils.environment import inspect_environment, write_report


def test_report_without_torch(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("diffusion_editor.utils.environment.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout="GPU, total, free, driver"))
    report = inspect_environment()
    assert report["python"]
    assert "torch_error" in report
    assert "packages" in report
    path = tmp_path / "nested" / "report.json"
    write_report(path, report)
    assert json.loads(path.read_text()) == report


def test_no_nvidia_smi(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi unavailable")
    monkeypatch.setattr("diffusion_editor.utils.environment.subprocess.run", unavailable)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert "nvidia_smi_error" in inspect_environment()
