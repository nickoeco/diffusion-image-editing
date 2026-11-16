"""Offline experiment plans, budgets, integrity and real measured artifacts."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from diffusion_editor.config import ProjectConfig
from diffusion_editor.evaluation.experiment_config import ExperimentSuite, RunSettings, SampleSpec, StudySpec, load_suite
from diffusion_editor.evaluation.experiments import ExperimentRunner
from diffusion_editor.evaluation.records import validate_completed
from test_reconstruction import FakeVAE, FakeText, FakeTokenizer, FakeUNet, FakeScheduler


@pytest.fixture
def setup(tmp_path):
    image = Image.fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3))
    image.save(tmp_path / "image.png")
    mask = Image.new("L", image.size)
    mask.paste(255, (8, 4, 18, 12))
    mask.save(tmp_path / "mask.png")
    sample = SampleSpec("coffee", "image.png", "coffee", "pencil", "synthetic offline fixture", "mask.png")
    study = StudySpec("strength", "edit", ("coffee",), "strength", (0, 0.5, 1), RunSettings(steps=3))
    suite = ExperimentSuite("test", (sample,), (study,))
    calls = []
    def loader(config):
        calls.append(config)
        return SimpleNamespace(vae=FakeVAE(), tokenizer=FakeTokenizer(), text_encoder=FakeText(),
                               unet=FakeUNet(), scheduler=FakeScheduler(), device="cpu")
    environment = lambda: {"packages": {"torch": "synthetic"}, "torch": {"cuda_available": False}}
    runner = ExperimentRunner(ProjectConfig(device="cpu"), backend_loader=loader, environment_reader=environment)
    return suite, runner, calls


def test_plan_has_one_variable_and_expected_34_cases():
    suite = load_suite("configs/experiments/phase7.yaml")
    assert len(suite.cases()) == 34
    for study in suite.studies:
        cases = [case for case in suite.cases() if case.study == study.id]
        for case in cases:
            assert replace(case.settings, **{case.axis: getattr(study.base, case.axis)}) == study.base
    result = subprocess.run([sys.executable, "-c", "import sys; from diffusion_editor.evaluation.experiment_config import load_suite; load_suite('configs/experiments/phase7.yaml'); assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key,value", [("steps", True), ("steps", 201), ("seed", -1), ("guidance", float("inf")),
                                      ("strength", "bad"), ("mask_mode", "bad"), ("refinement_steps", -1)])
def test_invalid_settings(key, value):
    with pytest.raises(ValueError):
        RunSettings(**{key: value})


def test_invalid_plans_and_unknown_yaml(setup, tmp_path):
    suite, runner, calls = setup
    with pytest.raises(ValueError, match="distinct"):
        replace(suite.studies[0], values=(1, 1.0))
    with pytest.raises(ValueError, match="Unknown sample"):
        replace(suite, studies=(replace(suite.studies[0], samples=("missing",)),))
    with pytest.raises(ValueError, match="requires a mask"):
        replace(suite, samples=(replace(suite.samples[0], mask=None),), studies=(replace(suite.studies[0], axis="mask_mode", values=("global", "latent_only")),))
    with pytest.raises(ValueError, match="IDs"):
        replace(suite, name="../escape")
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nsamples: []\nstudies: []\nunknown: true\n")
    with pytest.raises(ValueError, match="Invalid experiment"):
        load_suite(path)
    with pytest.raises(ValueError, match="local_files_only"):
        ExperimentRunner(ProjectConfig(local_files_only=False))


def test_run_counts_artifacts_and_resume_without_loading(setup, tmp_path):
    suite, runner, calls = setup
    output = tmp_path / "out"
    reports = runner.run(suite, output, root=tmp_path)
    assert len(calls) == 1
    assert [r["budget"]["unet_calls"] for r in reports] == [0, 7, 9]
    assert reports[1]["budget"]["unet_batch_elements"] == 11
    assert reports[1]["selection"]["reverse_steps"] == 2
    assert reports[1]["sampling_timesteps"][-1] == -1
    assert reports[1]["metrics"]["output_requested_background_weight_pixels"] > 0
    assert "metrics.csv" in reports[0]["artifacts"] and "output.png" in reports[0]["artifacts"]
    assert "output_vs_vae_psnr_db" in reports[0]["nonfinite_metrics"]
    assert reports[0]["metrics"]["output_vs_vae_psnr_db"] is None
    runner._loader = lambda config: pytest.fail("Resume must reuse completed artifacts without loading")
    runner.backend = None
    assert runner.run(suite, output, root=tmp_path, resume=True) == reports
    with pytest.raises(ValueError, match="nonempty"):
        runner.run(suite, output, root=tmp_path)
    with pytest.raises(ValueError, match="identical"):
        runner.run(replace(suite, name="changed"), output, root=tmp_path, resume=True)
    path = output / "runs" / reports[0]["case"]["id"] / "output.png"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        runner.run(suite, output, root=tmp_path, resume=True)


def test_mask_modes_and_reconstruction(setup, tmp_path):
    suite, runner, calls = setup
    masks = replace(suite.studies[0], id="masks", axis="mask_mode",
                    values=("global", "latent_only", "latent_feather", "pixel_only", "latent_pixel", "latent_feather_pixel"))
    reconstruct = StudySpec("reconstruction", "reconstruct", ("coffee",), "guidance", (0, 1),
                            RunSettings(steps=3, guidance=1, refinement_steps=3))
    reports = runner.run(replace(suite, studies=(masks, reconstruct)), tmp_path / "masks", root=tmp_path)
    by_mode = {r["case"]["settings"]["mask_mode"]: r for r in reports[:6]}
    assert by_mode["latent_pixel"]["metrics"]["output_requested_background_mae"] == 0
    assert by_mode["pixel_only"]["metrics"]["output_requested_zero_mask_mae"] == 0
    assert reports[-1]["fidelity_check"] and reports[-1]["refinement_diagnostics"]


def test_missing_inputs_and_failures_are_not_completed(setup, tmp_path):
    suite, runner, calls = setup
    with pytest.raises(FileNotFoundError):
        runner.run(replace(suite, samples=(replace(suite.samples[0], image="missing.png"),)), tmp_path / "missing", root=tmp_path)
    assert not calls
    def fail(config):
        raise RuntimeError("deliberate model loading failure")
    runner._loader = fail
    with pytest.raises(RuntimeError, match="deliberate"):
        runner.run(suite, tmp_path / "failed", root=tmp_path)
    directory = tmp_path / "failed/runs" / suite.cases()[0].id
    assert (directory / "error.json").exists()
    assert validate_completed(directory, "irrelevant") is None


def test_invalid_evaluation_mask_fails_before_loading(setup, tmp_path):
    suite, runner, calls = setup
    Image.new("RGB", (24, 16)).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="grayscale and exactly aligned"):
        runner.run(suite, tmp_path / "invalid", root=tmp_path)
    assert not calls


@pytest.mark.parametrize("change", ["input", "source", "software", "snapshot"])
def test_resume_rejects_changed_experiment_identity(setup, tmp_path, change):
    suite, runner, calls = setup
    output = tmp_path / "resume"
    runner.run(suite, output, root=tmp_path)
    if change == "input":
        Image.new("RGB", (24, 16), "red").save(tmp_path / "image.png")
    elif change == "source":
        code = tmp_path / "src/diffusion_editor/changed.py"
        code.parent.mkdir(parents=True)
        code.write_text("# changed source identity\n")
    elif change == "software":
        runner._environment = lambda: {"packages": {"torch": "changed"}, "torch": {"cuda_available": False}}
    else:
        (output / "inputs/coffee/image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identical|Changed input snapshot"):
        runner.run(suite, output, root=tmp_path, resume=True)
    assert len(calls) == 1


def test_saved_report_plots_and_grids(setup, tmp_path):
    pytest.importorskip("matplotlib")
    from diffusion_editor.evaluation.report import summarize
    suite, runner, calls = setup
    output = tmp_path / "summary"
    runner.run(suite, output, root=tmp_path)
    summary = summarize(output)
    assert summary["completed_cases"] == 3 and not summary["missing_cases"]
    assert (output / "summary/strength.svg").is_file()
    assert (output / "summary/strength--coffee.png").is_file()
    assert (output / "summary/metrics.csv").is_file()
