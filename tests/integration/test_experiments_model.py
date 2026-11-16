"""Opt-in tiny real-weight orchestration and labelled built-in baseline check."""
from pathlib import Path
import runpy

import pytest


@pytest.mark.integration
@pytest.mark.gpu
def test_real_suite_and_diffusers_baseline(tmp_path):
    from PIL import Image
    from diffusion_editor.config import ProjectConfig
    from diffusion_editor.evaluation.experiment_config import ExperimentSuite, SampleSpec, StudySpec, RunSettings
    from diffusion_editor.evaluation.experiments import ExperimentRunner
    image = Image.open("assets/examples/coffee.png").convert("RGB").resize((64, 64))
    image.save(tmp_path / "coffee.png")
    suite = ExperimentSuite("real_smoke", (SampleSpec("coffee", "coffee.png", "a photo of coffee", "a pencil drawing of coffee", "CC0; resized integration fixture"),),
                            (StudySpec("strength_global", "edit", ("coffee",), "strength", (0.5, 1.0), RunSettings(steps=4, guidance=3)),))
    output = tmp_path / "experiment"
    runner = ExperimentRunner(ProjectConfig())
    custom = runner.run(suite, output, root=tmp_path)
    baseline_module = runpy.run_path(str(Path("scripts/run_img2img_baseline.py").resolve()))
    baseline = baseline_module["run_baselines"](output, root=tmp_path, backend_loader=lambda config: runner.backend)
    assert len(baseline) == 2
    assert baseline[0]["sampling_timesteps"] == custom[0]["sampling_timesteps"][:-1]
    assert baseline[0]["budget"]["unet_calls"] == 2
    assert custom[0]["budget"]["unet_calls"] == 8
    assert baseline[0]["vae_dtype"] == "torch.float32"
