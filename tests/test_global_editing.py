from types import SimpleNamespace

import pytest
import torch

from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.pipelines.editing_pipeline import EditingPipeline


class StubText:
    """Known prompt values make routing errors observable without CLIP weights."""
    def encode_prompt(self, prompt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        return torch.tensor([{"": 0., "src": 0.2, "tgt": 0.8}[p] for p in prompts]).reshape(-1, 1, 1)

    def encode_prompt_pair(self, prompt, negative_prompt=""):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        negatives = [negative_prompt] * len(prompts) if isinstance(negative_prompt, str) else negative_prompt
        if len(negatives) != len(prompts):
            raise ValueError("negative batch mismatch")
        return torch.cat((self.encode_prompt(negatives), self.encode_prompt(prompts)))


@pytest.fixture
def editor(monkeypatch):
    # Isolate orchestration; real wrapper construction is exercised in integration.
    schedule = DDIMSchedule(torch.full((7,), 0.02), (6, 2, 0))
    monkeypatch.setattr("diffusion_editor.pipelines.editing_pipeline.DDIMSchedule.from_scheduler",
                        lambda scheduler, steps: schedule)
    pipeline = EditingPipeline.__new__(EditingPipeline)
    pipeline.backend = SimpleNamespace(device="cpu", scheduler=None)
    pipeline.vae = SimpleNamespace(encode_image=lambda x: x * 2 - 1,
                                   decode_latents=lambda z: (z / 2 + 0.5).clamp(0, 1))
    pipeline.text = StubText()
    calls = []
    def predict(z, t, e):
        calls.append((t, z.clone(), e.clone()))
        return torch.zeros_like(z) + e[:, 0, 0].reshape(-1, 1, 1, 1)
    pipeline.predict_noise = predict
    return pipeline, calls


@pytest.mark.parametrize("strength,k,t", [(0, 0, -1), (0.1, 1, 0), (0.5, 2, 2), (1, 3, 6)])
def test_strength_routes_exact_stored_state_and_source_target(editor, strength, k, t):
    pipeline, calls = editor
    image = torch.ones(1, 3, 8, 12) * 0.5
    result = pipeline.edit_global(image, "src", "tgt", strength, 1, 3, return_trajectory=True)
    assert result.selection.reverse_steps == k and result.selection.start_timestep == t
    assert len(result.target_sampling.timesteps) == k + 1
    torch.testing.assert_close(result.source_reconstruction, result.vae_reconstruction, atol=1e-6, rtol=0)
    if k == 0:
        assert calls == [] and result.inversion is None
        torch.testing.assert_close(result.edited_image, result.vae_reconstruction, rtol=0, atol=0)
    else:
        assert len(calls) == 3 + 2 * k  # Full source inversion, source control, target edit.
        torch.testing.assert_close(calls[3][1], result.inversion.trajectory[t])
        torch.testing.assert_close(calls[3 + k][1], result.inversion.trajectory[t])
        assert all(torch.all(call[2] == 0.2) for call in calls[:3 + k])
        assert all(torch.all(call[2] == 0.8) for call in calls[3 + k:])
        alpha = result.selection.alpha_cumprod
        expected = (image * 2 - 1) + ((1 - alpha) / alpha) ** 0.5 * (0.2 - 0.8)
        torch.testing.assert_close(result.target_sampling.final_latent, expected, atol=1e-6, rtol=0)
        assert not torch.equal(result.edited_image, result.source_reconstruction)
    assert result.edited_image.shape == image.shape


def test_same_prompt_and_repeatability(editor):
    pipeline, _ = editor
    image = torch.ones(2, 3, 8, 12) * 0.5
    first = pipeline.edit_global(image, ["src", "src"], ["src", "src"], 0.5, 3, 3)
    repeated = pipeline.edit_global(image, ["src", "src"], ["src", "src"], 0.5, 3, 3)
    torch.testing.assert_close(first.edited_image, first.source_reconstruction, rtol=0, atol=0)
    torch.testing.assert_close(first.edited_image, repeated.edited_image, rtol=0, atol=0)


def test_batch_errors(editor):
    pipeline, _ = editor
    image = torch.zeros(2, 3, 8, 12)
    with pytest.raises(ValueError, match="batch sizes"):
        pipeline.edit_global(image, "src", ["tgt", "tgt"], num_steps=3)
    with pytest.raises(ValueError, match="batch sizes"):
        pipeline.edit_global(image, ["src", "src"], "tgt", num_steps=3)
