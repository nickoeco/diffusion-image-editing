# Diffusion-Based Image Editing

A Computer Vision course project using pretrained Stable Diffusion 1.5 with custom
DDIM inversion, reverse sampling, Classifier-Free Guidance (CFG), and mask-guided
latent blending. **Phase 2 adds a scheduler-aligned noise grid and custom DDIM
inversion on top of VAE/CLIP conditioning. Reverse sampling and image editing are
not implemented yet.** All future changes must
follow [AGENTS.md](AGENTS.md).

Project documentation is maintained in English. Workflow communication may use
English, Chinese, or both.

## Installation (PowerShell, repository root)

Python 3.11 is recommended; supported versions are 3.10–3.12. Development and model
dependencies are installed separately:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
python scripts/check_environment.py --output outputs/environment.json
```

The current offline suite uses real CPU PyTorch tensors and small synthetic models;
it requires no GPU, pretrained weights, Diffusers, or Transformers. The `test` extra
installs its tensor/image dependencies. The original Phase 0 tests needed only the
lighter `dev` extra. Installing the project metadata alone does not verify the full
model dependency stack. CUDA and real-weight tests require explicit opt-in flags.

The model environment uses a pinned torch/torchvision pair and established library
versions. For NVIDIA GPU execution, install the CUDA wheels first instead of the
CPU wheel above, then install the complete model stack:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --output outputs/environment-model.json
```

For CPU environments, replace `cu124` with `cpu`. CPU execution is recommended for
lightweight tests; full model loading and generation are slow and require
substantial RAM. PyTorch CUDA wheels include the runtime libraries, so a separate
full CUDA Toolkit installation is not required for inference. Use the environment
script to check the driver and actual PyTorch CUDA availability.

Pinned versions include torch 2.5.1, torchvision 0.20.1, diffusers 0.35.1,
transformers 4.44.2, accelerate 1.0.1, safetensors 0.6.2, huggingface-hub 0.34.4,
and numpy 1.26.4. This combination passed Phase 1 CUDA component integration tests
on the recorded machine; downstream DDIM/editing behavior remains unverified.
Direct dependencies are pinned. A [tested Windows/Python 3.11/CUDA 12.4 constraints
snapshot](configs/constraints-windows-py311-cu124.txt) records the installed versions.
Use it with `pip install -c configs/constraints-windows-py311-cu124.txt -r requirements.txt`
after installing the CUDA torch wheels. It is not a cross-platform lockfile.
Gradio, OpenCV, plotting, and metric libraries will be introduced in their phases.

## Model setup and smoke checks

- ID: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- The [official model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  and [access/file metadata](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5)
  were checked on 2026-11-15: the repository was public and ungated, with the
  CreativeML OpenRAIL-M license. Use and distribution must comply with that license.
- The Diffusers-format checkpoint includes CLIP, a VAE, a four-channel U-Net,
  scheduler configuration, and FP16 safetensors. Only epsilon-prediction base
  models are supported; inpainting checkpoints are not compatible substitutes.
- Weight downloads and caching require several GB; reserve approximately 10 GB
  of disk space. An RTX 4080 Laptop GPU with 12 GB VRAM is a starting point for
  512×512 inference with batch size 1 and CUDA FP16. Measure actual peak memory.
- U-Net/CLIP precision follows the runtime policy; the VAE uses FP32. The cache is
  located at `.cache/huggingface`.

The default configuration is strictly offline. These commands require both the
model dependencies and cached weights:

```powershell
.\.venv\Scripts\python.exe scripts/check_model.py --forward
.\.venv\Scripts\python.exe scripts/test_generation.py --steps 20
```

After user confirmation, explicitly add `--allow-download` for the first download.
Do not use that flag in unit tests. `check_model.py` verifies component loading,
text encoding, and one U-Net forward pass. `test_generation.py` is a **built-in
text-to-image reference smoke check**, saving `outputs/generation/reference.png`.
It is not the final editing algorithm and retains the built-in safety checker.
Neither script downloads weights by default; disabling `local_files_only` in the
configuration also permits network access.

## Architecture and future usage

```text
configs/                  Explicit YAML configuration
src/diffusion_editor/
  config.py               Configuration validation
  models/                Component loading, VAE and CLIP wrappers (implemented)
  diffusion/              Schedule and inversion; sampler and CFG planned
  editing/                Planned: mask, blending, editor
  pipelines/              Planned: reconstruction and editing orchestration
  evaluation/             Planned: metrics and reproducible experiments
  demo/                   Planned: Gradio interface
  utils/                  Environment diagnostics and logging
scripts/                  Environment checks, component forward, reference generation
tests/                    Offline CPU unit tests
docs/                     Architecture, mathematical contracts, roadmap, experiment records
assets/examples/          Future input examples with provenance and usage permissions
outputs/                  Runtime artifacts excluded from version control
```

The demo, editing commands, and experiment commands are not implemented and cannot
be run at this stage. See the [roadmap](docs/roadmap.md) for development gates.

## Phase 1: VAE and text conditioning

```python
from PIL import Image
from diffusion_editor.config import load_config
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.models.text_encoder import TextEncoderWrapper

backend = SDBackend.load(load_config("configs/default.yaml"))  # cached weights
vae = VAEWrapper(backend.vae)
text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
with Image.open("input.png") as image:
    latent = vae.encode_image(image)
reconstruction = vae.decode_latents(latent)  # RGB BCHW [0,1], on VAE device
conditioning = text.encode_prompt_pair(["a red apple", "a blue apple"])
# [2B,L,D]: all negative/unconditional embeddings, then all positive embeddings
```

Image dimensions must be divisible by the configured VAE spatial factor. The
wrapper never silently resizes/crops; latent scaling comes from VAE configuration.
Posterior mode makes repeated encoding deterministic while preserving VAE loss.
Text prompts are padded/truncated to CLIP's token limit with a truncation warning.

```powershell
.\.venv\Scripts\python.exe scripts/check_conditioning.py
.\.venv\Scripts\python.exe scripts/check_conditioning.py --image input.png
.\.venv\Scripts\python.exe -m pytest -q --run-gpu
.\.venv\Scripts\python.exe -m pytest -q --run-integration --run-gpu
```

The first script defaults to a deterministic 512×512 synthetic test card, saving
`original.png`, `vae_reconstruction.png`, labelled `comparison.png`, and configuration,
device/dtype/shape/provenance/timing information in `report.json` under
`outputs/reconstruction/phase1/`. This is a VAE baseline, not a DDIM reconstruction
or a natural-image quality benchmark. Use `--image` to verify a real photograph.
The script defaults to cached weights; downloads require `--allow-download`.
Pytest integration checks always use local cache and never download weights.
Executed checks, numerical caveats, and output paths are recorded in the
[Phase 1 status report](docs/phase1-status.md).

## Phase 2: scheduler and DDIM inversion

```python
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.diffusion.inversion import DDIMInverter
from diffusion_editor.models.noise_predictor import UNetNoisePredictor

schedule = DDIMSchedule.from_scheduler(backend.scheduler, num_steps=50)
source = text.encode_prompt("a red apple")  # [B,L,D], conditional-only
result = DDIMInverter(schedule, UNetNoisePredictor(backend.unet)).invert(latent, source)
noisy_latent = result.final_latent
source_at_t = result.trajectory[result.timesteps[-1]]
```

The schedule preserves the pretrained training coefficients and actual inference
IDs. The separate clean ID is `-1` with alpha 1; a cloned scheduler explicitly uses
`set_alpha_to_one=True` for this boundary. The original scheduler is unchanged.
Snapshots are FP32 on CPU by default. At the clean state, the first U-Net query
uses the lowest selected training timestep; `-1` is never sent to the network.
Inversion is approximate. Its final state is not necessarily pure Gaussian noise.
Guidance scale 1 is supported; manual CFG is deferred to Phase 3.

```powershell
# Reuse the Phase 1 test card, or pass --image with a valid input image.
.\.venv\Scripts\python.exe scripts/test_inversion.py --steps 50
```

This cached-weight script saves `outputs/inversion/phase2/trajectory.pt` and
`report.json`, including state/query IDs, alpha coefficients, runtime, peak allocated
CUDA memory and environment. Load the locally generated trajectory with
`torch.load(path, weights_only=True)`; integer IDs map to tensors. This performs
no reverse reconstruction. See the [Phase 2 report](docs/phase2-status.md) and
[equation conventions](docs/implementation.md).

## Known limitations

Offline synthetic tests verify tensor contracts and equations; actual pretrained
verification is tracked separately in the phase status records. DDIM inversion is
approximate, and the VAE introduces reconstruction error. Latent masks cannot
guarantee exact pixel preservation; final pixel compositing will be independently
configurable. Project completion is governed by the definition of done in AGENTS.md.
