# Diffusion-Based Image Editing

A Computer Vision course project using pretrained Stable Diffusion 1.5 with custom
DDIM inversion, reverse sampling, Classifier-Free Guidance (CFG), and mask-guided
latent blending. **Currently at Phase 0: project initialization. Image editing is
not implemented yet.** All future changes must follow [AGENTS.md](AGENTS.md).

Project documentation is maintained in English. Workflow communication may use
English, Chinese, or both.

## Installation (PowerShell, repository root)

Python 3.11 is recommended; supported versions are 3.10–3.12. Development and model
dependencies are installed separately:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
python scripts/check_environment.py --output outputs/environment.json
```

Phase 0 offline tests require neither PyTorch, a GPU, nor model weights. Only
PyYAML, pytest, and build tools are installed. Installing the project metadata does
not establish that the full model dependency stack has been verified.

The model environment uses a pinned torch/torchvision pair and established library
versions. Candidate installation commands for this machine's NVIDIA GPU follow;
they download large runtime packages and have not yet been executed:

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
and numpy 1.26.4. This is a candidate compatibility baseline; actual GPU/model
compatibility still requires integration testing. Direct dependencies are pinned.
A full transitive dependency lock should be generated after the model environment
has been installed and verified; an unverified resolution is not a lockfile.
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
  models/sd_backend.py    Pretrained component loading (implemented)
  diffusion/              Planned: schedule, inversion, sampler, CFG
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
be run at this stage. See the [roadmap](docs/roadmap.md) for the development order
and acceptance gates. Development pauses after Phase 0 until the user requests Phase 1.

## Known limitations

No model weights, PyTorch installation, or real inference results are available
yet. Offline mocks verify loading-interface contracts only; they do not verify
CUDA, real checkpoint compatibility, or generation quality. DDIM inversion is
approximate, and the VAE introduces reconstruction error. Latent masks cannot
guarantee exact pixel preservation; final pixel compositing will be independently
configurable. Project completion is governed by the definition of done in AGENTS.md.
