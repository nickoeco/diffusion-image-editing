# Phase 1 verification record — 2026-11-15

## Deliverable

- `models/vae.py`: deterministic FP32 VAE encode/decode, RGB normalization,
  model-configured latent scaling/spatial factor, shape/range/finite checks.
- `models/text_encoder.py`: batched token-level CLIP embeddings, negative prompt
  broadcasting, explicit [unconditional batch, conditional batch] order, attention
  mask handling and warnings for truncated prompts.
- `utils/image_utils.py`: EXIF-aware RGB conversion, validated tensor conversion,
  uint8 export and labelled comparison saving; no implicit crop/resize.
- `scripts/check_conditioning.py`: opt-in real-weight verification and saved VAE
  baseline with provenance, shapes, device/dtypes, configuration and timing.
- Offline synthetic tests, explicit GPU and real-weight pytest flags, installation
  extras and a tested platform-specific dependency constraints snapshot.
- Updated README, architecture, implementation explanations, roadmap and experiment
  notes. No Phase 2 diffusion algorithm has been implemented.

## Environment and model

Installed torch 2.5.1+cu124, torchvision 0.20.1+cu124, diffusers 0.35.1,
transformers 4.44.2 and the project's fixed model dependencies in `.venv`.
PyTorch CUDA availability is true on NVIDIA GeForce RTX 4080 Laptop GPU.
The installed environment passed `pip check`.

The user explicitly authorized weight downloads. The pinned public SD1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14` was downloaded into the project cache.
The first attempt failed because an implicitly attached Hugging Face OAuth token
had expired. Retrying with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` for that command
succeeded anonymously without changing stored credentials. Later cached checks
ran without that override and without network access.

## Executed verification

| Check | Observed result |
| --- | --- |
| Default offline pytest suite | 55 passed, 3 skipped (two GPU checks and one real-model check require opt-in) |
| `pytest -q --run-gpu --run-integration` | 58 passed, no skipped or failed tests; one upstream tokenizer FutureWarning |
| `check_model.py --forward` using cache | Passed; real text encoding and U-Net output [1,4,64,64], CUDA FP16 |
| `check_conditioning.py` using cache | Passed; saved real VAE reconstruction comparison and metadata |
| `pip check` | Passed |
| `compileall -q src scripts tests` | Passed |
| Conditioning script `--help` and `git diff --check` | Passed |

The synthetic tests execute real CPU PyTorch tensors and small local models;
two explicit GPU tests verify transfer and dtype behavior. Real-model tests use
cached pretrained components and never initiate downloads.

## Saved VAE baseline

`outputs/reconstruction/phase1/` contains:

- `original.png`: generated deterministic 512×512 RGB gradient/checkerboard test
  card with a red circle and green rectangle; no external assets.
- `vae_reconstruction.png`: decoded posterior-mode latent from the real VAE.
- `comparison.png`: original and reconstruction side by side, visually inspected.
- `report.json`: configuration, model revision, software/hardware, provenance,
  shapes, device/dtypes, checks and measured runtime.

Observed image shape is [1,3,512,512]; latent shape [1,4,64,64]; VAE scaling factor
0.18215 and spatial factor 8; CFG embedding shape [4,77,768]. VAE execution is
FP32 and CLIP execution FP16 on CUDA. A cached validation run took approximately
4.50 seconds including model loading and the repeated checks, excluding image
saving/environment inspection. This is not a benchmark of isolated VAE latency.

Repeated VAE encoding of the same input passed exact equality. An initial CFG
check requiring exact equality between different batch sizes failed: FP16 CLIP
combined and split batches differed by up to 0.01953125 in absolute value. The
corrected reference check uses direct pretrained execution at identical batch size
and token order; it passes exact equality and records the split-batch difference
separately. This does not establish bitwise reproducibility across devices or
arbitrary batch shapes.

## Limits and next phase

The synthetic baseline verifies the interface; it does not establish natural-image
reconstruction quality. No PSNR/SSIM, ablations, editing results, DDIM inversion,
CFG arithmetic or UI are claimed. Outputs/cache remain excluded from Git.
The upstream tokenizer default-cleanup FutureWarning does not affect the token
embeddings used here; it remains visible rather than being globally suppressed.

Phase 1 is ready for review. Next is Phase 2: scheduler-derived noise levels,
explicit clean boundary and deterministic approximate inversion, with tests of
equations and timestep alignment before reverse sampling is developed.
