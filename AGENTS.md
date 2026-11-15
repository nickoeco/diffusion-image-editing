# Project instructions

These instructions apply to the entire repository and all future changes. Preserve
existing work. Explain major architectural or dependency changes before making them.

## Scope and implementation constraints

- Build an educational, implementation-focused Stable Diffusion image editor in
  Python >=3.10 with PyTorch. Prioritize correctness, readability and reproducibility.
- Pretrained VAE, CLIP tokenizer/text encoder and U-Net are allowed. Independently
  implement DDIM inversion, deterministic reverse sampling (eta=0), CFG, strength
  selection, mask blending, reconstruction and experiment orchestration.
- Never substitute an Img2Img/Inpainting pipeline for the core algorithm. Built-in
  pipelines are allowed only in clearly labelled reference/baseline smoke scripts.
- Verify model ID, revision, license, access and scheduler compatibility. Do not
  download large weights without confirmed hardware and user authorization.
- Keep unit tests offline, CPU-compatible and independent of model downloads.
  GPU and real-weight integration checks must be explicit opt-ins.
- No training, advanced inversion, Prompt-to-Prompt or other stretch goals until
  the core pipeline works and has passed its phase gates.

## Architectural principles

- `models/`: loading and pretrained execution only; `diffusion/`: schedule,
  mathematical updates, inversion, reverse sampling and guidance; `editing/`:
  masks and editing logic; `pipelines/`: orchestration; `evaluation/`: metrics and
  experiments; `demo/`: UI; `utils/`: image I/O, logs and reproducibility.
- Keep scripts thin. Configuration is explicit, validated and serializable.
  Do not import heavy ML libraries just to read configuration or inspect the host.
- Use type hints, concise docstrings and tensor shape/device/dtype contracts.
  Use model configuration for latent scale, dimensions and scheduler coefficients.
- CUDA FP16 is for inference on supported devices; CPU defaults to FP32. Diffusion
  coefficient/update arithmetic must use FP32 where needed. Avoid unnecessary
  transfers and retain only needed trajectory tensors. Freeze inference models.
- Do not silently swallow errors. Validate inputs and fail with actionable errors.

## Mathematical invariants for later phases

- Images: BCHW RGB, normalized to [-1,1] for VAE; latents: BCHW and model-configured
  scaling. Use posterior mean/mode for deterministic reconstruction.
- Use actual scheduler timesteps and training alpha products. Explicitly define a
  clean boundary distinct from training timestep 0 and the terminal noise level.
  Record trajectory states by timestep ID, never rely on ambiguous list offsets.
- Inversion and reverse traversal use the same levels. Document approximate
  inversion and its noise-prediction time convention; do not promise exact recovery.
- CFG order is [unconditional, conditional]; implement u + w*(c-u) explicitly.
- Mask 1 edits and 0 preserves. Validate spatial alignment and resize to latent
  resolution. Blending after a reverse update must use source state at the updated
  timestep, including the clean terminal state.
- Strength selects a valid stored level; document strength 0/1 and reverse step count.
- Latent blending and optional final pixel compositing are independent settings;
  VAE spatial coupling means latent blending alone cannot preserve exact pixels.

## Tests and evidence

- Each phase: explain purpose, I/O, assumptions and risks; implement a focused
  deliverable; run relevant tests; review mathematics, indexing, shapes, devices,
  numerical stability and complexity; report files, results and remaining blockers.
- Phase 0: configuration validation, device/dtype selection, mocked loading calls,
  offline policy, environment report, package installation and opt-in smoke entrypoints.
- Phase 1: VAE shapes/scaling/normalization, text batch order, device consistency;
  save an actual original/VAE reconstruction comparison when resources allow.
- Phase 2: schedule boundaries/order/alpha indexing, reference equation checks,
  inversion shapes/trajectory IDs/length, determinism and finite values.
- Phase 3: reverse coefficients/boundaries, CFG scales 0/1/>1 and batch shapes;
  actual end-to-end reconstruction and PSNR/SSIM against the VAE-only baseline.
- Phases 4–5: strength endpoints, source/target separation, zero/full/partial/soft
  masks, invalid dimensions, matching timestep blends and final clean step.
- Separate synthetic unit tests from real-weight/GPU tests. Report skipped and
  unexecuted integration checks honestly; mocks do not prove model compatibility.
- Experiments must save inputs/masks/prompts, seed, model revision, software/device,
  steps, strength, guidance, runtime, outputs and JSON/CSV metrics. Define regional
  denominators; never compute perceptual region metrics by zeroing background.
- Never fabricate results or claim improvement without measured evidence.

## Phase gates and workflow

0. Environment + scaffold: inspect OS/Python/dependencies/GPU memory; configure
   installation, logging and model-loading interface; offline tests; explicit real
   model smoke instructions. If unavailable, document pending verification. **Stop
   after Phase 0 and wait for the user's instruction before starting Phase 1.**
1. VAE + text: tested encode/decode and conditioning wrappers, saved VAE baseline.
2. Schedule + inversion: explicit equations, structured timestep trajectory;
   do not proceed to sampling until inversion unit tests pass.
3. Reverse sampling + reconstruction: custom eta=0 steps and CFG, PSNR/SSIM,
   intermediate visualization and same-prompt reconstruction.
4. Global editing: target conditioning, valid trajectory-based strength selection.
5. Local editing: mask preprocessing, matching-level latent blending, independent
   feathering/compositing switches, measurable background preservation.
6. Gradio: upload/draw mask, parameters, outputs/downloads, timing and useful errors;
   call our pipeline, not Diffusers' editing pipelines.
7. Evaluation: reproducible step/strength/guidance/mask ablations plus a labelled
   baseline comparison, automatic grids/plots and documented metric definitions.

Do not implement all phases in one pass. Maintain README and docs/architecture.md,
docs/implementation.md, docs/experiments.md and docs/roadmap.md as work progresses.
Completion requires real model loading, VAE, custom inversion/sampling/CFG,
reconstruction, global/local edits, demo, passing tests, two executed ablations,
measured metrics, saved examples and complete documentation. Scaffold completion
alone is not project completion.
