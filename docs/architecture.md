# Architecture

Phase 0 establishes a small pretrained boundary: `SDBackend.load(ProjectConfig)`
returns tokenizer, text encoder, VAE, U-Net, scheduler, device, dtype and configuration.
It loads components directly, freezes parameters and enters eval mode. It performs
no DDIM updates and makes no assumptions about future trajectory storage.

Configuration loading and host inspection do not import heavyweight ML dependencies.
Logging is configured by script entrypoints, not as an import side effect. Relative
cache paths resolve against the process working directory; run from repository root.
An explicit immutable model revision accompanies every load call.

Future data flow: RGB image -> deterministic VAE encode -> source-conditioned
custom inversion -> trajectory indexed by explicit noise-level ID -> custom reverse
sampling with target CFG -> aligned optional mask blending -> VAE decode -> optional
pixel composite. The clean boundary needs its own identity, distinct from training
timestep zero. Phase 2 implements and tests this contract before any editing code.

`DDIMSchedule` owns CPU FP32 training coefficients and ordered inference IDs.
It clones the backend's scheduler only to obtain the grid/configuration, with
an explicit alpha=1 clean boundary. It never uses the scheduler's step method.
`DDIMInverter` consumes a noise-predictor callable and returns an `InversionResult`
containing snapshots indexed by timestep, traversal/query IDs, schedule and guidance
configuration. The mapping prevents adding/removing states; its tensors should be
treated as read-only by consumers. `UNetNoisePredictor` casts into the U-Net dtype
and returns FP32 noise, separating numerical tests from pretrained execution.

Diffusion mathematics must not depend on UI or disk I/O. Pipeline orchestration
combines the modules; evaluation and Gradio call that pipeline. Full Diffusers
pipelines are confined to labelled reference/baseline scripts. No circular UI/model
dependencies or generic plugin framework is needed.

Phase 1 adds `VAEWrapper` and `TextEncoderWrapper` on top of this boundary.
Image conversion lives in `utils/image_utils.py`; it does not resize or crop inputs.
The wrappers accept pretrained components directly, allowing synthetic PyTorch
models in offline tests without importing Diffusers or Transformers.
`scripts/check_conditioning.py` orchestrates a real-weight smoke check and saving
results, while model wrappers remain independent of file paths and UI state.

Runtime policy: CUDA auto uses FP16 U-Net/CLIP, CPU uses FP32, VAE always uses FP32.
The VAE wrapper casts its inputs explicitly; CLIP token IDs stay integer while
output embeddings use the model dtype. Coefficients/critical math will use FP32.
Inversion retains snapshots on CPU by default to limit GPU memory; callers can
explicitly select another storage device. GPU state updates remain FP32.
Phase 3 adds `diffusion/guidance.py` for explicit batched CFG and `sampler.py` for
the shared descending grid. Both inversion and sampling call the same guidance
helper and `step.py` for the shared eta=0 equations, with direction-specific
coefficient checks. The raw U-Net adapter does not itself implement guidance.

`pipelines/reconstruction.py::ReconstructionPipeline` composes VAE, text, inversion
and sampling without performing disk I/O or metric calculations. Inversion and
sampling receive the exact same schedule object. Its structured output retains
original, VAE-only and DDIM images, plus inversion and optional sampling histories.
`evaluation/metrics.py` defines single-image RGB PSNR/SSIM independently;
`scripts/test_reconstruction.py` handles reproducible runs, saving and previews.
There is no editing mask, strength selection or target prompt in Phase 3.

Phase 4 adds `editing/strength.py` for discrete level selection and
`pipelines/editing_pipeline.py::EditingPipeline.edit_global` for source/target
orchestration. The sampler accepts an explicit selected start ID and traverses
only the suffix of the same reverse grid; an unknown ID fails. No timestep or
coefficient is reconstructed from a strength float.

Global editing encodes once, inverts with source conditioning, then looks up the
selected source state by timestep. Source-control and target branches both start
from that identical state with the same reverse settings. Strength 0 skips U-Net
execution and returns VAE-only output with `inversion=None` and clean-only sampling
histories. For positive strength, the full source trajectory remains available.
`pipelines/conditioning.py` shares validated B/2B prompt preparation between the
reconstruction and editing pipelines. The global API has no mask or blending step.

## Reconstruction-fidelity repair (before Phase 5)

`diffusion/refinement.py` numerically corrects an inverted candidate against the
same reverse DDIM function used by sampling. It uses inference-only, FP32 residual
arithmetic, a fixed U-Net evaluation budget, optional early stopping and best-point
acceptance with damping. `DDIMInverter` defaults to plain DDIM and exposes this
option explicitly; returned per-level diagnostics record initial/final RMS,
evaluation counts and convergence status. Initial prediction IDs retain their
existing meaning; extra destination queries are represented by these diagnostics.

`ReconstructionPipeline` defaults to 12 residual evaluations and RMS tolerance
1e-6, then compares the refined and plain *independently sampled* roundtrip latents
against the VAE-encoded source. It returns the lower-latent-MSE candidate and a
`ReconstructionFidelityCheck` containing the selection, both errors and candidate
refinement diagnostics, including when plain DDIM is selected. Local residual
improvement alone cannot justify accepting a global result. Original image pixels
are used for VAE encoding/evaluation; sampling consumes terminal latent and text.

The global editor exposes refinement as opt-in (default zero), preserving its
measured baseline and matched source/target comparison. Refinement targets the
source inversion settings; different reverse guidance or target text is a changed
mapping and does not inherit a source reconstruction guarantee. Target edits
intentionally change semantics, so original-latent MSE would penalize desired
modifications. The repair performs no model training, new checkpoint loading,
source-trajectory replay in sampling or pixel compositing
in this repair. Phase 5 remains unstarted while fidelity is reviewed.

## Phase 5 local editing

The user subsequently authorized Phase 5. `editing/mask.py` owns mask validation,
alignment, thresholding, optional Gaussian feathering and latent resizing.
`editing/blending.py` implements the explicit mixing equation and a source-state
lookup callback. The sampler has a small validated `after_step(latent,updated_id)`
hook; diffusion code does not import mask/editing modules. The hook is absent for
global editing and reconstruction, preserving their numerical paths.

`EditingPipeline` shares a private prepared source context between global/local
orchestration: conditioning, VAE encode, inversion, selected level and start latent.
Local sampling passes the same stateless source blender to source-control and
target branches. Only the updated IDs in the selected schedule suffix are allowed;
the clean state must be present. Optional final pixel compositing is outside the
sampler, and raw decoded outputs remain accessible for independent evaluation.

`LocalEditingResult` extends the global output contract with effective pixel/latent
masks, both uncomposited images and the independent operation flags. Background
metrics are separate from whole-image reconstruction metrics. No UI or high-level
Diffusers editing pipeline is introduced; prior reconstruction fidelity regression
checks remain part of the full suite.
