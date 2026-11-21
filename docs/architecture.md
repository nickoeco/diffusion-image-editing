# Architecture

Phase 0 establishes a small pretrained boundary: `SDBackend.load(ProjectConfig)`
returns tokenizer, text encoder, VAE, U-Net, scheduler, device, dtype and configuration.
It loads components directly, freezes parameters and enters eval mode. It performs
no DDIM updates and makes no assumptions about future trajectory storage.

Configuration loading and host inspection do not import heavyweight ML dependencies.
Logging is configured by script entrypoints, not as an import side effect. Relative
cache paths resolve against the process working directory; run from repository root.
An explicit immutable model revision accompanies every load call.

Implemented data flow: RGB image -> deterministic VAE encode -> source-conditioned
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
in this repair. The user authorized Phase 5 after the fidelity review.

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

## Phase 6 local demo

`demo/app.py` builds the optional Gradio interface and owns presentation and useful
error messages. `demo/inputs.py` validates oriented input images and masks; painted
masks are the alpha union of RGBA paint layers, never the photographic composite.
The canvas background must exactly match the original. `demo/ui_state.py` defines
mode/output selection rules and scoped CSS. Inactive mask controls are hidden
without unmounting the initialized canvas, avoiding Gradio initialization races
when returning to local editing. A dedicated custom-image upload button resets
the mask/canvas and previous results.

`demo/service.py::DemoService` lazily loads one backend from the existing offline
cache and reuses both custom pipeline wrappers. A lock serializes all model work;
the Gradio queue also limits inference concurrency to one. UI settings become a
validated immutable `DemoRequest`; there is no diffusion arithmetic in the UI.
Editing and independent source reconstruction retain their separate numerical
settings and output labels. The UI defaults to Edit-only, with VAE/DDIM/control
outputs explicitly opt-in. Service routing skips unselected branches. The custom
pipeline APIs retain their existing all-output defaults and expose optional flags
to skip baseline decoding, source-control sampling or target sampling. VAE-only
requests use the pretrained VAE wrapper with no diffusion or text execution.

Each request saves a unique output directory and ZIP with PNGs, JSON/CSV metrics,
input hashes, exact state IDs, model revision, software/device/dtype and timings.
The server binds to 127.0.0.1 without a share link, permits downloads from the run
output root and limits uploaded files to 20 MB. `demo/image_fit.py` owns the
demo-only, explicit original-to-content-to-canvas transform: proportional reduction
to a 1024-pixel long edge, followed by right/bottom replicated-edge padding to
multiples of 8. Masks share its geometry with zero padding. A session state retains
the original and pixel hashes so toggling fitting restores the source rather than
resizing the padded preview repeatedly. Core pipelines and research inputs remain
strict. Outputs and metrics use the fitted canvas; see [image fitting](demo-image-fit.md).
App construction loads no weights. HTTP checks are explicit `--run-demo` opt-ins;
default unit tests exercise inputs and service with CPU fakes and no sockets.

## Phase 7 evaluation

`evaluation/experiment_config.py` expands strict YAML studies into immutable cases
without importing ML libraries. A study changes one declared axis; IDs are safe
directory names and all settings are validated before execution. Paths resolve
against the explicit project root. Sample provenance is required.

`evaluation/experiments.py::ExperimentRunner` freezes inputs, hashes configuration,
source files, model revision and runtime, then reuses one cached backend for serial
custom-pipeline calls. It records measured U-Net forwards/batch elements with a
temporary hook, plus actual inversion/sampling IDs and refinement diagnostics.
The runner performs no diffusion arithmetic. Existing completed cases require
explicit compatible resume and checksum validation. Errors remain failed cases.

`evaluation/artifacts.py` measures floating outputs before quantization, writes
PNGs/CSV/strict JSON and retains masks. `evaluation/records.py` owns lightweight
JSON/integrity checks; `evaluation/report.py` rebuilds tables, grids and headless
Matplotlib PNG/SVG curves from saved reports without model loading. Undefined and
infinite PSNR values are distinguished; plotted nonfinite points are omitted.

Only `scripts/run_img2img_baseline.py` constructs the labelled built-in Img2Img
pipeline. It matches completed global custom cases, their input pixels, weights,
prompts, guidance and actual reverse IDs. FP32 VAE codec execution surrounds the
pipeline's supported latent input/output; random noising and reverse diffusion
remain built-in. Its random posterior/noise, update precision and lack of inversion
or a control branch are declared differences. Core modules never import it.

## Acceptance review layer

`configs/acceptance/cases.yaml` is a separate task registry, with balanced
development/holdout membership and explicit case-specific semantic, preservation
and boundary requirements. It is not a runnable sweep and its assets are pending.
`evaluation/acceptance.py` validates registration, creates blank review tables,
computes the independent three-score verdict and renders source-mask-based zooms
using Pillow/NumPy. It imports no model libraries, calls no pipeline and assigns
no visual rating. `scripts/prepare_acceptance_review.py` is the thin offline CLI.
The split/freeze protocol is binding workflow guidance; existing model runners
are unchanged and do not automatically enforce access to holdout cases.
Review tables omit reviewer identity and review-time columns; versioned raw
ratings, run IDs and output hashes provide the evidence association instead.

`demo/review_app.py` is a separate, model-free local Gradio interface for the
two-candidate development comparison. `evaluation/review_store.py` owns rating
validation, output-hash/shared-prompt checks, stale-write detection, snapshots and
atomic CSV replacement. `evaluation/review_figures.py` appends exact prompt text
to derived figures without changing the original image area. It does not change
editing algorithms or the inference demo. See [review-ui.md](review-ui.md).

## Experimental guided reconstruction extension

The [advanced reconstruction extension](advanced-reconstruction.md) uses an
output-local frozen runtime snapshot and a separate differentiable U-Net adapter
to optimize only per-timestep unconditional text inputs. Production inference
wrappers, model weights and pipeline defaults remain unchanged. Source pivots are
loss targets, never replacements for sampled states. A pivot-free replay verifies
the saved null map and calibrated execution layout. Initial separate-branch and
corrected batched-branch results remain independent artifacts. The strict source
reconstruction gate is still open, so no attention-control layer is integrated.

A separate output-local InstructPix2Pix reference calls the built-in pipeline only
from its labelled baseline script. Its eight-channel image conditioning and
three-branch guidance are not imported into the installed editor. Fixed inputs,
runtime/code hashes and measured work preserve a reviewable boundary between
this different pretrained method and the custom reconstruction implementation.
See [instruction-reference.md](instruction-reference.md).

The active research priority is the [custom DDIM route](ddim-research-plan.md).
Reference editors remain output-local evidence. The next calibration diagnostic
does not change installed inference; any future attention controller requires
the source gate, an explained architecture change and a no-op equivalence check.

The executed [calibration diagnostic](calibration-diagnostic.md) recovers actual
states by pivot-free replay, stores updated-level loss targets separately, and
instruments fixed-state optimization in an output-local adapter. Deterministic
settings are part of its experimental execution map. Parameter-value hashes
isolate FP16/FP32 arithmetic sensitivity without swapping checkpoints. No installed
controller or diffusion component is changed.

## User-requested text-to-image interface

`pipelines/generation.py` composes text conditioning, model-configured Gaussian
initialization, our DDIM sampler/CFG and FP32 VAE decoding. `DemoService.generate`
has a separate validated request/result contract, sharing the existing offline
model cache, inference lock and Gradio concurrency group with editing. The UI adds
an independent text-generation tab and `/txt2img` endpoint. Generation metadata/files are separate; no source-reference
metrics or research/acceptance claims are synthesized. See `txt2img-demo.md`.

The four-tab UI builds independent editor forms through one helper, sharing the
same service/cache/queue. Each form has separate upload state and output controls,
with scoped element IDs/CSS. In the original four-tab layout only Local edit
creates an ImageEditor; Local edit
is initially selected and all tab children remain mounted. Tab navigation has no
server callbacks that clear values or trigger inference. See `demo-tabs.md` for
the executed browser paint/switch and API regression checks.

## Forward-noising route (user-directed addition)

`diffusion/noising.py` owns the FP32 forward marginal equation. The separate
`editing/strength.py::select_forward_start` maps strength through floor to an
actual selected level, without changing inversion editing's existing ceil policy.
`editing/canvas.py` places source pixels into an explicitly expanded canvas and
defines new-area/overlap edit weights; it performs no model execution.

`pipelines/forward_editing.py::ForwardEditingPipeline` composes VAE posterior mode,
seeded CPU noise, target conditioning and our existing eta=0 sampler. Inpainting
uses a stateless after-update callback that computes known-region forward marginals
on demand at the updated ID with one fixed epsilon. It stores no inversion
trajectory, performs no null-text calibration and never substitutes source pivots.
Outpainting uses the same masked path on a new canvas. Optional pixel compositing
remains outside sampling and raw decoded output is retained.

`demo/forward_service.py` provides validated requests and reproducible artifacts,
reusing `DemoService`'s single backend and lock. CLI and Gradio call the same
service. No dependency or model download is added. Existing inference and review
interfaces remain intact.

## Task workspace presentation layer

`demo/workspace.py` builds four persistent task pages with a fixed settings
sidebar and one visible input/result work area. It owns Gradio components and
events only. `demo/workspace_run.py` translates tasks into existing generation,
forward-editing or inversion/reconstruction requests; it never implements model
math. `demo/workspace_state.py` owns model-free geometry previews, requested-mask
overlays, fixed boundary crops and input/settings fingerprints.

Every page keeps independent state; all inference shares the existing service,
model cache, lock and concurrency group. Paint surfaces remain mounted under
CSS view changes. Completion records an immutable run/settings/output association;
later edits mark prior results rather than relabelling them. `workspace.json`
and optional original-upload artifacts extend workspace archives while leaving
backend `report.json` intact. Hidden legacy forms in `app.py` / `forward_app.py`
retain old API schemas and callbacks during migration. See `workspace-ui.md`.
