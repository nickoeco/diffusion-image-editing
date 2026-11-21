# Implementation status and mathematical contracts

Phases 1–7 implement VAE/text conditioning, DDIM inversion, reverse sampling,
manual CFG, reconstruction, global/local editing, Gradio and reproducible evaluation.
`models/sd_backend.py::SDBackend.load` loads model configuration and weights;
`resolve_runtime` selects device/dtype. Scheduler configuration is loaded with
`DDIMScheduler.from_pretrained`; custom mathematics never calls its step method.

## VAE encoding and decoding (implemented)

`utils/image_utils.py::image_to_tensor` accepts PIL images or floating RGB BCHW
tensors in [0,1]. PIL inputs are EXIF-oriented and converted to RGB. Tensor layout,
range and finite values are validated. There is no implicit resize or crop.

`models/vae.py::VAEWrapper.encode_image` implements:

```text
x_normalized = 2*x - 1
z = mode(q_VAE(z | x_normalized)) * vae.config.scaling_factor
```

The model-configured spatial factor is `2**(len(block_out_channels)-1)`. Input
dimensions must be divisible by that factor; output is [B,latent_channels,H/f,W/f].
The posterior mode is also the Gaussian posterior mean. It removes random posterior
sampling variation but cannot undo the VAE bottleneck's information loss.

`VAEWrapper.decode_latents` implements:

```text
x_decoded = VAE.decode(z / vae.config.scaling_factor)
x_rgb = clamp(x_decoded / 2 + 0.5, 0, 1)
```

Both methods run in inference mode on the VAE device in FP32, validate model output
shape/finite values and return tensors on that device. Input tensors can originate
on CPU or use another floating dtype; conversion is explicit at the model boundary.
The wrapper requires an FP32 VAE. `tensor_to_images` performs a deliberate CPU
transfer and uint8 rounding only for saving/display, not for numerical evaluation.

See the pinned [AutoencoderKL documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/api/models/autoencoderkl)
for the pretrained latent scaling convention.

## Text conditioning (implemented)

`models/text_encoder.py::TextEncoderWrapper.encode_prompt` accepts a string or a
nonempty sequence of strings and returns [B,L,D] `last_hidden_state`. This is the
token sequence consumed by the U-Net, not CLIP's pooled projection. L comes from
the tokenizer's maximum length and D from the text encoder configuration.

Token IDs stay integer and move to the model device. Padding is to maximum length;
long inputs are truncated with an explicit warning. An attention mask is passed
only if `text_encoder.config.use_attention_mask` requests it. Model execution uses
eval/inference mode and the backend's CLIP dtype.

`encode_prompt_pair(prompt, negative_prompt="")` returns [2B,L,D] ordered as
`[negative_0,...,negative_B-1, positive_0,...,positive_B-1]`. A scalar negative prompt
broadcasts over B; a sequence must match B exactly. Empty negatives provide
unconditional conditioning. This method prepares the CFG batch but does not yet
implement CFG arithmetic. The CLIP contract follows the pinned
[Transformers reference](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/clip).

## Verification

Synthetic model tests check normalization endpoints, nondefault scaling, inverse
scaling, posterior mode (never sample), batches, shapes, deterministic repetition,
invalid inputs, token/embedding order, truncation, attention masks and device/dtype.
GPU tensor tests require `--run-gpu`; real cached-weight tests require
`--run-integration`. Neither default pytest nor integration tests download weights.
The opt-in `scripts/check_conditioning.py` saves a labelled original/VAE comparison
and reproducibility metadata; it supports an explicitly supplied image or a generated
test card. Real-model results are recorded separately in the Phase 1 status report.

On the tested GPU, FP16 CLIP embeddings changed by up to 0.01953125 in absolute
value between combined batch size 4 and split batch size 2 execution. Different
kernel batching can change rounding; do not require bitwise equivalence across
batch shapes. The real order check compares the wrapper with direct pretrained
execution using identical token order and batch size. Synthetic tests independently
verify that the first B entries are unconditional and the remaining B are conditional.

## Noise schedule and clean boundary (implemented)

`diffusion/schedule.py::DDIMSchedule.from_scheduler` clones the pretrained DDIM
scheduler configuration, reads its beta vector and actual inference IDs, and
checks that FP32 cumulative products match the scheduler. Training coefficients
and the original scheduler are unchanged. No adjacent-integer or fixed-stride
assumption is used by custom traversal, so nonuniform grids are supported.

```text
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_0, ..., alpha_t)
clean ID = -1; alpha_bar_clean = 1
```

The checkpoint's `set_alpha_to_one=false` would use alpha_bar_0 at the denoising
boundary. The VAE latent is actually clean, so our cloned DDIM view explicitly
sets `set_alpha_to_one=True`. Inversion and future reverse sampling must share
this choice. Training timestep 0 remains distinct from clean. Unsupported epsilon
singularities, clipping/thresholding, non-epsilon models and invalid grids fail.

For the pinned checkpoint and 50 leading steps, inversion levels are
`[-1, 1, 21, ..., 981]`; reverse levels will be `[981, ..., 21, 1, -1]`.
The terminal inverted level is 981, not an assumed 999 or pure Gaussian sample.
Requesting 1000 leading steps with offset 1 produces an out-of-range ID and is
rejected rather than silently clamped.

## DDIM inversion (implemented)

`diffusion/inversion.py::ddim_inversion_step` delegates to `diffusion/step.py::ddim_step`,
which implements the equations below for
the actual current/next higher-noise pair (t,s), where alpha_bar_s < alpha_bar_t:

```text
x0_hat = (z_t - sqrt(1-alpha_bar_t)*epsilon) / sqrt(alpha_bar_t)
z_s = sqrt(alpha_bar_s)*x0_hat + sqrt(1-alpha_bar_s)*epsilon
```

All coefficient/update arithmetic is FP32, with shape/device/finiteness checks.
No stochastic noise, clipping or library step is used. `DDIMInverter.invert`
records N+1 snapshots including clean, by explicit state ID, with query IDs.
The first noise prediction uses the lowest selected training timestep on the clean
latent; subsequent predictions use the current state timestep. At 50 steps the
query IDs are `[1, 1, 21, ..., 961]`. The clean bookkeeping ID never reaches the
U-Net. Noise prediction at the current state approximates the unknown higher-noise
state; this is not an exact inverse of future denoising.

`models/noise_predictor.py::UNetNoisePredictor` casts model inputs to U-Net runtime
dtype and returns FP32 epsilon. Phase 3 extends inversion with the shared manual
CFG helper: B conditional embeddings are valid at guidance 1; 2B paired embeddings
support nonnegative finite guidance scales.

Tests use independent NumPy float64 equations and an analytic constant-epsilon
trajectory, with irregular gaps, boundaries and deterministic repetition. Optional
reference tests compare fixed-epsilon updates with `DDIMInverseScheduler` at
1/20/50/100 steps, plus leading/linspace/trailing grid preservation. These verify
coefficient math, not equality between full neural pipelines with different
query-time conventions. Production never calls the reference step. See the
[DDIM paper](https://arxiv.org/abs/2010.02502) and pinned
[scheduler reference](https://huggingface.co/docs/diffusers/v0.35.1/en/api/schedulers/ddim_inverse).

## Reverse DDIM sampling (implemented)

`diffusion/sampler.py::ddim_sampling_step` uses the same explicit `ddim_step`
equations with alpha_bar_t < alpha_bar_s. `DDIMSampler.sample` traverses the exact
descending grid and queries the U-Net at each current noisy timestep. At 50 steps
the last call is timestep 1, followed by a step to clean ID -1 (alpha=1). There is
no model call at -1. Latents and coefficients use FP32; the U-Net may use FP16.
The supplied initial latent must belong to the highest selected level; our
reconstruction pipeline obtains it from the matching inversion result.

The final latent remains on the compute device. Optional snapshots include the
initial noisy level, every intermediate level and clean; default storage is CPU.
Tests cover irregular gaps, final clean updates, analytic constant-noise roundtrips
and Diffusers fixed-epsilon eta=0 references at 1/20/50/100 steps.

## Classifier-Free Guidance (implemented)

`diffusion/guidance.py::classifier_free_guidance` implements FP32
`epsilon = u + w*(c-u)` independently of the DDIM update. Scale 0 selects u, scale
1 selects c and larger scales extrapolate. Invalid/nonfinite scales or predictions
fail. `predict_guided_noise` accepts B conditional embeddings only at w=1, or
2B embeddings in [all unconditional, all conditional] order. For 2B, it duplicates
the latent batch once, makes one U-Net call, splits the result and applies CFG.
It keeps the paired batch at w=0/1 too; it does not change batch shape as a hidden
optimization. The pipeline uses a conditional-only B batch when a stage requests
w=1, so that the Phase 2 baseline is preserved. Negative prompts affect only CFG
stages with paired conditioning. Larger guidance can worsen source reconstruction.

## Reconstruction, metrics and previews (implemented)

`ReconstructionPipeline.reconstruct` encodes the image once, decodes a VAE baseline,
inverts with the source prompt, reverse-samples with that same prompt, then decodes
the DDIM output. It separately exposes inversion/sampling guidance, defaulting to
1/1, and validates image/prompt batch alignment. It makes no exact-recovery claim.

`evaluation/metrics.py::reconstruction_metrics` accepts one matching RGB [1,3,H,W]
pair in [0,1]. PSNR uses double-precision whole-image RGB MSE and data range 1.
SSIM uses scikit-image 0.25.2 with uniform 7×7 windows, channel averaging, sample
covariance and data range 1. H,W must be >=7; there is no resize, mask or zeroed
background. Identical images have PSNR +infinity; the saving script represents
this as null in strict JSON. Invalid images and nonfinite SSIM fail explicitly.

`utils/visualization.py::save_image_grid` saves labelled images without resizing.
Optional intermediate previews decode selected raw latent states and are labelled
with stage/timestep; noisy previews are not x0 predictions or final reconstruction.
Real smoke results and limitations are recorded in `phase3-status.md`.

## Residual-refined inversion and reconstruction selection (implemented repair)

Plain inversion estimates the next noisy state with a noise prediction at the
current state. The next-state prediction can differ, so applying a mathematically
correct reverse step need not recover the previous state. Fixed-epsilon unit tests
cannot reveal this model-dependent discrepancy. Diagnostic runs localize a large
error near the 21 -> 1 reverse transition on the measured 50-step natural image.

For inversion from source level t to noisier level s, write the actual reverse map:

```text
R_s(z) = a*z + b*epsilon_theta(z,s,c)
a = sqrt(alpha_bar_t / alpha_bar_s)
b = sqrt(1-alpha_bar_t) - a*sqrt(1-alpha_bar_s)
r(z) = R_s(z) - z_t
proposal = z - damping*r(z)/a
```

`diffusion/refinement.py::refine_inversion_step` implements this fixed-point residual
correction. The first candidate is the plain DDIM estimate. Every evaluated trial
is measured through the unchanged reverse equation at the actual destination
timestep s. The best evaluated point is retained; a non-improving trial halves
damping and restarts from the best point. Unevaluated proposals are never returned.
The budget counts all destination U-Net evaluations, including the initial check.
RMS is over the complete BCHW tensor in scaled latent units. No parameters or text
embeddings are optimized, and no autograd graph is built.

`DDIMInverter.invert(refinement_steps=0)` retains the original approximation.
Nonzero budgets attach per-timestep `RefinementDiagnostics`; defaults in the
reconstruction pipeline are 12 evaluations and tolerance 1e-6. Budget exhaustion
is reported explicitly and does not imply convergence. FP16 model rounding can
limit residual convergence despite FP32 coefficient/correction arithmetic.

Local improvement is not global monotonicity. A four-step CFG stress test found a
worse final candidate despite local corrections. `ReconstructionPipeline` therefore
also runs the plain path, evaluates both final roundtrip latents against the same
clean VAE latent, and selects the lower MSE. The returned fidelity check preserves
both errors and the refined candidate's diagnostics even on plain-DDIM fallback.
This guarantees no regression for that measured latent-MSE objective, not arbitrary
pixel/perceptual metrics or general editing quality. Both candidate images come
from the unchanged sampler and VAE decoder. Source pixels/trajectory states are
not substituted into reverse updates.

The implicit-equation view and fixed-point approach are standard numerical ideas;
see the [research-author explanation](https://developer.nvidia.com/blog/fast-inversion-for-real-time-image-editing-with-text/).
This project implements a small damped, budgeted correction, not RNRI, learned
inversion or null-text optimization. See `reconstruction-drift.md` for controls,
failed probes, actual measurements and runtime costs.

## Editing strength and global editing (implemented)

`editing/strength.py::select_edit_start` validates a finite strength in [0,1] and
selects a stored level from `schedule.inversion_timesteps`:

```text
N = number of selected inference levels (excluding clean)
k = ceil(strength * N)
t_start = inversion_timesteps[k]
noise_std = sqrt(1 - alpha_bar[t_start])
```

Decimal text arithmetic at the bin boundary avoids cases such as binary float
`0.14*50=7.000000000000001` incorrectly selecting step 8 instead of 7. The selection
stores the requested strength, actual reverse count, timestep and alpha. Strength
maps to a discrete step count, not a linearly interpolated noise level. At 50 steps:

| Strength | Reverse steps | Starting timestep |
| --- | --- | --- |
| 0 | 0 | -1 (clean) |
| 0.25 | 13 | 241 |
| 0.50 | 25 | 481 |
| 0.75 | 38 | 741 |
| 1 | 50 | 981 |

`DDIMSampler.sample(..., start_timestep=t_start)` validates membership and uses the
suffix of the shared descending grid. CLEAN gives a no-update result; a nonzero
start uses its actual stored latent and ends at clean. It never adds arbitrary
noise or substitutes an interpolated timestep.

`EditingPipeline.edit_global` handles source and target conditioning independently.
It inverts only with the source prompt, selects the exact stored state, and samples
two branches from it: source-prompt control and target edit. The branches share
reverse guidance and schedule, so the control exposes same-prompt drift at that
strength rather than comparing an edit against an unrelated full-depth run.
`pipelines/conditioning.py::encode_conditioning` provides shared batch validation
and conditional B/CFG 2B preparation. Reconstruction retains its existing behavior.

At strength 0, inversion is skipped (`inversion=None`); source and target sampling
perform no U-Net calls and return the clean latent. Both decoded outputs equal the
VAE baseline, not necessarily the original pixels. Positive strengths first execute
the full N-step inversion and retain its trajectory, followed by k steps per reverse
branch. The CLI's runtime includes the additional source-control branch.

Tests check analytic constant-noise edits, independent conditioning, matched initial
states, exact query IDs, invalid strengths/IDs, endpoints, full-strength equivalence
to reconstruction, batch errors and repeatability. Real cached-model checks verify
0/partial/full paths and source-vs-target differences, without asserting semantic
success. Global editing has no spatial preservation guarantee.

## Mask-guided local editing (implemented)

`editing/mask.py::prepare_mask` validates exact pixel alignment with the oriented
RGB image. Inputs are grayscale PIL L/1 or HW/B1HW floating/boolean tensors. RGB
masks, ambiguous layouts, incorrect dimensions/batches and nonfinite/out-of-range
weights fail explicitly. Binary masks threshold at >=0.5 and resize nearest;
soft masks resize bilinearly to the model-configured latent resolution. A vanished
tiny mask is warned about, rather than silently treated as a meaningful edit.

Feathering uses a separable FP32 Gaussian at pixel resolution, with sigma equal
to feather_radius and support ceil(3*sigma). Replicate padding and exact constant
mask handling preserve zero/full masks. The resulting soft pixel mask is then
resized bilinearly. Feathering expands the effective support beyond the original
hard region; that expansion is saved and evaluated separately.

`editing/blending.py::blend_latents` implements:

```text
z_s = M_latent*z_edit_s + (1-M_latent)*z_source_s
```

`SourceLatentBlender` prevalidates the required source states and looks them up by
the **updated** timestep s. `DDIMSampler.sample` calls its optional after_step hook
after t -> s and before saving the new snapshot. The final hook is s=-1, with the
actual clean VAE latent. A missing clean state is an error. Shape/device/dtype and
finite output checks guard callbacks. Source snapshots stay on CPU and only the
needed matching state is transferred; the latent mask moves once to the compute
device. This explicit local-edit operation does not alter reconstruction sampling.

`EditingPipeline.edit_local` shares source preparation with global editing and
applies the blender to both source-control and target branches. Strength zero
still takes the clean no-U-Net path. All-one masks match global editing, and with
all-zero latent masks the final latent is exactly the source clean latent.

`composite_pixels` separately implements
`x_final=M_pixel*x_decoded+(1-M_pixel)*x_original`. Literal zero weights preserve
original pixels exactly. `latent_blending` and `preserve_background` are independent
API flags. Disabling both gives the global path; enabling only pixel compositing
is a distinct control. Empty masks with compositing yield the original; latent-only
empty masks yield the lossy VAE baseline. Decoding spatially coupled latents can
change pixels outside a latent mask, so exact pixel preservation is not credited
to latent blending.

`evaluation/metrics.py::background_metrics` computes weighted background MAE and
MSE using weights 1-M and denominator `3*sum(1-M)`, then PSNR at data range 1.
Literal M==0 pixels have a separately normalized MAE. Empty regions return None;
defined perfect backgrounds have infinite PSNR (null in strict JSON, disambiguated
by positive region weight). No regional perceptual score is computed on zeroed
images. The CLI records requested-mask and effective-mask background scores,
uncomposited/final images and exact float masks.

## Gradio request and artifact contracts (implemented)

`demo/inputs.py::normalize_image` EXIF-orients and converts originals to RGB.
`resolve_mask` requires aligned L/1 uploads or RGBA paint layers matching the
original size. Draw mode checks the canvas background against original RGB pixels
and combines layer alpha as `1-product(1-alpha_i)`, independent of paint color and
the composited photograph. Empty painting is an actionable error; an explicitly
uploaded all-zero mask remains a valid preservation control.

`DemoRequest` validates mode, prompts, finite parameters, integer steps/seed and
boolean switches. `DemoService.run` rejects missing/misaligned masks before loading
models. The demo first applies the explicit `ImageFit` plan when `auto_fit=true`
(default), bounds the model canvas to 8–1024, then checks the actual VAE spatial
factor. Long edges above 1024 are reduced proportionally with Lanczos; edge pixels
pad the right/bottom to multiples of 8. Images below the limit are not enlarged.
Original/content/canvas sizes and pixel hashes accompany every report. Uploaded
masks may use these three coordinate spaces; original-size masks are resized
nearest for binary mode or bilinear for soft mode, then padded with zeros.
Drawing uses paint-layer alpha and still rejects stale backgrounds. A digest-bound
session state retains original pixels for toggling without cumulative padding.
Changing geometry clears masks/results. Disabling fitting preserves strict errors.
Outputs remain on the model canvas and metric denominators include its padding;
raw uploads and masks are retained when transformed. Feathering can expand the
effective mask into padding under the existing feathering semantics. No core
pipeline, VAE wrapper or research experiment silently adapts its inputs.
The first valid run initializes a cached backend and subsequent runs reuse it under
a lock. The process seed is set and recorded. Output selection routes only required
work: local/global Edit or source-control outputs call the editor; DDIM output
calls reconstruction; VAE-only uses the VAE wrapper directly without U-Net/CLIP
execution or mask validation. Selected VAE baselines reuse an already computed
baseline when available. Optional full reconstruction
on edit runs uses guidance 1/1, with separately enabled residual correction and
the existing plain/refined latent-MSE selection. Editing stays on plain inversion.

`include_vae_reconstruction`, `include_source_control` and `include_edit` are
optional editing API switches, all defaulting to true for existing callers.
Unrequested tensor/sampling fields are None, and their work is skipped. DDIM
reconstruction separately supports `include_vae_reconstruction=true` by default.
The demo explicitly disables unselected outputs and records `computed_outputs`;
changing a display checkbox is not implemented as hiding an already computed image.
At strength zero the final requested image still needs one VAE decode; it does
not imply an extra hidden VAE baseline. No sampling equation or mask convention
changes. The mode switch changes controls through scoped CSS so the paint canvas
stays initialized even when its panel is visually hidden.

Reports distinguish the matched editing source control from full source
reconstruction. Optional raw-latent previews use actual stored IDs and are labelled
as decoded latents, not clean predictions. Timing excludes metric calculation,
preview decoding and saving; model initialization is reported separately. Outputs
include input/output PNGs, effective/entered masks, exact float masks, a comparison
grid, parameters, hashes, state IDs, environment, device/dtype and peak CUDA memory.
JSON encodes infinite/undefined scores as null; CSV retains numeric infinities.
Regional denominators remain those defined above, with both effective-mask and
requested-mask evaluations. A ZIP bundles each unique run; errors remain visible
in the UI and logged with their traceback. The UI does not download model weights.

## Reproducible study execution (Phase 7)

`ExperimentSuite.cases` expands each study's sample IDs and values by changing
only its named `RunSettings` field. The checked-in plan separates plain and
residual-refined reconstruction so refinement is fixed within each step study.
Strength/guidance studies keep plain source inversion fixed. The categorical mask
study has global, latent-only, feathered-latent, pixel-only, latent+pixel and
feathered-latent+pixel controls. Pairwise comparisons isolate individual switches;
comparisons changing several switches must not be described as single effects.

Inputs are copied and checksum-verified before model loading. Each case resets
the declared seed, synchronizes CUDA around pipeline work and counts actual U-Net
forwards and batch elements. Timing excludes loading, metrics and saving; no
dedicated warmup is run, so first-case/shape overhead can remain. These are
descriptive timings, not matched-total-cost performance claims. The runner stores
actual returned IDs and selection rather than inferring states from list offsets.

JSON reports bind input/model/source/software identity and artifact checksums.
Explicit resume accepts only identical identity and unchanged completed artifacts.
A partial run can be summarized with missing cases identified, without treating
them as successes. Rebuilding figures reads measured metrics, never re-scores
quantized PNGs. Curves use actual numeric axis values; categorical mask modes are
labelled explicitly. No error bars or significance are invented for single-seed
per-image measurements.

The fixed requested mask uses the same denominator for global, latent and feather
comparisons. Effective-mask metrics remain separate. Whole-image original/VAE/
source-control/output and pre-composite outputs are independently available.
No regional SSIM or LPIPS is calculated by zeroing background. JSON null scores
with `nonfinite_metrics=positive_infinity` denote exact matches; other null region
scores remain undefined. Per-case CSV retains infinities.

The baseline explicitly uses `StableDiffusionImg2ImgPipeline`, the shared DDIM
scheduler with clean alpha=1, random VAE posterior sampling and Gaussian noise.
FP32 VAE encode/decode uses its supported four-channel latent input and latent
output, preserving the project's model precision policy. Actual callback IDs and
measured forward counts must match the selected reverse budget. Baseline floor
strength selection and custom ceil selection differ at some values; this
comparison deliberately uses matching 0.5/1 endpoints at 50 steps and rejects
mismatched budgets. It records different total work and pipeline latent precision.

See `phase7-status.md` and `experiments.md` for executed results and limitations.

## Semantic acceptance registration and evidence

`evaluation/acceptance.py` validates twelve unique cases balanced by category and
split, rejects repeated declared source groups/paths, and leaves missing asset
fields explicitly pending. Visual near-duplicate checks and verified provenance
remain dataset-audit work; different filenames alone do not prove independence.
Blank review rows are `not_run` with empty metrics/ratings. Tables retain run/output
references and raw reasons but omit reviewer identity and review timestamps.
`case_verdict` accepts
only integer 0/1/2 or missing ratings; a pass requires all three ratings to be 2,
independently of numeric preservation scores.

Boundary evidence uses EXIF-oriented, identically sized images and a partial L/1
requested mask. At threshold 0.5, a morphological contour supplies deterministic
top/right/bottom/left anchors, with ties toward its centroid. Identical clipped
source boxes are applied to original/raw/final images and enlarged with nearest
sampling (64 source pixels, 4× by default). Pre-registered additional centers are
supported and recorded. Locator, panel, mask and RGB/mask hashes are saved; the
helper refuses evidence overwrite and performs no metric recomputation or model
execution. See `acceptance.md` for the freeze protocol and rating definitions.

References: [DDIM paper](https://arxiv.org/abs/2010.02502),
[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598).

## Model-free development review

The separate [review interface](review-ui.md) reads actual generation reports and
offers per-dimension radio controls and reason fields for two candidates per
development source. Only rating fields and derived verdicts can be saved. Saved
output SHA-256 values are checked, shared prompts must match, stale table revisions
are rejected and original table bytes are snapshotted before atomic replacement.
One writer server per table is supported. No reviewer/time fields are collected.
Prompt annotation wraps complete source/target strings below comparisons, keeps
their existing pixel area intact and creates supplements for hashed run figures.
Offline persistence/figure tests are separate from opt-in Gradio construction and
browser checks. The real rating table is never populated by UI smoke tests.

## Experimental null-text input calibration

The opt-in [advanced reconstruction extension](advanced-reconstruction.md) minimizes
the MSE between our actual reverse DDIM update and the source pivot at its updated
timestep, optimizing only the unconditional text embedding. Source text, model
weights, schedule and decoder remain frozen. The actual reverse result is propagated;
no pivot/image substitution is permitted. Best evaluated embeddings are retained
and reloaded for independent sampling without source-pivot access.

The existing model predictor is inference-only; an output-local adapter enables
autograd to input embeddings without changing that predictor. Normal FP32 clones
separate inference tensors from autograd inputs. Null embeddings are keyed by actual
current timestep IDs, including the update that ends at CLEAN=-1. FP16 branch
batching is part of the calibrated map: optimizing/replaying with different layouts
measurably changes results. The batch-matched prototype has exact free replay and
substantially improved reconstruction, but latent-error and local convergence
requirements are not yet met. It is not an installed editing mode or a semantic
acceptance result.

The independent [instruction-editing reference](instruction-reference.md) uses
cached, pinned InstructPix2Pix FP16 safetensors and its native Euler ancestral
scheduler. It retains safety checking, records literal instructions, and measures
50 U-Net forwards / 150 branch-batch elements per case. Two actual GPU runs
completed with finite latents; six offline contract tests and a 24-artifact audit
pass. This output-local baseline does not alter production math or defaults.

The user has returned research priority to the
[custom DDIM implementation](ddim-research-plan.md). Next is a bounded diagnostic
of local optimization losses, gradients, effective FP16 embedding updates and
precision/budget sensitivity. Attention editing remains a conditional future
deliverable; no new controller or numerical candidate is implemented by this
documentation change.

That [diagnostic has since executed](calibration-diagnostic.md). An initial strict
optimization control failed and was retained; the original helper also varied
under repeated GPU backward execution. A separately registered deterministic
runtime passes six exact repeated controls and cap80 prefix checks. Local
gradients/updates, effective input casts and best evaluated candidates are saved.
FP16 cap80 reaches tolerance on2/6 problems; FP16/FP32 cap20 on0/6. Sixteen focused
offline tests pass and58 artifacts are verified. No production math/default change
or full-trajectory improvement is claimed.

## Text-to-image orchestration

`TextToImagePipeline.generate` draws FP32 CPU noise using a local seeded generator,
with shape `[1,vae.latent_channels,H/f,W/f]` and scheduler `init_noise_sigma`.
It encodes conditional B or paired CFG 2B text through the existing helper, then
samples from the highest actual selected ID through clean -1 using our eta=0
equations. No source image, VAE encode, inversion or calibrated embedding is used.
The FP32 VAE decodes the final latent into RGB BCHW [0,1]. Dimensions, finite
guidance/prior scale, prompt and seed are validated. This is a Gaussian-prior
generation convention, not an exact-inversion claim. The demo saves noise/output
hashes, actual state IDs, settings and runtime; source-reference metrics are null.
See `txt2img-demo.md` for the API and explicit test/verification scope.

The UI has separate Text to image, Local edit, Global edit and Reconstruction
tabs. `create_app` builds three independent editor forms with one helper; global
and reconstruction forms use hidden state instead of extra paint canvases. Mode
fields are fixed/hidden in the UI, legacy APIs remain compatible, and new per-tab
APIs have distinct suffixes. Scoped CSS avoids duplicate IDs or cross-tab hiding.
`render_children=True` retains initialized controls across switches, with Local
edit initially selected. Browser CPU smoke confirmed prompt/result retention and
continued drawing/execution after switching. Numerical paths are unchanged.

## Forward-noising Img2Img and masked canvas editing

Let z0 be the scaled posterior-mode encoding of the input/canvas and epsilon one
seeded FP32 Gaussian tensor. `forward_noise` implements
`z_t=sqrt(alpha_bar_t)*z0+sqrt(1-alpha_bar_t)*epsilon`, including clean alpha=1.
`select_forward_start` uses k=floor(strength*N) and the kth ascending grid level;
k=0 means clean, k=N means highest selected training ID. Strength 1 still includes
the actual alpha coefficient, rather than replacing z0 by pure noise. No scheduler
add_noise/step or built-in Img2Img/Inpainting pipeline executes this route.

At every reverse update t->s, masked sampling blends
`M_latent*z_edit_s+(1-M_latent)*q(z_s|z0)` with the SAME epsilon. This defines coupled
forward marginals, not independently sampled Markov paths. The clean update uses
z0 exactly. Binary/soft/feathered masks reuse existing alignment and resizing rules.
The known reference is the encoding of the full canvas, including its chosen
initialization; VAE coupling can still affect decoded known pixels. Independent
pixel compositing gives exact literal zero-mask pixels and does not prove semantic
success or seam quality. Zero steps skips all U-Net updates but retains VAE loss
and selected compositing.

Outpainting records (left,top,right,bottom) source-box coordinates and
(left,right,top,bottom) margins. Edge fill replicates source boundary pixels; neutral
fill is RGB 0.5. Exterior mask weights are one. Overlap is a linear editable band
inside only the source edges with positive expansion, strictly less than half the
smaller source dimension. Optional feathering further changes effective support.
The core API never resizes/crops. The demo/CLI require SD 1.5-compatible final
dimensions divisible by 8 and at most 1024 pixels per side.

Reports retain initial noise, masks, actual IDs/alpha, prompt/settings, seed,
software/device/precision, raw/final images, timings and JSON/CSV metrics.
Outpainted borders have no ground truth; initialized-canvas preservation scores
are explicitly not generated-content quality scores. Retained-source and fixed
requested-mask metrics are separate from effective-mask metrics. See
`forward-noising.md` for executable examples and measured validation.
