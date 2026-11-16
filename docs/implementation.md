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
models, bounds image sides to 8–1024, then checks the actual VAE spatial factor.
The first valid run initializes a cached backend and subsequent runs reuse it under
a lock. The process seed is set and recorded. Local/global modes call our editor;
Reconstruct source calls our reconstruction pipeline. Optional full reconstruction
on edit runs uses guidance 1/1, with separately enabled residual correction and
the existing plain/refined latent-MSE selection. Editing stays on plain inversion.

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

References: [DDIM paper](https://arxiv.org/abs/2010.02502),
[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598).
