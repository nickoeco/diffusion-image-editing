# Phase 5 verification record — 2026-11-15

## Deliverable

- `editing/mask.py`: strict aligned masks, binary threshold, soft weights,
  explicit batch broadcast, Gaussian feathering and latent resizing.
- `editing/blending.py`: manual `M*edit+(1-M)*source`, source lookup by updated
  timestep and independent pixel compositing.
- `DDIMSampler.after_step`: validated post-update FP32 hook including the clean
  terminal state, absent from global/reconstruction calls by default.
- `EditingPipeline.edit_local`: shared source preparation with global editing,
  matching source/target masked branches, independent operation flags and raw/final
  outputs. Existing global/refined-reconstruction behavior remains tested.
- CLI mask/feather/independent-disable controls, exact float mask artifacts,
  background-region metrics, authored example mask and English documentation.

No dependency, checkpoint, UI or training change was introduced.

## Implementation decisions

Mask 1 edits and 0 preserves. PIL masks are grayscale L/1 and EXIF-oriented;
tensors are HW or B1HW floating [0,1] or boolean. A single mask explicitly
broadcasts over the image batch. Input pixel dimensions must match the oriented
image; only latent resizing changes spatial size. Ambiguous RGB/integer masks,
misaligned dimensions, invalid batches/ranges and nonfinite data fail.

Binary uses threshold >=0.5 and nearest latent resizing; soft or feathered masks
use bilinear. Feather radius is Gaussian sigma in pixels, with separable support
ceil(3*sigma) per axis and replicate padding. Per-image constant masks remain
exactly constant. Tiny masks lost at latent resolution produce an explicit warning.

After each reverse update t -> s, the hook reads the original source inversion
state at **s**, including clean ID -1. Required states are validated before model
execution. Source snapshots stay on CPU, with only the matching state transferred;
the mask moves once to the compute device. Full masks behave as global editing;
zero masks finish at the exact original clean latent and decode to the VAE baseline.

Final pixel compositing is separate, controlled by `preserve_background`. Zero
effective pixel weights preserve original pixels exactly. With compositing enabled,
an empty mask returns the original; disabling both operations gives global output.
For soft feathering, the effective edit support extends outside the requested hard
region, so original-hard-mask and effective-mask preservation are reported separately.

The global/local pipelines share one prepared source path instead of duplicating
inversion/conditioning. Both local source-control and target branches receive the
same after-step mask operation. Raw source/target decoded outputs are retained
separately from final composited outputs. Reconstruction's fidelity selection and
its independent sampler remain intact; residual source refinement is still opt-in
for editing, with the previously documented high-guidance limitations.

## Executed checks

| Check | Observed result |
| --- | --- |
| Default offline suite | 189 passed, 20 skipped (3 GPU, 17 integration/reference) |
| `pytest -q --run-gpu --run-integration` | 209 passed, no failures/skips; five existing tokenizer FutureWarnings |
| Real cached local mask integration | Zero/full/partial soft, feathering, exact zero-support pixels and repeated run checks passed |
| Real 400×600 hard and feathered local edits | Passed; all output dimensions preserved and grids visually inspected |
| Global vs both-disabled control | Exported target PNGs exactly equal |
| Saved hard-mask trajectory verification | All background source states match at every explicit ID, including clean |
| Exported PNG zero-support preservation | Exact equality: 171,241 hard-mask pixels; 156,457 feathered-mask pixels |
| `pip check`, compileall, CLI help, language scan, `git diff --check` | Passed |

Unit coverage includes 0/1/partial/soft masks, feathering and constant masks,
shape/range/batch failures, independent flags, callback output validation,
missing clean state and deliberately distinct timestep source tensors. Mask
errors are tested before diffusion execution. Strength zero remains a no-U-Net
path. The full suite includes the prior reconstruction drift/fallback regressions.

## Actual local edit

Input: `assets/examples/coffee.png` (CC0, 400×600 RGB). Mask:
`assets/examples/coffee_mask.png`, a coarse authored ellipse with bounding box
(155,25,445,325), white inside and black outside. It is not segmentation ground
truth. Provenance is recorded in `assets/examples/README.md`.

Parameters: pinned SD1.5 revision, seed 42, 50 inversion steps, source guidance 1,
strength 1 (start 981, 50 reverse updates per branch), reverse guidance 7.5,
empty negative prompt, plain source inversion. Source prompt:
`a photo of a white ceramic cup filled with coffee on a wooden table`.
Target prompt: `a black and white pencil drawing of a cup of coffee, detailed sketch, pencil shading on white paper`.

| Setting | Weighted background MAE | Weighted background PSNR | M==0 pixel MAE |
| --- | --- | --- | --- |
| Global (both operations disabled) | 0.256230 | 9.554 dB | 0.256230 |
| Hard latent blending, raw decoded | 0.042941 | 24.275 dB | 0.042941 |
| Hard blending + pixel compositing | 0 | +infinity | 0 |
| Feather sigma 4, raw decoded | 0.035245 | 25.358 dB | 0.035542 |
| Feather sigma 4 + pixel compositing | 0.000500 | 44.433 dB | 0 |

Metrics use floating RGB [0,1]. Weights are 1-M and the MAE/MSE denominator is
3*sum(weights). Empty regions return undefined values, while defined exact-match
PSNR is infinite (null in strict JSON, disambiguated by positive region weight).
Feathered weighted scores include boundary mixtures and are not directly the same
region as the hard-mask score. Feathered final MAE outside the original hard mask
is 0.000292 because effective support expands; outside literal effective M==0 it
is exactly zero. No zero-masked perceptual metrics are reported.

Artifacts under ignored `outputs/editing/`:

- `phase5-hard/`: images, `comparison.png`, `preservation_comparison.png`, raw/final
  outputs, effective mask preview, exact `mask_tensors.pt`, CSV/JSON and full
  inversion/source/target `trajectories.pt` keyed by explicit IDs.
- `phase5-feather/`: the same image/mask/metric artifacts with sigma 4, without
  optional trajectory recording.
- `phase5-global-control/`: no-mask global baseline.
- `phase5-global-mask-evaluation/`: the same global algorithm with both local
  flags disabled, supplying the mask only for background evaluation. Its edited
  PNG exactly matches the no-mask control.

Reports retain source/mask hashes, exact settings, prompts, seed, model revision,
selected alpha/noise level, state IDs, environment, runtime and memory. Raw images
are measured before compositing, so pixel preservation is not credited to latent
blending. Comparison grids show localized pencil-like appearance, along with
geometry/seam artifacts from the coarse mask and strong edit.

## Limits and next phase

This single-image check demonstrates background preservation under the recorded
settings; it does not establish general semantic improvement or complete Phase 7.
The final exact-zero background is an algebraic pixel-compositing result, while
latent-only outputs retain VAE/spatial-coupling error. Feathering expands support,
thin masks can disappear at latent resolution, and coarse masks can cut through
objects. Existing color-edit reliability and high-guidance fidelity limitations
remain. None is hidden by reporting only composited images.

Phase 5 implementation and validation were committed and pushed as `3bb8253`
(local editing implementation) and `1329640` (tests, example mask and evidence)
before the user-authorized Phase 6 work began. See `phase6-status.md`.
