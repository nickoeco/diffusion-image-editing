# Phase 3 verification record — 2026-11-15

## Deliverable

- `diffusion/guidance.py`: explicit FP32 CFG and one-call paired U-Net batching.
- `diffusion/step.py`: shared explicit eta=0 update with direction-specific checks;
  inversion retains its public wrapper and now supports the shared CFG path.
- `diffusion/sampler.py`: reverse traversal of the matching grid, explicit clean
  endpoint, FP32 final latent and optional CPU trajectory.
- `pipelines/reconstruction.py`: VAE-only and source-prompt inversion/reverse
  reconstruction with separate inversion/sampling guidance settings.
- `evaluation/metrics.py`: precisely defined RGB PSNR and SSIM.
- `utils/visualization.py` and `scripts/test_reconstruction.py`: labelled grids,
  optional decoded-latent previews, saved trajectories, strict JSON and CSV reports.
- Synthetic tests, fixed-epsilon Diffusers comparisons, cached pretrained CFG
  reconstruction and repeatability checks. All project documentation is English.

No image editing, strength selection, masks or UI has been implemented in this phase.

## Mathematical and API decisions

The reverse traversal uses the same `DDIMSchedule` object as inversion. At 50 steps
the states are `[981,961,...,21,1,-1]`. Each noisy state is predicted at its current
training ID, and the last update reaches clean alpha=1; the U-Net never receives -1.
The shared step implements the two explicit DDIM equations, in FP32, without
clipping, stochastic noise or calls to a library sampling step.

CFG is `u + w*(c-u)`. Paired embeddings must be [all unconditional, all conditional]
and produce one doubled-batch U-Net call per step. Scales 0 and 1 select exact
endpoint predictions. Conditional-only B batches remain supported at scale 1,
preserving the Phase 2 path. The reconstruction pipeline uses conditional batches
at guidance 1, paired batches otherwise. Higher guidance can change image fidelity
and batch-shape rounding; both guidance values are always recorded.

Metrics operate before uint8 saving on matching single-image RGB tensors in [0,1].
PSNR uses whole-image RGB MSE in double precision and data range 1. SSIM uses
scikit-image 0.25.2, uniform 7×7 windows, channel averaging and sample covariance.
Identical images have infinite PSNR; reports encode that as null with a documented
meaning. Invalid data and nonfinite SSIM fail rather than silently producing scores.

## Dependencies

Added the fixed optional dependency `scikit-image==0.25.2` for SSIM, in the
`evaluation` extra and CPU `test` extra. The full requirements include evaluation.
Its installed dependencies and the existing CUDA stack are recorded in the updated
Windows/Python 3.11/CUDA 12.4 constraints snapshot. No model change or new model
download was needed. The original Phase 0 `dev` extra alone does not run the current
tensor/metric suite; use `test` for offline tests or the full requirements.

## Executed checks

| Check | Observed result |
| --- | --- |
| Default offline suite | 122 passed, 17 skipped (3 GPU and 14 explicit integration/reference checks) |
| `pytest -q --run-gpu --run-integration` | 139 passed, no failures or skips; two upstream tokenizer FutureWarnings |
| Pretrained synthetic 512×512 reconstruction, guidance 1/1 | Passed; images, metrics and intermediates saved |
| Pretrained natural 512×512 reconstruction, guidance 1/1 | Passed; images, metrics and intermediates saved |
| Pretrained natural reconstruction, guidance 3/3 | Passed; real paired CFG execution and metrics saved |
| `pip check`, compileall and reconstruction script `--help` | Passed |
| Documentation language scan and `git diff --check` | Passed |

Synthetic tests cover CFG 0/1/>1, pairing/order/single-call behavior, shape/range
errors, independent float64 reverse equations, irregular schedules and constant-noise
roundtrips including clean. Optional Diffusers tests compare fixed-epsilon reverse
steps at 1/20/50/100 steps. A real cached-model integration check runs source-prompt
CFG reconstruction twice at four steps and verifies shape, alignment, finiteness
and repeatability. It does not assert a visually faithful reconstruction.

## Measured outputs

The natural input is a 512×512 public-domain NASA example exported from
`skimage.data.astronaut()`, with provenance recorded in `assets/examples/README.md`.
All three smoke runs use 50 steps, seed 42, the pinned SD1.5 revision, empty negative
prompt, CUDA FP16 U-Net/CLIP and FP32 VAE/latent arithmetic.

| Input and guidance (inversion/sampling) | VAE PSNR / SSIM | DDIM PSNR / SSIM |
| --- | --- | --- |
| Generated test card, 1/1 | 29.066 dB / 0.8367 | 10.990 dB / 0.6174 |
| Natural example, 1/1 | 26.799 dB / 0.8706 | 18.711 dB / 0.6283 |
| Natural example, 3/3 | 26.799 dB / 0.8706 | 12.888 dB / 0.4405 |

Outputs are saved in ignored directories `outputs/reconstruction/phase3/`,
`phase3-astronaut/` and `phase3-cfg3/`. Each contains original/VAE/DDIM PNGs,
`comparison.png`, `report.json` and `metrics.csv`. The first two runs also contain
`latent_previews.png` and `trajectories.pt`. Noisy previews are explicitly labelled
as decoded latent states, not clean predictions. The comparisons and natural-image
intermediate grid were visually inspected.

Reports preserve input hashes, source/negative prompts, model ID/revision, seed,
both guidance settings, exact traversal IDs, metric settings, environment, runtime
and peak allocated CUDA memory. Measurements here are individual smoke runs and
do not complete Phase 7's ablation or baseline comparison requirements.

## Review and limits

The DDIM reconstruction has substantial visible appearance drift, including face,
pose and fine-detail changes on the natural example. Its error exceeds VAE-only
loss; guidance 3 further worsens preservation in this check. This is retained as
an observed limitation rather than hidden or described as an improvement.

Equation/reference tests establish coefficient and timestep correctness, not exact
invertibility of neural noise predictions. The source-conditioned inversion still
uses the documented current-state approximation and lowest-timestep clean proxy.
These tests do not establish general quality, exact recovery or hardware-independent
bitwise determinism. Improved inversion is outside this phase's scope.

Phase 3's implementation and execution gates are satisfied, with reconstruction
quality limits documented. Phase 4 (global target-prompt editing) has not started.
