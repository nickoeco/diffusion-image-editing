# Phase 2 verification record — 2026-11-15

## Deliverable and decisions

- `diffusion/schedule.py`: model-compatible beta/alpha products, actual selected
  scheduler timestep IDs, ascending/reverse grids and explicit clean boundary.
- `diffusion/inversion.py`: readable deterministic DDIM inversion equations,
  current-state noise prediction, structured trajectory and validation.
- `models/noise_predictor.py`: conditional-only pretrained U-Net execution adapter,
  FP16 model inference with FP32 noise/update arithmetic.
- `scripts/test_inversion.py`: cached-weight real-image-latent inversion smoke and
  saved trajectory/report, without reverse sampling or reconstruction claims.
- Numerical, boundary, irregular-grid, determinism, invalid-input and reference
  tests; updated English documentation. No dependencies were added.

The clean VAE latent uses ID -1 and alpha 1. The checkpoint's scheduler originally
sets `set_alpha_to_one=false`; a cloned DDIM view explicitly sets it true for the
clean boundary without changing training coefficients or the original scheduler.
This contract must be preserved by Phase 3 reverse sampling. The first U-Net query
uses the lowest selected training ID; later queries use the current state's ID.

Conditional-only guidance scale 1 is supported in this phase. Non-1 scales fail
explicitly until Phase 3 adds CFG arithmetic. Trajectory snapshots default to CPU
FP32 to avoid retaining the full history on the GPU. Results contain the clean and
terminal states, all intermediate states, explicit timestep/query IDs and schedule.

## Executed checks

| Check | Observed result |
| --- | --- |
| Default offline suite | 89 passed, 12 skipped (3 GPU and 9 explicit integration/reference checks) |
| `pytest -q --run-gpu --run-integration` | 101 passed, no failures or skips; one existing upstream tokenizer FutureWarning |
| Real cached SD1.5 50-step inversion | Passed; all states finite and correctly shaped |
| `pip check` | Passed |
| `compileall -q src scripts tests` | Passed |
| Inversion script `--help` | Passed |

Independent NumPy float64 equations verify individual updates. Constant-epsilon
synthetic trajectories verify every intermediate level analytically. Optional
Diffusers comparisons check all aligned fixed-epsilon steps at 1/20/50/100 steps,
while grid tests cover leading, linspace and trailing spacing. A checkpoint offset
that creates an out-of-range ID is explicitly rejected. Reference comparisons do
not claim that differently queried full neural pipelines are equivalent.

The first full-suite collection exposed a unit/integration module basename
collision. Renaming the reference module to `test_ddim_reference.py` resolved it;
the final recorded full suite ran successfully. No failed check was hidden.

## Actual inversion output

Executed `scripts/test_inversion.py --steps 50` with the pinned pretrained model
and Phase 1's generated 512×512 RGB test card, using source prompt
`a red apple on a colorful background` and guidance 1. No new weight download
was required. This input is synthetic and is not a natural-image quality benchmark.

Saved under `outputs/inversion/phase2/` (excluded from version control):

- `trajectory.pt`: 51 CPU FP32 states, each [1,4,64,64], keyed by integer ID.
- `report.json`: model/config/environment, source image/prompt, state/query IDs,
  alpha coefficients, dtype/storage, finiteness, runtime and peak allocated memory.

The state sequence was `[-1,1,21,...,981]`, with prediction IDs
`[1,1,21,...,961]`. The measured inversion-only runtime was about 1.69 seconds,
including CPU snapshot transfers and excluding model loading/encoding/file saving.
Peak allocated CUDA memory reported by PyTorch was 2,422,075,904 bytes (about
2.26 GiB), including resident models; this is not total system GPU memory use.
These are one-run measurements, not performance ablations or general guarantees.

## Limitations and next phase

DDIM inversion is approximate and the terminal latent is not assumed pure noise.
No reverse reconstruction, PSNR/SSIM, global/local editing, CFG arithmetic or demo
has been implemented or evaluated. Phase 3 will add explicit reverse sampling and
CFG, then compare VAE-only and inversion+reverse reconstruction using the exact
same grid and clean boundary. Phase 2 unit-test gates have passed.
