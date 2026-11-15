# Phase 4 verification record — 2026-11-15

## Prior phase commits

Phase 3 was reviewed, tested (139 passed) and pushed to `origin/main` as:

- `d9518a0`: manual CFG, reverse DDIM, reconstruction/metrics and tests.
- `8792daa`: cached-model smoke script, example provenance and measured reports.

## Phase 4 deliverable

- `editing/strength.py`: validated discrete strength-to-stored-level selection.
- `DDIMSampler.sample(start_timestep=...)`: reverse only the matching suffix,
  retaining exact timestep IDs and the existing clean boundary.
- `pipelines/editing_pipeline.py`: independent source/target conditioning, source
  inversion, same-start source control and target edit, typed outputs and metadata.
- `pipelines/conditioning.py`: shared B/2B conditioning and image/prompt batch
  validation, also used by reconstruction without changing its mathematical path.
- `scripts/run_edit.py`: offline cached-model CLI with labelled comparisons,
  source-control outputs, preservation metrics and optional saved trajectories.
- Strength, partial traversal, prompt routing, analytic edit, endpoint and cached
  real-model checks; English documentation and a CC0 coffee example.

No dependencies were added. Mask editing, feathering and pixel compositing remain
Phase 5 work. No high-level Diffusers editing pipeline is used.

## Strength contract and review

For N scheduler-selected levels, choose k=ceil(strength*N), then look up
`inversion.timesteps[k]`. The actual selected alpha/noise level is recorded.
Decimal text multiplication prevents binary-float bin-boundary off-by-one errors
(for example 0.14 at 50 steps selects 7, not 8). At 50 steps, strength 0.5 starts
at 481 with 25 reverse updates; 0.75 starts at 741 with 38; 1 starts at 981 with 50.

Strength 0 skips inversion/U-Net and returns VAE-only outputs, with
`inversion=None` and clean-only source/target sampling results. It is not an exact
pixel copy of the original. Positive strength executes the full source inversion,
then independently samples source and target from the same saved latent at the
selected ID, using the same reverse guidance. No arbitrary extra noise is added.
Timesteps, input dimensions, batch order, dtype/device and output finiteness were
reviewed; no coefficient/grid changes or mask behavior were introduced.

## Executed checks

| Check | Observed result |
| --- | --- |
| Default offline suite | 150 passed, 18 skipped (3 GPU, 15 integration/reference checks) |
| `pytest -q --run-gpu --run-integration` | 168 passed, no failures/skips; three existing upstream tokenizer FutureWarnings |
| Five real cached global editing smoke runs | Passed execution and output-shape checks; semantic outcomes below |
| `pip check`, compileall, `run_edit.py --help` | Passed |
| Language scan and `git diff --check` | Passed |

Unit checks use nonuniform grids, analytic constant noise and distinguishable
source/target conditioning. They verify shared start tensors, actual partial
query IDs/counts, zero strength without U-Net calls, exact same-prompt controls,
invalid inputs and repeatability. The real-model integration checks zero,
partial, repeated, and full-strength paths. Full strength with identical prompts
matches the reconstruction pipeline; partial source/target outputs differ.

## Actual image runs

All runs use the pinned SD1.5 revision, 50 inversion steps at guidance 1, seed 42,
empty negative prompts, FP32 VAE/updates and CUDA FP16 U-Net/CLIP.
Each run saves original, VAE-only, source-control and edited PNGs, `comparison.png`,
`report.json` and `metrics.csv`. The first run also saves explicit-ID trajectories.
Inputs have provenance recorded in `assets/examples/README.md`.

| Output directory under outputs/editing/ | Input / target | Strength / reverse guidance | Observed outcome |
| --- | --- | --- | --- |
| phase4 | Astronaut / blue spacesuit | 0.5 / 5 | Limited appearance change; requested blue recoloring not achieved |
| phase4-strong | Astronaut / blue spacesuit | 0.75 / 7.5 | Larger geometry/appearance drift; blue suit still not achieved |
| phase4-monochrome | Astronaut / monochrome portrait | 0.75 / 7.5 | Retained color and changed appearance; monochrome request not achieved |
| phase4-coffee | Coffee / blue cup | 0.75 / 7.5 | Cup geometry changed but requested blue color not reliably achieved |
| phase4-sketch | Coffee / black-and-white pencil drawing | 1 / 7.5 | Clear pencil-style target output, with photographic source control; geometry/detail drift remains |

The coffee image is 400×600 RGB and all corresponding outputs retain those
dimensions. All comparison grids were visually inspected. The first 0.5 run
selected alpha 0.3022954464/noise std 0.8352871085; the 0.75 runs selected alpha
0.0603555217/noise std 0.9693526078. These are actual stored levels, not interpolated
timestep coefficients. The full-depth sketch starts at 981 and executes 50 reverse
updates per branch. It retains recognizable cup/coffee structure but not exact
original geometry, texture or surrounding pixels.

Reports include both prompts, source hash, seed/config/model revision, requested
strength, selected ID/alpha/noise std, actual counts and traversals, environment,
runtime and peak allocated memory. Runtime includes VAE/text work, full inversion
and both reverse branches, excluding model loading, saving and metric calculation.
The measurements are exploratory smoke runs, not controlled performance benchmarks.

Preservation metrics are recorded in `experiments.md`. They do not measure prompt
alignment. For the sketch, edited-vs-original PSNR is 9.921 dB and SSIM 0.3187;
that low pixel similarity is compatible with a visible style change and is not
evidence of either general semantic success or algorithmic improvement.

## Status and limits

The global editing implementation, strength controls, reproducibility checks and
one clear global appearance modification are verified. Color editing remains
unreliable on the tested examples; stronger settings can amplify source drift.
Failed semantic cases remain documented and are not described as successful edits.
No exact preservation, quantitative prompt alignment, general quality improvement,
completed formal ablation or local edit is claimed.

Phase 4 is ready for review with these quality limitations. Phase 5 has not started.
