# Advanced reconstruction extension: high-CFG audit and null-text calibration

Date: 2026-11-17 (America/Los_Angeles client date).

Status: **measured reconstruction improvement; strict engineering gate not fully
passed; attention editing not started.** These are local, explicitly opted-in
experiments on two exposed development photographs, not a semantic acceptance
result or a replacement of the installed editing/reconstruction defaults.

## Why this extension was started

Earlier diagnostics showed that the custom sampler responds to green/yellow
text from random noise, while source-derived editing states retain original color
or alter geometry when made more editable. Parameter changes, disabling latent
blending and source-reference extrapolation did not solve that trade-off.
Color postprocessing can recolor material but adds explicit color supervision
and does not improve diffusion's own semantic control.

Before adding attention control, this extension checks a prerequisite: whether
the source image can be reconstructed under the CFG used for editing, without
masks, compositing or changed text concealing defects. The original custom
DDIM equations, pretrained checkpoint and previously recorded runtime source
remain the controls.

## Gate 1: guided source reconstruction

Use the red-apple and blue-car development inputs, original source prompts,
50-step schedule, selected timestep 741 and 38 reverse updates. The source prompt
is also the reverse prompt. Inversion guidance rows 1/7.5/12 are crossed with
reverse guidance columns 1/7.5/12. Columns within a row share the exact start;
different rows have different inverted states at the same timestep.

An additional control executes the existing residual correction at guidance12
for inversion/reverse, maximum 12 destination evaluations per inversion level,
tolerance 1e-6. It is saved independently, even when worse; no automatic fallback
hides its output. Neither this control nor any plain grid profile passes the
high-CFG numeric screen on both inputs. Matching high guidance alone is insufficient.

The thresholds were declared before output: on both inputs at reverse guidance12,
PSNR against VAE-only reconstruction >=30 dB, SSIM against VAE >=.95 and terminal
latent MSE against the clean VAE latent <=.001, followed by geometry inspection.
These are local engineering thresholds, not universal quality standards. A good
source reconstruction would not itself establish editability or semantic success.

All comparisons are unmasked and uncomposited. Original-image and VAE-relative
metrics are reported separately. The prior raw inversion1/reverse12 source PNGs
replay exactly, linking the new audit to earlier editing evidence.

## Gate 2: bounded null-text prototype

The measured failure justified a separate per-image calibration inspired by
[Null-text Inversion](https://null-text-inversion.github.io/). The method optimizes
unconditional text embeddings against DDIM source pivots while holding model
weights and conditional embeddings fixed. This implementation is an independently
written partial-trajectory adaptation, not an exact reproduction of the paper.

At each current timestep t, optimize only the FP32 null embedding u_t:

```text
epsilon = epsilon_unconditional(z_t, t, u_t)
          + 12 * (epsilon_source(z_t, t, source_text)
                  - epsilon_unconditional(z_t, t, u_t))
z_s = our_eta0_ddim_step(z_t, epsilon, alpha_t, alpha_s)
loss = mean((z_s - source_pivot[s])**2)
```

The current state is the actual propagated model output. The source pivot is
only a loss target; it is never copied into the sample. After optimizing a level,
retain the best **evaluated** null embedding and replay its actual update.
There is no final unevaluated Adam proposal and no pixel substitution.

Fixed settings: 20 evaluations per level, Adam learning rate .01 multiplied by
`1-local_step_index/100`, local MSE target 1e-5, loss scale 1024 for FP16 backward,
warm-start from the previous best null and a fresh optimizer per timestep.
All model parameters stay frozen with no gradients. Only the input null embedding
is optimized; the model checkpoint and source text are not trained or changed.

The installed U-Net predictor is inference-only. An output-local differentiable
adapter permits gradients to the null embedding without modifying that predictor.
CUDA FP16 U-Net/text and FP32 VAE/update arithmetic remain unchanged. No new
weights, editing pipelines, training dataset or attention manipulation is used.

## Execution-layout correction

The initial prototype calibrated the conditional and unconditional predictions
as separate batch1 forwards. It freely replayed its own saved nulls exactly in
that mode, but replaying them with batched CFG degraded the car: VAE-relative
PSNR fell from 31.26 to 26.07 dB. The original result remains in its own directory.
It would have been incorrect to silently deploy those embeddings with the
production batch layout and retain the calibration metrics as the deployed result.

A separately registered trial changed **only branch batching**. Calibration now
evaluates `[unconditional, conditional]` together, matching the primary free replay.
All optimizer settings, pivots, prompts, checkpoint, precision and gate thresholds
stay fixed. The new primary batched replay is bitwise identical to the calibration
final latent for both inputs. Separate replay is still saved as a sensitivity
test and is not required to match a differently calibrated FP16 execution map.
Embeddings therefore need compatible runtime/branch-layout metadata and validation.

## Final measured results

Numbers below compare the image to VAE-only reconstruction of the same input.
They do not compare to an independently generated target image.

| Input | Ordinary empty-null CFG12 PSNR / SSIM | Batch-matched calibrated free replay PSNR / SSIM | Terminal latent MSE, ordinary -> calibrated |
| --- | --- | --- | --- |
| Apple | 19.525 dB / .7140 | 36.952 dB / .9674 | .122115 -> .002158 |
| Car | 15.895 dB / .4002 | 31.315 dB / .9524 | .186194 -> .001283 |

The source structure is visually substantially closer to the VAE baseline than
with ordinary CFG12. VAE codec loss remains visible and must not be confused with
diffusion-only reconstruction fidelity. Original-image metrics remain in reports.

Both final images meet the declared VAE-relative PSNR/SSIM thresholds, but both
latent MSEs exceed .001. **Zero of 38 local levels per input reached 1e-5 within
the fixed 20-evaluation budget.** The full gate is not passed and the thresholds
are not relaxed post hoc. Improved pixel similarity is not evidence of exact
per-step inversion, convergence, green/yellow editing or generalization.

The next gate is convergence/accuracy under a separately registered optimization
budget or numerical-precision study. Attention editing stays separate until source
reconstruction requirements are resolved. This is a narrower, measured blocker
than an undiagnosed inability to respond to color words.

## Verification, resources and local artifacts

- Frozen-runtime core tests: 49 passed / 1 skipped. Initial null-text prototype:
  8 offline tests passed. Batch-matched prototype: 9 passed, including a new test
  that every calibration forward uses the deployed doubled-batch layout; the 9
  include the earlier 8 tests and must not be counted as 17 distinct tests.
- Independent free replay reloads the saved initial latent, conditional embedding
  and per-timestep null map. It has no source-pivot or input-pixel argument.
  Matching-layout final latents replay exactly. All model parameters remain frozen
  with `grad is None`. Inputs, code, pivot and output hashes are checked.
- Audit: 2,360 U-Net calls / 4,392 batch elements, 130.799 summed case seconds.
- Initial separate calibration plus controls/replays: 2,128 calls / 2,280 batch
  elements and 1,444 backward evaluations, 166.203 summed seconds.
- Batch-matched calibration plus controls/replays: 2,052 calls / 3,800 batch
  elements and 1,444 backward evaluations, 269.747 summed seconds. Peak allocated
  CUDA memory about 4.77 GiB on apple and 4.30 GiB on car.
- Whole extension so far: 6,540 new forwards / 10,472 batch elements and 2,888 backward
  evaluations. The two inherited source-pivot inversions are already included in
  the audit count; do not add them again. Forward counts alone do not represent
  backward cost. Timings include case artifact work and exclude model loading;
  these are single-run measurements, not a throughput benchmark.
- No old rating table, mask, acceptance prompt, model weights, default pipeline,
  running demo service or Git state was changed by the inference experiments.
  No holdout output, new human rating or method freeze was produced.

Local output root (not guaranteed present in a clean clone):
`outputs/acceptance/2026-11-16-side-validation/`.

- `development-round-07/`: registered reconstruction matrix, independent residual
  control, actual state IDs, floating metrics, source trajectories and exact-prompt
  comparison grids.
- `development-round-08-null-text/`: initial separate-branch calibration, all
  optimizer losses, saved null embeddings and FP16 batching-sensitivity evidence.
- `development-round-08-null-text-batched/`: independently registered layout
  correction, exact primary replay, per-level CSV/JSON, all unsuccessful levels,
  original/VAE/plain/calibrated grids and error plots. Start with `REVIEW.md`.

Experimental numerical code remains output-local while the strict gate is open.
Adoption into the installed `diffusion/` and `pipelines/` layers is a separate
reviewable change; there is no installed null-text inference mode yet.

## Follow-up: actual target-text editability

An explicitly requested diagnostic then held the saved initial state, guidance12,
precision, batching and 38 reverse updates fixed, comparing empty-null and
calibrated-null source reconstruction with minimal one-word color edits. All four
live source controls replayed their prior PNGs exactly. No new optimization,
mask blending, pixel compositing, RGB color correction or attention operation was
used. Masks served only as fixed evaluation regions/source detail coordinates.

The calibrated apple remained predominantly red/orange with changed texture/shape;
the calibrated car retained a blue/teal body with localized yellow and substantial
geometry/background changes. Improved source reconstruction did not translate
into reliable recoloring under this saved-null/text-swap configuration. It does
not prove that a converged full method or attention-controlled editing cannot work.
The source convergence gate stays open, no attention extension or six-case
expansion was started, and all human target ratings remain unreviewed.

The new diagnostic used 304 forwards / 608 batch elements with no backward passes,
inversions or recalibration. Source/target grids, exact prompts, raw outputs,
metrics and unsuccessful examples are retained locally in
`development-round-09-editability/` under the same isolated output root. Start
with its `FINDINGS.md` and `REVIEW.md`.

## Separate pretrained instruction reference

The separately authorized two-image InstructPix2Pix reference uses a different
pretrained editor and built-in Euler ancestral sampling. Six offline contract
tests pass, and both real-weight runs completed after explicit download
authorization. Both visible dominant colors changed as requested, with unrelated
background/material changes still present and human scores unassigned. This baseline neither
changes the source convergence gate nor introduces attention editing into the
custom pipeline. See [the reference setup](instruction-reference.md) for pinned
identity, literal instructions, compute/precision differences and execution status.

## Current research decision

The user has selected custom DDIM editing as the active direction again. Follow
[the consolidated evidence review and plan](ddim-research-plan.md): first audit
calibration gradients, effective updates, bounded evaluation budget and precision
at fixed levels, then attempt one justified reconstruction candidate. Attention
control remains conditional on the unchanged source gate. Instruction-editing
references and their localization derivatives are preserved, not the next
development branch. No future diagnostic is executed by this decision.

The subsequent authorized [optimization/precision diagnostic](calibration-diagnostic.md)
has now executed. Deterministic local repeats pass, while the retained initial
attempt establishes non-repeatable optimization in its original GPU runtime.
Cap80 improves early local convergence; same-budget FP32 execution does not solve
the selected problems. Source replays remain exact, but no new full-trajectory
source or target edit is produced. The source gate is still open. Next is a
separately registered deterministic full-trajectory budget comparison, without
splicing fixed-state optima or copying pivots into sampled states.
