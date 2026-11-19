# Custom DDIM editing: evidence review and research plan

Decision date: November 18, 2026 (America/Los_Angeles).

## Active direction

The user has selected the independently implemented DDIM editor as the active
research route again. The installed editor already uses that route; this decision
changes experiment priority, not the inference backend. Keep custom inversion,
eta=0 reverse updates, CFG, timestep-indexed trajectories and mask handling, with
the existing pretrained SD 1.5 components as the backbone.

InstructPix2Pix, direct RGB color constraints and their composited derivatives
remain labelled reference evidence. Their development is no longer the next
deliverable. Preserve outputs, failures, provenance and ratings; do not present
them as fixes to custom DDIM or silently migrate them into the editor. Final
pixel compositing remains an independently configured preservation operation in
the custom editor; it cannot establish semantic success.

This document summarizes evidence and proposes stages. It does not execute a new
study, bypass a gate or promise successful recoloring. Numerical prototypes still
live outside the installed package.

## What has been implemented and tried

Local paths below are relative to
`outputs/acceptance/2026-11-16-side-validation/`. This ignored directory is not
guaranteed to exist in a clean checkout. Retain failed attempts and distinguish
actual user ratings from visual observations.

| Method or diagnostic | Executed evidence | Result and implication |
|---|---|---|
| Core custom DDIM inversion/sampling/CFG and global/local editing | Phase 0–7 status records, equation/shape/device tests, real-weight runs, UI and ablations | Core features have recorded phase evidence. Approximate inversion and high-guidance natural-image drift remain limits; arithmetic tests do not establish edit quality. |
| Budgeted residual inversion refinement | `docs/reconstruction-drift.md`; high-CFG control in `development-round-07/` | Helps the verified low-guidance reconstruction setting, with documented adverse cases and fallback. The independent guidance-12 control fails the later gate; this is not a general high-CFG solution. |
| Higher strength or guidance | `development-strength-1/`, `development-guidance-12/`; human ratings in round02 | Each six-case profile has three passes, two partials and one failure. In both, apple scores 1/2/2 (mixed red/green), car 0/2/2 (still blue). Some replacement/style tasks succeed; recoloring remains unresolved. |
| Disable per-step source-latent blending | `development-round-03/` | Recoloring candidates retain source colors; actual human scores remain apple 1/2/2, car 0/2/2. Requested-background MAE rises in all six pairs. Three other candidate rows remain unreviewed. Disabling blending alone is insufficient. |
| Literal prompt/token/conditioning checks and same-noise generation | `development-round-04/` | No truncation; source/target embeddings and noise predictions differ. Text generation produces green apples/yellow cars. This argues against entirely missing target conditioning, without certifying the whole implementation. |
| Source-derived starts 741 and 981 without spatial blending/compositing | `development-round-04/` | Start741 retains source color. Start981 can yield a yellow car while changing its scene/geometry; the apple remains mixed. Editing strength and preservation compete. |
| Source-reference epsilon extrapolation | `development-round-05/` | `eps_source + 12*(eps_target-eps_source)` at the same current state fails the pilot's dominant-color screen and changes texture/geometry. No six-case expansion. This is not Delta Denoising Score optimization. |
| Explicit color-only and diffusion-plus-color postprocessing | `development-round-06/` | All four actual user verdicts are partial, with 1/2/2 scores: overly/neon-green apple and still-blue car mask edges. Extra RGB supervision changes color but does not improve DDIM semantic guidance. |
| Same-prompt inversion/reverse guidance matrix and residual control | `development-round-07/` | No tested high-guidance profile meets the reconstruction screen on both inputs. Matching inversion/reverse CFG alone is insufficient. |
| Per-timestep null-text calibration with separate CFG branches | `development-round-08-null-text/` | Source reconstruction improves and matching-layout replay is exact. Doubled-branch replay degrades the car, revealing FP16 execution-layout dependence. Keep failed-layout evidence and cost. |
| Batch-matched null-text calibration and pivot-free replay | `development-round-08-null-text-batched/` | Substantially better source reconstruction; strict latent/convergence gate still unmet. Exact replay establishes reproducibility, not equality with the original clean latent or editability. |
| Actual target-text swap with calibrated nulls | `development-round-09-editability/` | All live source controls replay exactly. Target apple remains red/orange; car blue/teal with localized yellow and scene/geometry changes. Better reconstruction does not deliver reliable recoloring here. Four target rows remain unreviewed. |
| Fixed-state budget/precision and repeatability diagnostic | `development-round-12-calibration-diagnostic/` and `-v2/` | The old GPU optimization is not bitwise repeatable, though source replay is exact. Separately registered deterministic repeats match on6/6 problems. FP16 cap80 reaches local tolerance on2/6; FP16/FP32 cap20 reach0/6. This is not a full-trajectory or semantic pass. |
| Different pretrained instruction editor | `development-round-10-instruct-pix2pix-reference/` | One fixed InstructPix2Pix run makes the apple green and car body yellow, with unrelated changes. Weights, conditioning, scheduler, precision and work differ; this is not model-only causal isolation or a custom DDIM fix. Two rows remain unreviewed. |
| Hard/inward-feather compositing of fixed reference outputs | `development-round-11-localization/` | Composites restore requested-mask exterior exactly in float and PNG, giving requested-background MAE zero. Visual inspection shows red apple rims and blue car strips; inward feathering does not remove them. Four rows remain unreviewed. |

The initial six-case pilot received one pass, two partials and three failures.
Formal verdicts above refer to saved tables, not newly assigned scores. Pixel
metrics cannot overrule a failed human dimension. No held-out output or method
freeze has occurred.

## Best current custom reconstruction result

The batch-matched calibration results compare images against VAE-only
reconstruction and final latents against the encoded clean latent. These are not
editing scores.

| Input | Plain source PSNR / SSIM vs VAE | Calibrated PSNR / SSIM vs VAE | Calibrated clean-latent MSE | Local levels reaching 1e-5 within 20 evaluations |
|---|---|---|---:|---:|
| Apple | 19.5251 dB / 0.714021 | 36.9525 dB / 0.967389 | 0.00215806 | 0 / 38 |
| Car | 15.8946 dB / 0.400203 | 31.3147 dB / 0.952438 | 0.00128305 | 0 / 38 |

The existing screen requires PSNR vs VAE >=30 dB, SSIM >=0.95 and clean-latent
MSE <=0.001 on both inputs, plus declared local-convergence requirements and
geometry inspection. Do not relax thresholds after seeing results. The first
two numerical criteria pass; latent/local-convergence requirements do not.

Independent free replay compares two clean reverse endpoints: one produced during
calibration and another from reloaded start/null inputs. Matching endpoints are
expected for the same deterministic execution map. They do not prove that either
equals the original encoded clean latent. DDIM inversion re-evaluates the
predictor at different states/times and is approximate.

## Interpretation

The evidence supports a limitation of the tested source-state/text-swap
configuration, rather than a proven unique implementation bug or inability of
SD 1.5 to represent target colors. Reconstruction and editability are separate
requirements. Mask/contour mismatch contributes to boundary defects, but cannot
explain away source-color retention in fully editable interior pixels.

The partial-trajectory calibration is inspired by
[Null-text Inversion](https://null-text-inversion.github.io/), which optimizes
unconditional inputs while retaining model weights and conditional text. It is
not an exact reproduction, and the target-text diagnostic is not the complete
published attention-based editing setup. Its failure does not reject that full
method.

## Next stages, in order

### A. Diagnose bounded calibration before another editing sweep

The immediate deliverable is a registered optimization/precision diagnostic on
the same two development inputs. Determine why local losses stop above tolerance
before adding another editing mechanism.

1. Reproduce the tested batch-matched control; verify model/code/input/pivot hashes.
   Keep CFG12, start741, 38 reverse updates, actual timestep IDs and
   `[unconditional, conditional]` batching. Do not conceal source errors with
   masks, RGB correction or final pixel compositing.
2. Preregister the first, middle and final actual current levels. Use saved freely
   propagated calibration states as fixed local inputs; pivots remain targets
   only. Log every evaluated loss, gradient norm, null update size, prediction
   change and fraction of embedding updates surviving the FP16 cast. Check the
   gradient path and best-evaluated-state selection; finite gradients alone do not
   prove optimization is effective.
3. Compare the 20-evaluation control with one preregistered 80-evaluation cap at
   identical precision, learning-rate schedule and initialization. Retain failures;
   do not turn this into an unrestricted prompt/scale search.
4. Compare FP16 and FP32 forward/backward execution on the same selected local
   problems at the same evaluation cap. Check memory first; record inability to
   fit instead of silently changing resolution/batching/precision. Embeddings
   calibrated under a different execution map need fresh compatible calibration
   before deployment.

These checks distinguish insufficient budget, FP16 update/gradient effects and
sequential drift; none is an established root cause. Record actual forwards,
batch elements, backward work, runtime and memory, including unsuccessful trials.

### B. Attempt one justified full-trajectory reconstruction improvement

Choose a candidate only from stage A evidence and register it before outputs.
Change one mechanism at a time, keep the existing numerical/visual gate and test
both inputs. Save per-level failures and clean-endpoint errors.

Reload saved initial latents and timestep-indexed nulls for pivot-free replay at
the calibrated dtype/batching. Freeze weights and conditional text. Never copy
pivots/pixels into sampled states. Reproducibility and fidelity remain separate
results. If the bounded candidate fails, stop and review objective/gradient/pivot
evidence; do not stack attention features or silently raise the budget again.

### C. Test minimal custom attention control after the gate

After source reconstruction requirements pass, separately register one minimal
color-token control study using our DDIM sampler. The intended mechanism is
source/target token alignment and attention control to preserve unchanged spatial
structure while allowing the requested color change.
[Prompt-to-Prompt](https://prompt-to-prompt.github.io/) supplies the relevant
cross-attention-control reference, without guaranteeing success on these inputs.

First verify that a disabled/no-op controller reproduces the current baseline.
Then compare one preregistered controller with plain text swap, holding other
settings fixed. Test CFG branch isolation, subword alignment, actual timestep
scheduling, attention shapes/dtypes and hook cleanup after errors. Explain any
attention-cache/runtime architecture change before coding. Keep the eta=0 update
independently implemented; do not substitute a built-in editing pipeline.

### D. Localize and expand semantic validation

After the two cases show usable colors with retained structure, compare masks
and blending as preservation mechanisms while retaining raw outputs. Requested
evaluation masks stay fixed within each study; mask refinement is a separately
registered input change, not a retrospective score improvement. Neither higher
PSNR nor exact exterior pixels waives semantic/boundary review.

Next register the balanced six-case development evaluation, including replacement
and style. Freeze assets, prompts/masks, method, seeds/settings and review rules
after development decisions. Only then generate the six holdout outputs and
obtain independent three-dimension review. After holdout exposure, changed methods
need fresh unseen confirmation cases; retain old cases as labelled regression
material.

## Current status and stopping points

- Stage A has now completed; see [the measured diagnostic](calibration-diagnostic.md).
  Both old source replays are exact, the deterministic correction passes repeated
  controls, and increased local budget helps the early selected level. Precision
  alone does not solve the selected local problems. All failed attempts remain.
- Stage B is next: separately register deterministic full-trajectory cap20/cap80
  calibration on actual propagated states, rather than splicing local probes.
  The source thresholds remain unchanged. Stage C stays conditional on that gate;
  holdout stays conditional on freeze.
- The CPU localization study used no new U-Net/backward work and passed seven
  offline tests. Browser save/navigation verification did not complete: the
  computer-use tool could not reliably determine the current Chrome URL. Do not
  claim a browser pass or fabricate scores. Existing review data is preserved.
- The route decision itself was documentation-only; its subsequent authorized
  stage A executed the isolated GPU diagnostic. Neither changes production
  defaults, ratings, reference weights, existing services or Git history.
