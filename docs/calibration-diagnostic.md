# Null-text calibration: optimization and precision diagnostic

Date: November 18, 2026 (America/Los_Angeles).

Status: **local diagnostic completed; source reconstruction and semantic gates
remain open.** No full-trajectory candidate, attention editor or new target-text
output was generated. All work uses the cached original SD 1.5 components and
independently implemented DDIM/CFG. No new pretrained weights were downloaded.

## Registered question and controls

Why did no local level meet MSE 1e-5 within the previous 20-evaluation calibration
budget? The two exposed development inputs remain fixed. The saved calibrated
start/null inputs freely replay all 38 updates from current level741 to CLEAN=-1
without source-pivot access. Both final PNGs match the prior calibrated and free
replay PNGs exactly. Intermediate actual sampled states are recovered from this
replay; no source pivot replaces a state.

The first, middle and final current levels were preregistered as 741, 381 and 1,
with updated loss-target levels 721, 361 and CLEAN=-1. At each selected level the
current sampled state, prior warm-start null, source conditional embedding and
updated-level loss target remain fixed. Only the FP32 null leaf is optimized.
Source pivots are loss targets only; they are stored separately from replay inputs.

Profiles: FP16 cap20, FP16 cap80 and FP32 cap20. Adam, learning-rate schedule,
local tolerance1e-5, loss scaling1024, CFG12, batch2 branch order and custom FP32
DDIM arithmetic remain fixed. The FP32 branch **upcasts the same FP16-rounded
U-Net weights**, with identical saved conditional values. Ordered canonical FP32
parameter hashes match across branches. This is execution-precision sensitivity,
not a test of the original full-precision checkpoint. TF32 is disabled there.

## Retained failed control and deterministic correction

The initial strict cap20 check required the optimized null to match the old saved
null bitwise. It failed after both source replays passed. That failed directory,
its plan/error and measured work are retained.

A separately recorded bounded investigation ran the original helper twice with
identical fixed inputs and module residency, alongside the instrumented helper
and an additional residency control. Initial losses were identical, but the
original helper itself returned different nulls and best losses. For the original
repeated offloaded branches, best loss was 2.1765563e-5 vs2.1797468e-5. Across all
four checks it ranged from2.1709679e-5 to2.1797468e-5. All remain above1e-5.
This establishes optimization non-repeatability in that runtime; it does not
identify a specific kernel or prove it caused the reconstruction failure.

A new registration enables `torch.use_deterministic_algorithms(True)`, cuDNN
deterministic mode, disables cuDNN benchmarking and sets process-local
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. It adds a repeated cap20 control. All six
repeated loss/gradient/update histories, best nulls and candidate latents match
bitwise within this pinned execution. Every cap80 first20 loss sequence equals
its cap20 control. Deterministic settings belong to the experimental map and are
not silently adopted into production or assumed portable to other hardware.

The old source/semantic thresholds are unchanged. Replacing an impossible old
bitwise-optimization comparison with a separately registered deterministic
control does not relax the reconstruction gate or hide the failed first attempt.

## Actual local results

These are MSEs for **one update from a fixed previously sampled state**, not new
full-trajectory terminal errors or editing scores.

| Input | Current level | FP16 cap20 best MSE | FP16 cap80 best MSE | FP32 cap20 best MSE |
|---|---:|---:|---:|---:|
| Apple | 741 | 2.1774244e-5 | 9.9338949e-6 | 2.0243746e-5 |
| Apple | 381 | 5.7885476e-4 | 4.6961952e-4 | 5.7503261e-4 |
| Apple | 1 | 2.1582302e-3 | 1.9846307e-3 | 2.1565815e-3 |
| Car | 741 | 3.6002632e-5 | 9.9675399e-6 | 3.8421709e-5 |
| Car | 381 | 6.5966317e-4 | 5.1962299e-4 | 6.5863470e-4 |
| Car | 1 | 1.2824207e-3 | 1.1681207e-3 | 1.2803525e-3 |

- FP16 cap20 and FP32 cap20: **0/6** selected problems reach tolerance.
- FP16 cap80: **2/6**, both at the earliest selected level. Apple reaches tolerance
  at evaluation47; car at66. Later probes use all80 evaluations and still fail.
- Gradients are finite and nonzero. In cap20 FP16 controls, the mean fraction of
  embedding coordinates changed after the FP16 cast ranges from0.7547 to0.9254
  across the six problems. The recorded predictions also change. This rejects
  the explanation that every update is lost to the cast, without proving partial
  quantization irrelevant.
- FP32 cap20 does not consistently improve local loss: the car's early loss is
  higher. Middle/final improvements are small. This restricted execution-only
  comparison does not support a precision upgrade alone as a sufficient fix.
- The DDIM epsilon-to-latent coefficient magnitude is0.08366/0.04890/0.04131 at
  current levels741/381/1. Later fixed states already inherit errors from the old
  20-evaluation trajectory. Poor late local results do not establish that the
  same errors would occur after a freshly calibrated, more accurate prefix.

## Supported conclusion and next experiment

Budget20 is insufficient at the early selected step. The evidence favors testing
error accumulation with a more accurately optimized prefix before adding an
editing controller. It does not yet distinguish inherited drift, the local
objective's attainable minimum and the full optimization strategy as unique
causes. Non-repeatable backward execution was an evidence-quality issue, but
cannot alone explain the large tolerance gaps observed here.

The justified next stage is a separately registered, deterministic **full
38-update cap20 vs cap80 calibration comparison**, using both original inputs.
Reoptimize each level from the actual current state produced by the new preceding
updates. Never splice these independent local optima into a trajectory or copy a
pivot into the sample. Keep all remaining settings and source thresholds fixed.
Measure local convergence, terminal clean-latent MSE, image/VAE-relative metrics,
geometry and independent pivot-free replay under each calibrated execution map.

If that bounded full-trajectory study fails, stop and review objective/pivot/
gradient evidence instead of escalating the budget without registration. Attention
editing stays conditional on the existing source gate; no new semantic pass or
holdout execution is authorized by local convergence.

## Verification, work and artifacts

Sixteen distinct offline contracts pass: nine existing null-calibration tests and
seven diagnostic tests, including finite-difference gradients, original-optimizer
equivalence, unchanged frozen inputs, budget-prefix agreement and target-free
replay. The real GPU evidence separately verifies repeated controls, measured
branch work, finite candidate/null tensors and unchanged parameter values.
Fifty-eight recorded output artifacts are hash-verified, in addition to frozen
code/input checks. No model parameter requires gradients or receives them.

Deterministic v2 used893 U-Net forwards /1,786 batch elements and769 persisted
backward evaluations. The retained first attempt and investigation used181
forwards /362 batch elements:76 persisted investigation backwards, plus19
backwards inferred from the completed failed cap20 code path whose error record
did not persist that counter. Total new forward work is1,074 /2,148; inferred
backwards are disclosed separately rather than labelled as measured counters.

Recorded v2 control/probe timers sum to192.059 seconds, excluding model loading
and weight hashing; control timers include some artifact I/O. These mixed-scope
single-run timers are not a throughput benchmark. Peak allocated CUDA memory is
about3.074 GiB for FP16 and6.165 GiB for FP32. No OOM/fallback occurred.

Local evidence, not guaranteed in a clean clone:

- `outputs/acceptance/2026-11-16-side-validation/development-round-12-calibration-diagnostic/`:
  frozen first plan, exact source controls, failed strict cap20 check and bounded
  original-helper repeat investigation.
- `outputs/acceptance/2026-11-16-side-validation/development-round-12-calibration-diagnostic-v2/`:
  independent deterministic registration, code/runtime snapshot,24 local reports,
  iteration CSVs, best-evaluated null/candidate tensors, model-value hashes,
  verification JSON and `REVIEW.md`. Standalone loss/gradient/effective-update
  figures are exported as PNG and SVG.

Defaults, source gate, masks, human ratings, reference weights, existing services
and holdout status remain unchanged. No attention editing, target-text output,
full-trajectory candidate, commit or push was included in this stage.
