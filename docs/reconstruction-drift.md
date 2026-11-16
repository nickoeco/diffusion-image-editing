# Reconstruction drift: diagnosis and repair

Date: 2026-11-15. This work takes priority over starting Phase 5, as requested by
the user. Phase 4 was first committed as `3852a20` (core/tests) and `2d677e5`
(sample CLI and measured limitations). No mask/UI phase has been started.

## Diagnosis

The original 50-step same-prompt reconstruction changed visible geometry and
appearance despite passing fixed-epsilon reference tests. Those tests validate
DDIM coefficients and equations; they do not validate the approximation that the
network noise prediction remains unchanged between source and destination states.

The controlled diagnostic holds image, VAE latent, prompt, guidance 1, model
revision, coefficient equations and reverse sampler fixed. It varies inversion
query time, selected step count or U-Net arithmetic precision. All runs use the
pinned cached SD1.5 model, seed 42 and the recorded astronaut example.

| Probe | Original-image PSNR (dB) | SSIM | Final roundtrip latent MSE |
| --- | --- | --- | --- |
| Current-state query, 50 steps, FP16 U-Net | 18.711 | 0.6283 | 0.105339 |
| Destination-time query on current state, 50 steps | 18.797 | 0.7132 | 0.093019 |
| Current-state query, 100 steps | 19.158 | 0.7178 | 0.085431 |
| Current-state query, 200 steps | 19.320 | 0.7271 | 0.080246 |
| FP32 U-Net arithmetic, 50 steps | 18.713 | 0.6280 | 0.105355 |

The FP32 probe widens the same FP16-quantized U-Net weights and holds CLIP
embeddings fixed. It isolates arithmetic precision rather than comparing a
separately downloaded full-precision checkpoint. Step-count probes retain the
checkpoint's actual scheduler grids; their terminal IDs differ slightly, so they
are not pure fixed-endpoint step-count ablations.

Teacher-forced local replay reverse-samples each stored inverted state one step
and compares it to the preceding source state. The baseline's largest measured
one-step error is 21 -> 1: MSE 0.00100020, max absolute error 0.240711. Correct
coefficient indexing alone does not prevent this network-dependent discrepancy.
The global final error then accumulates along the sampled trajectory. Merely
changing query time or widening arithmetic did not resolve the observed drift.

## Numerical repair

For a source state z_t and a noisier destination s, the actual reverse DDIM map is
`R_s(z)=a*z+b*epsilon_theta(z,s,c)`, with `a=sqrt(alpha_bar_t/alpha_bar_s)`.
The new residual correction evaluates `r=R_s(z)-z_t` and proposes
`z <- z - damping*r/a`. It retains the best evaluated candidate, halves damping
after a rejected trial, and enforces a fixed evaluation budget. This is a damped
fixed-point numerical solve of our existing DDIM equation, not model training.
See `diffusion/refinement.py` and the derivation in `implementation.md`.

The plain inverter remains available with `refinement_steps=0` and keeps the
original timestep contract. The reconstruction pipeline defaults to 12 destination
evaluations per inversion step, RMS tolerance 1e-6 and unchanged eta=0 sampling.
Every run records per-level residuals, evaluation counts and convergence flags.

### Local residual improvement is insufficient

An initial unprotected four-step, guidance-3 synthetic integration test failed:
the refined final latent MSE was approximately 0.1502 versus plain 0.0973. This
was retained as a stress case, not removed as irrelevant. Reconstruction now runs
both refined and plain roundtrips and selects the lower final latent MSE against
the original clean VAE latent. `ReconstructionFidelityCheck` records both errors,
the selected method and candidate refinement diagnostics, including on fallback.
The regression checks this objective does not worsen; a separate real natural-image
test at 50 steps requires a substantial measured improvement.

Both candidates are independently reverse-sampled from their own terminal latent.
The selected image is decoded from the selected sampler result. Original pixels
and stored source states are not substituted into reverse updates. The check is
specific to reconstruction and does not guarantee arbitrary perceptual metrics
or preservation under changed target conditioning.

## Final verified 50-step results

Guidance values below apply identically to inversion and sampling. Metrics use
floating RGB [0,1], whole-image PSNR, and channel-averaged uniform 7x7 SSIM with
sample covariance, before uint8 saving. Inputs, prompts, hashes, model revision,
grids, environment and metrics are stored with each run.

| Input / guidance | VAE PSNR / SSIM | Plain DDIM PSNR / SSIM | Selected reconstruction PSNR / SSIM |
| --- | --- | --- | --- |
| Astronaut, 1/1 | 26.799 / 0.8706 | 18.711 / 0.6283 | 26.799 / 0.8706 (refined) |
| Coffee, 1/1 | 25.656 / 0.7399 | 20.356 / 0.5842 | 25.656 / 0.7399 (refined) |
| Astronaut, 3/3 | 26.799 / 0.8706 | 12.888 / 0.4405 | 12.888 / 0.4405 (plain fallback) |

For guidance 1/1, refined-vs-VAE-only PSNR is 68.462 dB on the astronaut image
and 68.058 dB on coffee. Their roundtrip latent MSEs fall from 0.105339 to
4.491e-6 and from 0.100584 to 3.344e-6 respectively. Visible geometry/appearance
drift is greatly reduced on both inspected examples; VAE reconstruction loss
remains. This is a two-image verification, not a broad quality benchmark.

At guidance 3/3 the tight refined candidate's latent MSE was 0.474369, worse than
plain 0.413589, so the final pipeline correctly selected plain. High-guidance
fidelity remains unresolved by this correction. A prior loose-tolerance candidate
at 1e-4 measured 18.753 dB / SSIM 0.7133, but the sensitivity shows that more local
refinement is not a universal quality guarantee. Matching guidance 1/1 is the
verified reconstruction setting. Changed guidance/text is a different reverse map.

### Tolerance, convergence and cost

A loose 1e-4 early-stop threshold used 168 destination evaluations on the astronaut
image and measured 25.905 dB / SSIM 0.8610; coffee used 165 evaluations and measured
25.350 dB / SSIM 0.7360. Tightening to 1e-6 used the full 600 evaluations in both
50-step runs and recovered the VAE-level results above.

**None of the 50 levels reached the strict 1e-6 local RMS threshold within the
12-evaluation budget in the final real-image runs.** The maximum remaining local
RMS was 0.001862 on astronaut and 0.001646 on coffee. These outcomes are explicitly
warned and reported; close final images are not evidence of an exact per-step
inverse or numerical convergence to that threshold.

With 50 steps and a full budget, reconstruction performs up to 800 U-Net calls:
50 initial inversion estimates, 600 residual evaluations, 50 refined reverse
steps, and 100 plain-path calls for the end-to-end selection. Paired CFG calls
contain twice the latent batch. The final observed reconstruction runtimes were
about 38.52 s (astronaut 1/1), 38.31 s (coffee 1/1) and 56.55 s (astronaut 3/3),
excluding model loading, metrics and file saving. These are single-run observations,
not controlled throughput benchmarks; frequency/load and earlier probes varied.
The memory strategy keeps snapshots on CPU and builds no gradient graph.

## Reproduction and artifacts

```powershell
# Plain baseline; supply a unique directory to preserve historical runs.
.\.venv\Scripts\python.exe scripts/test_reconstruction.py --image assets/examples/astronaut.png --source-prompt "a portrait of an astronaut wearing a spacesuit" --refinement-steps 0 --output-dir outputs/reconstruction/plain-check
# Current default: budgeted correction with end-to-end latent-error selection.
.\.venv\Scripts\python.exe scripts/test_reconstruction.py --image assets/examples/astronaut.png --source-prompt "a portrait of an astronaut wearing a spacesuit" --save-intermediates --output-dir outputs/reconstruction/refined-check
# Controlled query/step-count/precision probes; CUDA required, weights cached.
.\.venv\Scripts\python.exe scripts/diagnose_reconstruction.py --local-residuals
```

Historical plain probe data: `outputs/diagnostics/reconstruction/report.json`.
The first fixed-12 prototype is under `outputs/diagnostics/refined-astronaut/`.
Loose-tolerance runs: `outputs/reconstruction/drift-fix-astronaut/`,
`drift-fix-coffee/` and `drift-fix-cfg3/` (these precede the global selection check).
Final guarded runs: `drift-fix-astronaut-tight/`, `drift-fix-coffee-tight/` and
`drift-fix-cfg3-tight/`. Coffee's original baseline is `drift-coffee-plain/`.
Final JSON reports identify the selected method and preserve the rejected candidate
statistics; image/trajectory previews are optional as before.

Before/after grids are saved as `outputs/diagnostics/drift-comparison-astronaut.png`
and `drift-comparison-coffee.png` (original, VAE, plain DDIM, refined DDIM).
The final astronaut and coffee comparisons were visually inspected.

## Verification and remaining work

- Full explicit GPU/integration suite: **178 passed**, four existing upstream
  tokenizer FutureWarnings, no failed/skipped tests.
- Default offline CPU suite: **159 passed, 19 skipped** (explicit GPU/integration).
- Affine closed-form inverse, unstable fixed-point best-candidate retention,
  early stopping/budget boundaries, invalid configuration and state-dependent
  analytic roundtrip tests passed. The unprotected adverse test led to the
  end-to-end selection guard; its regression now passes. A real 128x128 natural
  image at 50 steps requires >10x latent-MSE improvement.
- `pip check`, compileall and diagnostic/reconstruction CLI help passed.
- No new dependencies, checkpoint download, masking or UI phase was introduced.

The user-facing default reconstruction is now verified near its VAE-only baseline
at guidance 1/1. High guidance, coarse schedules and global editing remain quality
limitations; the global editor keeps plain inversion as its default and exposes
refinement as an explicit source-inversion option without a reconstruction guard.
Do not generalize this repair to target-edit quality. Phase 5 remains deferred
while these fidelity results are reviewed.

Subsequent status: the user explicitly authorized Phase 5 after reviewing/pushing
this repair. Local editing adds mask-guided source-state blending and optional
pixel compositing; reconstruction retains its independent sampler and the fidelity
regression tests above. See `phase5-status.md`. The high-guidance limitation remains.
