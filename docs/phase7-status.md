# Phase 7 status: reproducible evaluation

Date: 2026-11-15. Phase 6 was committed and pushed as `bb55617` (implementation
and tests) and `9d5adff` (documentation/evidence) before Phase 7 began.

## Purpose and scope

Provide reproducible study configuration, actual controlled ablations, an explicitly
labelled built-in Img2Img baseline, automatic tables/grids/plots, and honest metric
definitions. Core DDIM inversion, sampling, CFG, strength and mask equations remain
unchanged. No training, new checkpoint, advanced editing method or perceptual-model
download is introduced. Matplotlib 3.10.7 is the optional evaluation dependency.

The fixed plan is `configs/experiments/phase7.yaml`. Its 34 custom runs comprise:

| Study | Samples | Changed variable | Cases |
| --- | --- | --- | --- |
| Plain reconstruction | Astronaut, coffee | 20 / 50 / 100 steps | 6 |
| Refined reconstruction | Astronaut, coffee | 20 / 50 / 100 steps | 6 |
| Global editing strength | Astronaut, coffee | 0.25 / 0.5 / 0.75 / 1 | 8 |
| Global editing guidance | Astronaut, coffee | 3 / 5 / 7.5 / 10 | 8 |
| Mask operations | Coffee + authored ellipse | Six named operation presets | 6 |

Within each step study the refinement method is fixed. Global edits use 50 source
inversion steps, inversion guidance 1, no residual refinement and pencil-drawing
targets; strength defaults to 0.75 for guidance studies and guidance to 7.5 for
strength studies. The mask study uses strength 1, guidance 7.5 and feather sigma 4
where selected. Seed 42 and empty negative prompts remain fixed. Astronaut is
512×512; coffee is 600×400. Input provenance, exact prompts and hashes are saved.
This two-image, single-seed study is descriptive, not a generalization benchmark.

Mask presets are global, latent only, feathered latent, pixel only, latent+pixel
and feathered latent+pixel. Pairwise global/latent-only, global/pixel-only,
latent-only/latent+pixel and hard/feathered controls isolate a switch. Comparing
global against a combination changing several operations is not a single-effect
estimate. All coffee outputs use the same **requested-mask** evaluation region;
effective masks and their changed feather-support denominators are separate.

## Reproduction and artifact contracts

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,dev,evaluation,demo]"
.\.venv\Scripts\python.exe scripts/run_experiments.py --dry-run
.\.venv\Scripts\python.exe scripts/run_experiments.py --output-dir outputs/experiments/phase7
.\.venv\Scripts\python.exe scripts/run_img2img_baseline.py outputs/experiments/phase7
.\.venv\Scripts\python.exe scripts/summarize_experiments.py outputs/experiments/phase7
```

Inputs are frozen and verified before loading weights. `manifest.json` records
the complete plan, source-code checksums, input hashes, pinned model configuration
and environment. Each `runs/<case-id>/` saves original/VAE/output/control images,
masks where available, strict JSON, CSV and a labelled grid. Local outputs retain
pre-composite target and source-control images and exact float masks. Reconstruction
reports retain the selected plain/refined method, both clean-latent errors and
per-level refinement diagnostics, including fallback cases.

`--resume` requires the same input, configuration, code and software/runtime
identity, verifies completed artifact hashes, and runs only missing cases. Failures
write `error.json` and propagate. A partial report explicitly lists missing cases;
it cannot silently become complete. Existing directories are protected from
accidental overwrite. Plot/table regeneration uses saved floating metrics rather
than rescoring quantized PNGs.

Timing surrounds pipeline work with CUDA synchronization and excludes model
loading, metrics, previews and saving. There is no dedicated warmup; first-case and
shape-specific initialization can remain. The earlier local demo was paused during
the batch to reduce VRAM contention. U-Net hooks measure actual forwards and batch
elements (including CFG's doubled batch). Timings are descriptive, not a controlled
latency competition. Peak CUDA allocation refers to the current process.

## Labelled baseline comparison

The standalone `scripts/run_img2img_baseline.py` uses Diffusers 0.35.1's built-in
`StableDiffusionImg2ImgPipeline` with DDIM eta=0 and the same clean-alpha convention.
Only this labelled script constructs the editing baseline; custom pipelines never
call it. It loads the same cached component weights and uses the pipeline's
supported latent input/output so VAE codec execution remains FP32. Unlike custom
editing, it samples the VAE posterior and adds Gaussian noise, then uses target
conditioning with no source inversion or matched source-control branch.

Strengths 0.5/1 at 50 requested steps match actual reverse IDs exactly: 25/50 reverse
updates starting at 481/981, ending through timestep 1. Custom sampling also records
the separate clean -1 state. Actual callback IDs and measured forwards are checked;
the baseline rejects strengths where its floor mapping differs from custom ceil.
Weights, input pixels, seed, target/negative prompt and reverse guidance are shared.

The custom editor requires 100/150 total U-Net forwards at strengths 0.5/1 because
it includes 50 source-inversion calls and 25/50 calls in each source-control/target
branch. The baseline uses 25/50 total calls. CFG batch-element totals also differ.
Thus reverse budgets match, **total budgets do not**. Baseline update arithmetic
follows its inference dtype, while custom latent updates use FP32. These differences
preclude attributing score or runtime differences solely to inversion. No new safety
checker is loaded for the fixed-input offline baseline; it is not a deployment tool.

## Metric definitions and limits

Whole-image PSNR uses RGB float [0,1], double MSE and data range 1. SSIM uses uniform
7×7 windows, channel averaging and sample covariance. VAE-only, output-vs-original
and output-vs-VAE scores remain separate. Source controls and raw/final local
outputs are separately available. These scores measure preservation, not whether
the output satisfies the target prompt.

Background MAE/MSE use weights 1-M and denominator `3*sum(1-M)`. Literal M==0 MAE
uses `3*count(M==0)`. A fixed requested mask enables meaningful cross-mode region
comparison; effective masks separately quantify feather support. No regional
perceptual score is computed by replacing background with zeros. Empty regions are
undefined. Exact positive-infinite PSNR is null in JSON and identified explicitly
in `nonfinite_metrics`; per-case CSV retains infinity. Plots omit nonfinite points
and state this, rather than substituting a finite score.

No confidence intervals, statistical significance or CLIP/LPIPS claims are made.
Final pixel equality is a compositing result and must not be credited to latent
blending. VAE loss, approximate inversion, shape drift, unreliable color editing,
mask seams and high-guidance reconstruction limits remain. Refined residual
tolerance warnings are retained; final candidate selection does not imply local
convergence or guarantee arbitrary pixel/perceptual metrics.

## Measured reconstruction and editing ablations

All numbers below were read from completed per-case floating-tensor reports.
The original VAE-only references are astronaut 26.799345 dB / 0.8706192 SSIM
and coffee 25.656064 dB / 0.7398561 SSIM.

| Sample | Steps | Plain PSNR / SSIM | Refined-selected PSNR / SSIM | Plain / refined total U-Net forwards |
| --- | --- | --- | --- | --- |
| Astronaut | 20 | 15.372721 / 0.4130588 | 26.799812 / 0.8706344 | 40 / 320 |
| Astronaut | 50 | 18.710870 / 0.6282616 | 26.798922 / 0.8706198 | 100 / 800 |
| Astronaut | 100 | 19.158393 / 0.7178167 | 26.799174 / 0.8706209 | 200 / 1600 |
| Coffee | 20 | 16.449939 / 0.4530054 | 25.646349 / 0.7395258 | 40 / 320 |
| Coffee | 50 | 20.356445 / 0.5841647 | 25.656006 / 0.7398565 | 100 / 800 |
| Coffee | 100 | 20.690429 / 0.6006284 | 25.657227 / 0.7398506 | 200 / 1600 |

Plain-DDIM preservation improved with more steps for these two inputs but remained
below VAE-only reconstruction. Residual-refined candidates were selected in all
six runs and stayed close to their VAE references. Their total forward count was
eight times the plain path, including the additional plain candidate and final
selection. The strict local tolerance was not reached at all levels; warnings and
diagnostics remain saved. Output-vs-VAE scores are in the complete CSV/JSON.
More steps are not asserted to improve refined scores monotonically.

| Changed value | Astronaut PSNR (dB) | Coffee PSNR (dB) |
| --- | --- | --- |
| Strength 0.25, guidance 7.5 | 24.176172 | 25.100248 |
| Strength 0.50, guidance 7.5 | 18.739499 | 21.912628 |
| Strength 0.75, guidance 7.5 | 14.150940 | 16.855654 |
| Strength 1.00, guidance 7.5 | 8.821085 | 9.920939 |
| Guidance 3, strength 0.75 | 15.497181 | 19.024951 |
| Guidance 5, strength 0.75 | 14.796441 | 17.649204 |
| Guidance 7.5, strength 0.75 | 14.150940 | 16.855654 |
| Guidance 10, strength 0.75 | 13.467095 | 16.499426 |

Stronger settings reduced whole-image preservation in these measured series.
Full-depth outputs visibly have pencil-like rendering, but the astronaut changes
from a recognizable portrait into a different full-body composition and cup
geometry/details also change. These are retained outcomes, not identity-preserving
edit successes. Smaller strengths retain more source detail but do not necessarily
achieve the requested style. No semantic ranking is inferred from preservation.

## Measured mask and baseline results

The requested ellipse has 171,241 background pixels; this denominator is identical
in every row below. Feathered effective-mask denominators are reported separately.

| Operation | Raw requested-background MAE | Final requested-background MAE |
| --- | --- | --- |
| Global | 0.2562298 | 0.2562298 |
| Latent only | 0.0429411 | 0.0429411 |
| Feathered latent (sigma 4) | 0.0351322 | 0.0351322 |
| Pixel composite only | 0.2562298 | 0 |
| Latent + pixel composite | 0.0429411 | 0 |
| Feathered latent + pixel composite | 0.0351322 | 0.0002924 |

Feathered latent+pixel output has effective weighted-background MAE 0.0005002;
literal zero-support pixels remain exact. Its nonzero **requested** background
error reflects the expanded feather support. Hard pixel-composite zeros follow
directly from compositing and do not prove VAE/latent-space pixel identity. The
raw latent-only reduction is measured on this one image, not a general quality
guarantee. Coarse masks can cut across changed object geometry and create seams.

The four completed baseline cases use the exact reverse grids described above:

| Sample / strength | Custom PSNR / SSIM | Baseline PSNR / SSIM | Custom / baseline total forwards |
| --- | --- | --- | --- |
| Astronaut / 0.5 | 18.739499 / 0.6592281 | 15.285879 / 0.4765154 | 100 / 25 |
| Astronaut / 1 | 8.821085 / 0.2677486 | 8.287496 / 0.2394363 | 150 / 50 |
| Coffee / 0.5 | 21.912628 / 0.6078337 | 18.843112 / 0.4657659 | 100 / 25 |
| Coffee / 1 | 9.920939 / 0.3186833 | 8.463579 / 0.1596296 | 150 / 50 |

Custom preservation scores are higher in these four cases, with unequal total
work and declared algorithm/precision differences. This is not evidence of better
target-prompt alignment or equal-cost superiority. The full-depth coffee custom
output is monochrome and the baseline has a brown pencil-like appearance; both
change cup geometry and source detail. The observed baseline final latent dtype
is FP16, U-Net/CLIP are FP16 and VAE remains FP32 in both paths.

## Executed verification and deliverables

| Check | Observed result |
| --- | --- |
| Offline `pytest -q` | 225 passed, 23 skipped, 4.27 s |
| `pytest -q --run-demo --run-gpu --run-integration` | 248 passed, 10 warnings, 46.17 s |
| `pip check` | No broken requirements |
| Fixed custom suite | 34/34 completed, five studies, no missing cases |
| Labelled Img2Img baseline | 4/4 completed, exact reverse ID/budget checks passed |
| Real-suite `--resume` | All 34 completed records checksum-verified and reused; no model reload |
| Saved metric/figure regeneration | Complete CSV/JSON, five PNG/SVG plots, per-study grids and four baseline grids |

Default skips are three CUDA-only tensor tests, nineteen integration checks and
one HTTP demo check. All were enabled in the full suite. Seven warnings are
Transformers tokenizer-default FutureWarnings and three are Gradio/Starlette's
deprecated HTTP-422 constant. Warnings were retained. The larger experiment logs
also preserve the six residual-tolerance warnings noted above.

Tests cover one-variable plan expansion and validation, lightweight config reads,
frozen inputs, finite/invalid settings, actual forward budgets, all mask modes,
reconstruction selection/diagnostics, no-load resume, changed input/source/software/
snapshot rejection, output integrity, explicit failure records and plots/grids.
The opt-in 64×64 cached-weight test exercises the real custom runner and built-in
baseline, checking actual IDs, forwards and FP32 VAE execution. Existing component,
mathematical, fidelity, global/local and HTTP regression checks still pass.

Raw artifacts are under `outputs/experiments/phase7/`: `manifest.json`, immutable
`inputs/`, 34 `runs/`, four `baseline/` cases and `summary/`. Execution, baseline
and resume logs are adjacent under `outputs/experiments/`. Summed custom pipeline
compute was 443.517 s; the largest process CUDA allocation was 3,369,428,992 bytes.
No general speed or memory advantage is claimed. The local Gradio service was
restored at `http://127.0.0.1:7860` after GPU validation.

A [portable result bundle](results/phase7/README.md) is included for repository
review: full aggregate CSV, baseline comparison CSV, source/input/model/software
identity and per-case metrics in JSON, PNG/SVG plots and representative JPEG grids.
JPEGs are explicitly resized documentation previews; measured tensors and full
PNGs remain in the run directories. Figures and selected grids were visually
reviewed for labels, numeric axes, layout, missing metrics and retained failures.

Phase 7's reproducible evaluation gate is complete within the recorded scope,
including more than two executed ablations and a labelled baseline comparison.
Phase 7 implementation, configuration and tests are committed as `2e9af60`, with
results and documentation in the following logical commit. No further stretch
phase was started. Dataset size, single seed, semantic reliability and changed geometry
remain material limits. Configured seeds do not promise bitwise reproduction
across different hardware/software.

## References

- [Diffusers Img2Img API](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img)
- [Pinned Diffusers implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py)
- [Matplotlib installation](https://matplotlib.org/3.10.0/install/index.html)
- [Input provenance](../assets/examples/README.md)
