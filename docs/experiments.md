# Experiments

The November 18 demo-only [image-fitting checks](demo-image-fit.md) validate
geometry and input handling. They do not alter frozen research inputs, calibration
settings, semantic acceptance cases or existing study metrics.

Reconstruction smoke metrics are available from Phase 3. No formal ablation suite,
editing evaluation or baseline comparison has been completed.

The subsequent [reconstruction-drift investigation](reconstruction-drift.md)
records controlled query-time/step-count/precision probes and the final residual
correction. At guidance 1/1, the repaired 50-step reconstruction matches the
VAE-level baseline on the two recorded natural images. The historical Phase 3/4
tables below remain unchanged as baseline evidence. Guidance 3/3 still triggers
plain-DDIM fallback in the strict final run. These diagnostics do not replace the
planned formal editing evaluation or establish a general quality improvement.

Phase 1 produced an unscored VAE-only smoke comparison at
`outputs/reconstruction/phase1/comparison.png`, using the real pretrained VAE and a
deterministic synthetic test card. Its configuration and environment are recorded
in the adjacent `report.json`. This is a structural baseline rather than an
ablation or a natural-image quality benchmark; no PSNR/SSIM or improvement claims
are reported. See `phase1-status.md` for executed verification.

Phase 2 saved a real 50-step inversion trajectory and execution metadata under
`outputs/inversion/phase2/`. This is a correctness smoke run, not reconstruction,
an ablation or an editing-quality result. Scheduler reference tests at different
step counts do not count as completed experimental ablations.

## Phase 3 measured reconstruction checks

All runs use the pinned SD1.5 revision, seed 42, 50 steps, clean alpha=1,
FP32 latent arithmetic and FP16 U-Net/CLIP. VAE-only and DDIM outputs are measured
against the original RGB image with data range 1. SSIM uses uniform 7×7 windows,
channel averaging and sample covariance. These are single-image smoke measurements.

| Input | Inversion/sampling guidance | VAE PSNR (dB) | VAE SSIM | DDIM PSNR (dB) | DDIM SSIM |
| --- | --- | --- | --- | --- | --- |
| Generated test card | 1 / 1 | 29.066 | 0.8367 | 10.990 | 0.6174 |
| Natural astronaut example | 1 / 1 | 26.799 | 0.8706 | 18.711 | 0.6283 |
| Natural astronaut example | 3 / 3 | 26.799 | 0.8706 | 12.888 | 0.4405 |

The generated card uses source prompt `a red apple on a colorful background`.
The natural example uses `a portrait of an astronaut wearing a spacesuit` and has
[documented public-domain provenance](../assets/examples/README.md). Negative prompts
are empty. VAE-only metrics are not DDIM inversion scores. The DDIM outputs exhibit
visible appearance drift; guidance 3 worsens preservation in this check. These
observations do not establish a general guidance relationship or improvement.

PNG comparisons, optional trajectories/previews, JSON reports and CSV metrics are
saved under `outputs/reconstruction/phase3/`, `phase3-astronaut/` and `phase3-cfg3/`.
Reports record input hashes, prompts, model revision, both guidance scales, seed,
exact grids, environment, runtime and peak allocated memory. No mask or editing
strength is used for reconstruction. This does not fulfill the two-ablation
completion requirement. See `phase3-status.md` for validation details.

## Phase 4 global editing smoke checks

The global editing script stores a matched source-prompt control in addition to
the target edit. Both branches use the exact same selected source latent, reverse
guidance and step suffix. The following actually executed checks use 50 inversion
steps (guidance 1), empty negative prompts and the pinned model with seed 42:

| Input / target | Strength | Reverse guidance | Start ID / reverse steps | Edited vs original PSNR / SSIM |
| --- | --- | --- | --- | --- |
| Astronaut / blue spacesuit | 0.50 | 5.0 | 481 / 25 | 22.081 dB / 0.7738 |
| Astronaut / blue spacesuit | 0.75 | 7.5 | 741 / 38 | 13.591 dB / 0.4953 |
| Astronaut / monochrome photo | 0.75 | 7.5 | 741 / 38 | 13.924 dB / 0.5257 |
| Coffee / blue ceramic cup | 0.75 | 7.5 | 741 / 38 | 20.252 dB / 0.5969 |
| Coffee / pencil drawing | 1.00 | 7.5 | 981 / 50 | 9.921 dB / 0.3187 |

Preservation scores are not prompt-alignment scores. Neither blue-color task
reliably achieved the desired recoloring, and the monochrome portrait retained
color. Strong settings produced more shape/appearance drift. These unsuccessful
semantic outcomes are retained. The 400×600 coffee case also verifies non-square
dimension preservation; its input has [CC0 provenance](../assets/examples/README.md).
The full-depth pencil-drawing target visibly produced a grayscale pencil style,
while its source-prompt control remained photographic. Cup geometry and surrounding
details also changed. Its low preservation score is not a failed style-alignment
metric; no quantitative prompt-alignment claim is made.

Outputs and per-run prompts/configuration, input hash, selected alpha/noise level,
exact state IDs, timings, environment and CSV/JSON metrics are in
`outputs/editing/phase4/`, `phase4-strong/`, `phase4-monochrome/`, `phase4-coffee/`
and `phase4-sketch/`.
The first run includes full inversion and partial source/target trajectories.
Changing both strength and guidance in these exploratory checks is not a controlled
ablation. Formal experiments and general editing-quality claims remain pending.

## Phase 5 local editing preservation check

Using the same 400×600 coffee image, authored ellipse mask, seed 42, pinned model,
50 inversion steps (guidance 1), strength 1 and reverse guidance 7.5, the target
is the same pencil-drawing prompt as Phase 4. Scores below use RGB [0,1] before
uint8 saving. Hard-mask rows have weights 1-M with 171,241 background pixels.

| Path | Background MAE | Background PSNR (dB) | Literal zero-mask MAE |
| --- | --- | --- | --- |
| Global control, both local operations disabled | 0.256230 | 9.554 | 0.256230 |
| Hard latent blend, before pixel composite | 0.042941 | 24.275 | 0.042941 |
| Hard latent blend + pixel composite | 0 | +infinity | 0 |
| Feather sigma 4, before pixel composite | 0.035245 | 25.358 | 0.035542 |
| Feather sigma 4 + pixel composite | 0.000500 | 44.433 | 0 |

Feather rows use the **effective soft mask**, so their weighted region differs
from the hard-mask region. There are 156,457 literal zero-mask pixels after
feathering. Relative to the original requested hard region, feathered final MAE
is 0.000292: the Gaussian support deliberately expands the editable area. Final
zero-support pixels were also checked for exact equality in exported PNGs.

The latent-only decrease is measured on this single example; exact final zeros
come from pixel compositing and are not evidence of exact latent-space or VAE
preservation. No regional SSIM/LPIPS is computed by zeroing other pixels, and no
prompt-alignment improvement is inferred from preservation scores. The edit
visibly applies pencil-like appearance inside the coarse mask, with geometry
changes and seam limitations retained.

Runs are saved under `outputs/editing/phase5-hard/`, `phase5-feather/`,
`phase5-global-control/` and `phase5-global-mask-evaluation/`. The last run supplies
the mask for evaluation while disabling both local operations; its PNG exactly
matches the no-mask global control. Hard/feather runs save uncomposited images
separately, allowing independent before/after pixel-composite evaluation without
another diffusion run. Exact float masks, settings/hashes, grids, metrics and
optional trajectories are retained. See `phase5-status.md`. This single-image
check does not complete Phase 7's formal ablation/baseline requirements.

## Phase 6 interface verification

The local Gradio flow was exercised with an actual drawn diagonal mask on the
600×400 coffee photograph, 50 steps, strength 1, reverse guidance 7.5, inversion
guidance 1, seed 42 and both local preservation operations enabled. A separate
full source reconstruction used guidance 1/1 and the existing residual-refined
selection. This is UI/integration evidence, not a formal ablation.

The successful browser run is stored in
`outputs/demo/4d0cef5917ee43e9978205bac8f61ff6/`. Loading took 7.535 s and combined
pipeline compute 40.138 s, excluding metrics, previews and saving. Full source
reconstruction scored 25.656006 dB / 0.7398565 SSIM against the original; VAE-only
scored 25.656064 dB / 0.7398561. The uncomposited edit background MAE was 0.0334535;
final compositing preserved all 224,992 literal zero-mask pixels exactly. The
15,008-pixel narrow stroke did not yield a convincing pencil drawing. Preservation
scores do not establish target-prompt success. JSON/CSV, images, exact masks and
the downloadable ZIP retain the settings and evidence. See `phase6-status.md` for
the front-end initialization issue found and repaired during browser testing.

## Phase 7 executed controlled evaluation

The fixed plan `configs/experiments/phase7.yaml` completed all **34 custom cases**
and **four labelled Diffusers DDIM Img2Img baseline cases** using the same cached
pinned revision. Two images and seed 42 were held fixed within each study. The
five studies cover plain/refined 20/50/100-step reconstruction, strength
0.25/0.5/0.75/1, guidance 3/5/7.5/10 and six mask-operation controls. This exceeds
the two-executed-ablation requirement; it does not establish broad generalization.

Plain reconstruction PSNR increased from 15.373 to 19.158 dB (astronaut) and
16.450 to 20.690 dB (coffee) between 20 and 100 steps. Refined-selected outputs
remained near their VAE-only references, with eight times as many U-Net forwards
in these runs and retained residual-tolerance warnings. Higher strength/guidance
reduced preservation in the measured editing series. At full strength, pencil-like
rendering appeared with substantial composition/geometry changes.

Using the fixed coffee requested-background region, global MAE was 0.256230,
latent-only 0.042941 and feathered-latent 0.035132. Hard pixel compositing gave
exact zero background error; feathered latent+pixel gave 0.000292 because support
expands beyond the original region. Effective-mask metrics are separate and
literal zero-support pixels remain exact. Compositing is not a latent-algorithm
improvement, and preservation metrics do not rank prompt alignment.

The baseline matches 25/50 reverse updates at strengths 0.5/1 but performs 25/50
total forwards versus custom 100/150 including inversion and source control.
Random posterior/noising and FP16 baseline latent updates also differ. Custom
preservation scores were higher on these four pairs; this does not establish
equal-cost or semantic superiority. No inpainting checkpoint was downloaded.

All runs save inputs, masks, prompts, seed, model/revision, source/software identity,
settings, actual IDs/budgets, runtime, outputs and CSV/JSON metrics. The manifest
and frozen inputs support strict checksum-verified resume. Raw records, five
PNG/SVG curves, per-study grids and baseline grids are in
`outputs/experiments/phase7/`. A portable [review bundle](results/phase7/README.md)
includes aggregate CSV/JSON, plots and explicitly resized JPEG previews.
See [the complete Phase 7 report](phase7-status.md) for tables, actual validation
totals, metric denominators and remaining limitations. CLIP/LPIPS were not run.

## Text-to-image interface checks (2026-11-18)

The separately requested txt2img UI/API was checked with synthetic CPU components:
39 focused offline checks passed (four opt-in skips), and 41 focused checks passed
with local HTTP enabled (two real-model skips). The endpoint generated/downloaded
synthetic outputs and reported seed, dimensions, state IDs and hashes. No real
checkpoint/GPU check, browser-generated result, semantic rating or new acceptance
case was executed. Browser automation stopped on uncertain URL recognition.
These initial checks are interface/mathematical contracts only; they do not alter the
calibration comparison or attention gate. See `txt2img-demo.md` for details.

## Four-tab UI follow-up (2026-11-18)

The tabbed interface's focused CPU/HTTP suite passed 43 checks with two real-model
skips. A separate in-app-browser smoke visited all four modes, completed synthetic
CPU operations, retained distinct prompts/results across switches and preserved
the local paint layer/toolbars. Painting and local execution worked after return.
This later check completes the tab UI/lifecycle smoke; it does not turn the earlier
synthetic outputs into pretrained or acceptance evidence. The temporary server
was stopped and no research run or production server was changed. See
`demo-tabs.md` for the screenshot, API mapping and precise test scope.
