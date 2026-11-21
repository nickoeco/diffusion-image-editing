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

## Next acceptance set: registered, not executed

The user requested a separate semantic acceptance gate. The new
[protocol](acceptance.md) and [12-row table](acceptance-review-template.csv)
pre-register four recoloring, four replacement and four localized style tasks,
with 6 development/6 holdout cases. Inputs, masks and provenance are pending; no
new benchmark inference, tuning or holdout scoring has been performed. Historical
coffee/astronaut outputs cannot serve as unseen holdout material.

Future comparisons must report semantic success, unrelated-content preservation
and boundary quality separately, along with existing floating PSNR/SSIM/background
MAE. Fixed requested-mask boundary zooms and pre-registered critical details are
required. A passing pixel metric cannot override a semantic or boundary failure.
The method and parameters must be frozen before holdout execution. Any renderer
smoke using old outputs is tooling evidence only, not a new acceptance result.

### Isolated local development follow-up

A subsequent actual-editability diagnostic compared empty-null and saved
calibrated-null inputs on the same timestep741 latent at guidance12, using a
single color-word substitution. All four source controls replayed old PNGs exactly.
Unmasked/uncomposited target branches still failed to show reliable requested
color with preserved structure: calibrated apple remained red/orange and car
blue/teal with substantial geometry/background changes. This separates improved
source reconstruction from actual editability; it does not waive the incomplete
calibration gate or demonstrate failure of an unimplemented attention method.
Work:304 new U-Net calls /608 batch elements, no new inversion/calibration/backward.
No six-case expansion, human grade, holdout output or default change. Full evidence
is local under `development-round-09-editability/` in the isolated acceptance root;
see [advanced-reconstruction.md](advanced-reconstruction.md) for the lineage.

The next [advanced reconstruction extension](advanced-reconstruction.md) crossed
inversion/reverse guidance 1/7.5/12 on the two exposed recoloring sources, with no
masks, compositing or edited text, and retained an independent residual-corrected
12/12 control. Existing methods failed the declared high-CFG gate. A bounded
null-text input-calibration prototype then improved VAE-relative reconstruction
PSNR from 19.525 to 36.952 dB on apple and 15.895 to 31.315 dB on car, with frozen model
weights. A separate batch-layout correction was required after FP16 separate-vs-
batched replay sensitivity was measured; both trials are retained. Primary saved-null
replays are exact, but latent MSE .002158/.001283 exceeds the predeclared .001 limit,
and 0/38 local levels per input reached 1e-5 in 20 evaluations. No attention editing,
new human rating, holdout execution, threshold relaxation or default change follows
this incomplete gate. See the extension report for full budgets and controls.

An explicitly labelled color-postprocessing experiment then compared a color-only
baseline against prior diffusion plus the same correction on the two development
recoloring cases. Fixed numeric green/yellow swatches provide additional supervision;
these are not prompt-only DDIM improvements. The source/effective masks and feather
were unchanged. A gamut-bounded chroma-direction projection preserved operation-input
linear-sRGB luminance to numerical precision before PNG export. Body colors visibly
changed, with remaining car blue edges, vivid apple green and inherited hybrid
geometry/texture changes requiring human review. Exact-zero-mask pixels stayed
unchanged. Ten offline tests passed; no new inference was performed, and the hybrid's
prior diffusion cost is disclosed. Two-case/four-output grading is available in the
local review page; no pass or default change is claimed. Full evidence, assumptions,
W3C luminance reference, quantization errors and matched-input metric definitions
are retained under
`outputs/acceptance/2026-11-16-side-validation/development-round-06/FINDINGS.md`.

A two-case source-reference pilot subsequently changed only the reference epsilon
branch: `epsilon_source + 12*(epsilon_target-epsilon_source)` replaced conventional
unconditional CFG, using both predictions at the same edited latent/timestep.
Original prompts, start 741/strength .75, masks, blending and final compositing
stayed fixed. The controls exactly replayed previous PNGs. Candidates retained
red/orange apple color and blue/teal car color while altering texture/shape, so
the pre-registered screening condition was not met and six-case expansion was
not executed. Apple PSNR rose despite unsuccessful recoloring. Six offline tests
passed; actual work was 328 U-Net calls / 556 batch elements. The experimental
predictor remains output-local; defaults, human ratings and holdout execution
were unchanged. Qualified findings and unsuccessful examples are retained in
`outputs/acceptance/2026-11-16-side-validation/development-round-05/FINDINGS.md`.

A subsequent two-task conditioning diagnostic used identical seed-42 Gaussian
starts across source/full-target/color-word-only prompts and compared those with
stored source-inverted starts at actual timestep 741 and 981. Direct custom-DDIM
generation produced the requested green apples and yellow cars. Actual token
records showed no truncation, and CLIP embeddings/U-Net predictions responded to
the target text. At timestep 741, unmasked target samples still retained much of
the source color; at 981, the car became yellow with substantial scene/geometry
changes while the apple remained mixed red/green. This narrows the issue to the
current editing trade-off under source-derived states; it does not prove a unique
inversion bug. Source/full-target 741 outputs exactly replayed the prior raw
diagnostic. Work: 988 U-Net calls / 1,816 batch elements; four offline numerical
tests passed. No human ratings or holdout outputs were added. Exact prompts,
token/epsilon records, tensors and qualified findings are stored locally under
`outputs/acceptance/2026-11-16-side-validation/development-round-04/`.

An additional isolated development diagnostic changed only `latent_blending`
from true to false relative to the guidance-12 profile, keeping the requested
mask, prompts, strength, steps, seeds, precision and final compositing fixed.
Because shared pipeline files had concurrent changes, an output-local runtime
snapshot restored the prior recorded source hashes. Six fresh controls reproduced
the prior saved PNGs exactly before six new candidates ran. Preliminary visual
inspection still found a blue car and a predominantly red apple after blending
was disabled; fixed-requested-background MAE increased for all six cases. These
are diagnostic observations, not automatic acceptance grades or a unique root
cause. Human review remains pending. The successful runs used 1,512 U-Net calls;
an initial evidence-validator tuple/list failure is retained and brings total
work to 1,638 calls. See the local
`outputs/acceptance/2026-11-16-side-validation/development-round-03/ANALYSIS.md`
for controls, per-case metrics, verification limits and the unchanged holdout gate.

The follow-up now includes a [model-free human review UI](review-ui.md) for the
two candidates, with independent score/reason controls and literal source/target
prompts beside all comparisons. Prompt-bearing derived figures retain original
pixel areas; hashed raw run comparisons remain untouched. This adds no inference,
human rating, profile selection or holdout evidence. The verification run recorded
244 passed / 24 skipped default tests, plus 11 passing opt-in review tests covering
Gradio construction and callback navigation with temporary ratings. The live
loopback server returned HTTP 200. Front-end browser interaction verification
remains pending: computer-use stopped because it could not confidently determine
the Windows browser URL. Construction/callback checks do not prove rendered UI
behavior. No synthetic test scores were added to the real human review table.

After the initial registration, a separate local working copy at
`outputs/acceptance/2026-11-16-side-validation/` assembled twelve real-photo inputs
and masks. Six first-round development outputs were reviewed by the user: 1 pass,
2 partial and 3 fail, with all preservation/boundary scores 2. Reasons identify
unchanged colors, weak style and a floating vase. Raw ratings/reasons are retained;
reviewer identity/time fields were removed from current tables at user request.

Two declared one-variable profiles produced twelve new development outputs on
the same six sources (strength 1, or guidance 12; other conditions fixed). Their
manual ratings remain blank. Parameter/hash/crop verification passed, but this is
not semantic acceptance. Holdout outputs remain zero and method selection/freeze
is pending the next development review. No per-case best-output selection or
PSNR-based improvement claim is made.

## Editing demo computation follow-up

The demo now defaults to Edit-only and skips unselected VAE/DDIM/source-control
work. Synthetic call-count checks distinguish real branch skipping from hiding
computed images: at three levels and strength 0.5, Edit-only uses five U-Net
forwards, plain DDIM-only six and VAE-only zero. Combined outputs share one VAE
baseline decode. The cached-weight 64×64 check measured six forwards for a
four-step, strength-0.5 edit and zero for VAE-only. Browser validation uploaded the
astronaut image, checked mode-specific mask visibility and painting after a mode
switch, and displayed a VAE-only result with 0.61 s compute / 7.01 s first loading.
This validates UI routing and resource use, not semantic editing quality or any
acceptance rating. See `demo-controls.md` for the complete behavior and checks.

## Pretrained instruction-editing reference

After explicit authorization for new weights, fixed-revision InstructPix2Pix
generated one output for each exposed development recoloring case: a green apple
and yellow car body. Background tint and unrelated roof/window/detail changes
remain; no human scores or acceptance pass were assigned. The two cases used
100 measured U-Net forwards / 300 batch elements total, zero backward passes and
7.451 summed inference seconds. Six offline tests pass and 24 recorded artifacts
were verified. Different weights, conditioning, scheduler, precision and total
work prevent model-only or matched-compute causal claims. See
[instruction-reference.md](instruction-reference.md) for settings, measured
preservation metrics, evidence locations and limits. No holdout output was made.

## Fixed-output localization reference and route decision

Two registered composites per input were computed from the same fixed reference
output: a hard requested mask and a four-pixel inward smoothstep feather, blended
in linear RGB. Requested masks and source-selected boundary coordinates were
unchanged. Both methods restore the requested-mask exterior exactly in float and
PNG, with requested-background MAE zero. Effective-mask metrics are separate;
feathered effective-background MAE is 0.001250/0.001850 for apple/car. Visual
inspection shows retained red apple rims and blue car strips. These results do
not pass semantic/boundary acceptance. Seven CPU tests pass; no new U-Net or
backward work was used. Four human rows remain unreviewed. Browser verification
did not complete because the computer-use tool could not reliably identify the
current Chrome URL; no browser save/navigation pass is claimed.

On November 18, 2026, the user redirected the main research priority to custom
DDIM editing. Auxiliary pretrained/color/compositing artifacts remain evidence.
[ddim-research-plan.md](ddim-research-plan.md) consolidates all trials, actual
rating status and the next bounded calibration optimization/precision diagnostic.
Reconstruction gate thresholds, human ratings, defaults and holdout state are
unchanged. This route decision does not execute new inference or optimization.

## Executed calibration optimization/precision diagnostic

The subsequent authorized stage A recovered the old calibrated source states by
pivot-free replay and tested fixed current levels741/381/1 on both inputs. The
initial cap20 bitwise-optimization check failed; bounded repeats of the original
helper also differed, despite identical initial losses. That attempt is retained.
A separate deterministic registration passes6/6 exact repeated local controls;
all cap80 first20 losses match cap20. Parameter hashes verify identical values
between FP16 weights and their FP32 upcast.

Cap20 FP16 and FP32 reach tolerance on0/6 problems. Cap80 FP16 reaches it on2/6,
both early levels (apple evaluation47, car66); later fixed problems remain above
tolerance. Effective FP16 updates are not entirely lost to casting. This supports
a bounded full-trajectory budget comparison as the next test, not a precision-only
fix or a source/semantic pass. Sixteen offline tests pass and58 artifacts are
verified. All retained new forward work totals1,074 calls /2,148 batch elements;
backward accounting distinguishes persisted counters from one failed attempt's
inferred count. See [calibration-diagnostic.md](calibration-diagnostic.md) for
tables, scientific plots, precise scope/cost and unchanged gates.

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

## Forward-noising implementation smoke (2026-11-18)

The user requested traditional Img2Img/inpainting/outpainting. Three cached-model
smokes reuse the exposed coffee input (600×400), seed 42, 50 requested steps and
guidance 7.5. Global and masked style targets use strength 0.85, selecting actual
level 821 and 42 reverse updates; right-side outpainting uses strength 1, level 981
and 50 updates, with 128 new pixels (728×400). No inversion/calibration runs.

The inpaint requested-background MAE is 0.04063265 before pixel compositing and 0
afterward across 171,241 pixels. Outpaint source-rectangle MAE is 0.03076215 before
compositing and 0 afterward across 240,000 pixels, with overlap 0 and feather 0.
These exact zeros are compositing results, not acceptance passes. Visible cropped
cup geometry and the outpainting texture seam are retained in the examples.
No new twelve-case or holdout study, human rating, or broad quality claim is made.
See [forward-noising.md](forward-noising.md) and the
[portable smoke examples](results/forward-noising/README.md).

## Task workspace regression checks

The approved interface redesign changes task presentation and state/provenance
handling, with no sampling or model change. The complete offline + opt-in HTTP
suite passed **339 tests, 24 skipped**, with 19 existing Gradio/Starlette HTTP-422
deprecation warnings in 60.17 s. Cached-weight/GPU integration checks were not
enabled for this UI change. The 26 new workspace checks cover route separation,
exact floor/ceil display, source/paint fingerprints, explicit fitting/alignment,
canvas preview parity, fixed boundary crops, archived run associations, upload
preflight, failed-run retention and guarded paint-reference synchronization.
After the last canvas synchronization correction, the focused UI/input/legacy
HTTP suite passed **112 tests, 3 skipped**, with 19 existing deprecation warnings
in 46.30 s. The browser also confirmed a completed painted-mask run accurately
matched its current settings after a programmatic source reset.

Browser validation uses a separate synthetic CPU service on port 7864, never a
real acceptance table. Its artifacts/screenshots are UI/lifecycle evidence only;
see `workspace-ui.md`. Prior real-model examples and all unsuccessful semantic
outcomes remain preserved; no metric or semantic improvement is claimed.

## FLUX Gradio reference verification

The user-directed FLUX page is verified separately from the custom DDIM studies.
Offline/HTTP checks use synthetic workers; browser runs use the pinned local
distilled Klein 4B protected INT8 export. The checks cover real generation,
coffee-image instruction editing, explicit reference preparation, progress,
PNG/JSON/ZIP downloads, stale-result labels, failed-request retention and paint
preservation across task navigation. See `flux-gradio.md` and
`results/flux-ui/browser-observations.json` for evidence and limitations.

The coffee edit is historical regression material, not an unseen holdout case.
It changes both the cup and saucer and uses a requested square output; it does
not establish exact background/structure preservation or pass semantic acceptance.
No previous inversion, calibration, ablation or unsuccessful editing evidence is
replaced by these UI checks.

## Publication regression verification (2026-11-21)

Before the logical Git publication, the complete offline and opt-in local HTTP
suite (`python -m pytest -q --run-demo`) passed **363 tests, 24 skipped** in
63.94 seconds, with 22 existing Gradio/Starlette HTTP-422 deprecation warnings.
Real-weight/GPU checks were not enabled or rerun. Existing browser/model evidence
above retains its original scope and execution records.

The local root `models/` cache is excluded from Git. A byte-preserving portable
[FLUX evidence archive](results/flux-klein-int8/README.md) retains the original
source/quantization manifests, accepted and rejected smoke outputs, and incomplete
attempt records. Packaging these files does not constitute a new experiment.
