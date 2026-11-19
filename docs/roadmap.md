# Development roadmap

Demo maintenance, November 18: explicit automatic resizing/padding and aligned
mask transforms are implemented at the UI/service boundary. See
[image fitting](demo-image-fit.md) for validation and the pending browser check.
This maintenance does not advance a research phase or change its gates.

| Phase | Working deliverable | Gate |
| --- | --- | --- |
| 0 | Environment, package, config/logs, component loader, smoke scripts | Offline tests, installation, honest integration status; stop for user |
| 1 | VAE and prompt wrappers | Shape/device tests; saved VAE reconstruction baseline |
| 2 | Schedule and approximate DDIM inversion | Tested equations, boundaries, deterministic trajectory with explicit timestep IDs |
| 3 | Custom reverse DDIM + CFG + reconstruction | Unit tests, actual same-prompt reconstruction, PSNR/SSIM |
| 4 | Global target-prompt editing | Stored-level strength selection; reproducible simple edits |
| 5 | Local mask editing | Synthetic mask tests, aligned levels, measured background preservation |
| 6 | Gradio demo | Upload/mask/parameters/outputs/download/timing/errors calling custom pipeline |
| 7 | Reproducible evaluation and baseline | At least two ablations executed, metrics, outputs, plots and limitations |

Every phase follows explain -> implement -> test -> review -> summarize. Do not
implement later-phase placeholders as if functional. Phase 1 was authorized and
started on 2026-11-15; it adds VAE/text wrappers, offline tensor tests and explicit
GPU/real-weight checks. Phase 2 adds the shared schedule, explicit clean boundary,
custom inversion, equation/reference tests and a real 50-step inversion smoke.
Phase 3 adds manual CFG, shared eta=0 arithmetic, reverse sampling, same-prompt
reconstruction, PSNR/SSIM, labelled previews and measured smoke results.
Phase 4 adds discrete stored-level strength selection, partial reverse sampling,
source/target global editing, matched same-prompt controls and real smoke results.
Reconstruction drift was addressed for the verified guidance-1 setting, with
adverse-case fallback and documented high-guidance limits. The user subsequently
authorized Phase 5: aligned masks, updated-ID source blending, feathering,
independent pixel compositing and measured background preservation.
Phase 5 was committed and pushed as `3bb8253` and `1329640` before Phase 6 began.
Phase 6 adds the local Gradio app, alpha-based drawing, cached serial inference,
downloadable images/reports and explicit HTTP/real-weight checks. Phase 6 was
committed and pushed as `bb55617` and `9d5adff` before Phase 7 began.
Phase 7 adds strict one-variable plans, serial cached-model runs, frozen inputs,
resume/integrity validation, measured budgets, a labelled Img2Img baseline and
automatic tables/grids/plots. See `phase7-status.md` for executed verification and
`reconstruction-drift.md` and the earlier phase status records for
executed verification and AGENTS.md for binding constraints and test acceptance details.

Phase 7 completed 34 custom runs across five studies and four matched-reverse-grid
Img2Img baseline cases, with saved CSV/JSON, grids, PNG/SVG curves and a portable
review bundle. The full explicit HTTP/CUDA/model test suite passed. The core phase
gates have recorded evidence; semantic reliability, dataset size, high-guidance
fidelity and geometry preservation remain limitations. No unrequested stretch
phase has started. Phase 7 implementation and tests are committed as `2e9af60`,
followed by the separate measured-results and documentation commit.

The next user-directed step establishes a semantic acceptance protocol before
further method tuning: twelve independent task slots, balanced development/holdout
splits, independent three-dimension ratings, boundary zooms and a per-case table.
`acceptance.md` defines the freeze rule. Assets are pending audit and no new
holdout/model evaluation has begun. This does not extend the earlier two-image
results into a general editing-success claim.

A later isolated working copy assembled the twelve inputs and ran six development
cases, receiving human verdicts of 1 pass, 2 partial and 3 fail. The next controlled
development round generated twelve outputs across two one-variable profiles;
these require fresh human ratings. Per-row reviewer/time fields were removed by
user request. Holdout execution and method freeze remain pending; none of these
local trials establishes a completed semantic acceptance gate.

The local development comparison now has a dedicated model-free human review
interface with direct score/reason input, draft/completed saves, revision checks
and versioned raw tables. Comparison figures and the UI include exact source and
target prompts. This improves evidence review; it does not establish a semantic
pass, select a parameter profile, freeze the method or authorize holdout inference.

The editing demo follow-up adds explicit custom-image upload, mode-specific mask
visibility and independently selected VAE/DDIM/edit/control outputs. Edit-only is
the default, with unselected computation skipped. Existing pipeline defaults and
semantic acceptance tooling remain intact. See `demo-controls.md` for focused
compute-budget, HTTP and browser validation; these UI changes make no semantic
acceptance claim.

An explicitly requested advanced reconstruction extension now audits the high-CFG
source path before attention-based editing. The ordinary/matched-guidance and
existing residual controls fail the new engineering screen on two development
inputs. Output-local null-text calibration with frozen weights substantially
improves source reconstruction and has exact batch-matched independent replay.
The latent-error/local-convergence gate remains open; the next deliverable is a
separately registered convergence/precision study, followed by attention editing
only after that gate is resolved. Defaults and holdout status are unchanged.
See [advanced-reconstruction.md](advanced-reconstruction.md).

The subsequent authorized [instruction-reference study](instruction-reference.md)
completed two real InstructPix2Pix recolorings with remaining background/material
changes. It establishes an alternative editing route for these inputs without
closing the custom reconstruction gate or passing semantic acceptance. A separate
fixed-output localization study then generated hard and inward-feather composites.
Both restore requested-mask exterior pixels exactly, while red apple rims and blue
car strips remain. Seven CPU tests pass; four new human rows remain unreviewed and
browser verification did not complete. Broader development/holdout validation
still requires the declared registration, freeze and independent review rules.

## Active research priority: custom DDIM

On November 18, 2026, the user selected the custom DDIM route as the main research
direction again. The installed editor already uses it; pretrained instruction
editing and explicit color postprocessing remain preserved auxiliary references.
The authoritative [evidence review and plan](ddim-research-plan.md) specifies:

1. Register a bounded calibration optimization/precision diagnostic, beginning
   with fixed local levels and gradient/update evidence rather than another
   strength/guidance sweep.
2. Use its evidence for one controlled source reconstruction improvement, keeping
   existing gates, calibrated execution layout and independent pivot-free replay.
3. After the source gate passes, register minimal custom attention control with
   no-op equivalence, token alignment and matched text-swap controls.
4. Validate localization and all six development cases; freeze before six holdout
   cases and independent semantic/preservation/boundary ratings.

This decision does not execute future stages or promote numerical prototypes into
production. Failed gates stop progression; thresholds are not relaxed. Auxiliary
reference polishing is no longer the active deliverable.

Stage A has since completed with recorded real GPU probes. Two source replays
match prior PNGs exactly; an initial optimization-repeatability failure is
retained and a separately registered deterministic correction passes6/6 repeated
local controls. FP16 cap80 reaches tolerance on2/6 problems; FP16/FP32 cap20
reach0/6. Sixteen offline tests pass and58 artifacts are verified. See
[calibration-diagnostic.md](calibration-diagnostic.md). Stage B now requires a
separate full-trajectory cap20/cap80 registration using actual propagated states.
No new full-trajectory candidate or attention editor has been executed.

The user separately requested a txt2img demo interface. A custom-sampler generation
pipeline, independent panel and `/txt2img` API now support prompt-only generation
and downloads. CPU and HTTP contracts passed; at that initial delivery the
pretrained/GPU smoke and completed browser check were unexecuted. This UI utility does not advance or replace the
registered full-trajectory calibration comparison or any acceptance gate. See
`txt2img-demo.md`; the pre-existing demo process must be restarted to load the UI.

A subsequent user request separates the UI into four persistent operation tabs.
Independent form state, scoped controls and one shared backend keep switching
simple without losing prior work. CPU/HTTP checks passed 43 tests; an in-app
browser confirmed all modes, result retention and paint/switch lifecycle behavior.
Real-model/GPU generation remains unexecuted for these UI changes. See
`demo-tabs.md`; no calibration or acceptance gate was advanced.
