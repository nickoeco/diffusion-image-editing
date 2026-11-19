# Workspace review and demo eligibility

Date: November 18, 2026 (America/Los_Angeles).
Baseline: `cd33682` on `main`, equal to the remote before this review.

## Scope and decisions

This review organizes existing acceptance/review tooling and research reports.
It adds no editing method, performs no GPU experiment, changes no live rating
table, and leaves the four-tab inference app and its defaults intact. No new
dependency or model download is needed.

| Work | Retention decision | Demo decision |
| --- | --- | --- |
| Acceptance registry, blank table and three independent ratings | Keep the protocol and tested offline tools | Supporting evaluation workflow, not an automatic semantic score |
| Fixed source-mask boundary crops and original/raw/final panels | Keep, including the labelled historical renderer example | Suitable for a later explicit evidence output; not connected to the main app here |
| Version-checked review store and literal prompt annotations | Keep as reusable model-free utilities | Available to the standalone review app; preserve output/run associations and raw versions |
| `demo/review_app.py` | Keep as an optional development review demo | Separate loopback app on port 7861; do not merge into the main four tabs yet |
| High-CFG, null-text and precision/budget findings | Keep reports and failed results without relaxing gates | No installed calibration mode: the source gate is still open; prototypes are output-local |
| InstructPix2Pix, RGB constraints and composite studies | Preserve labelled auxiliary references | No new default editing mode; custom DDIM remains the active route |

The review app supports two development candidates per case plus a saved baseline.
It is not a general run browser, holdout runner or freeze system. Browser
save/navigation/stale-write verification on a copied table remains pending.
Construction/callback checks do not close that gap. Keep the app explicitly opt-in.

For a future boundary output in the main demo, use the selected run's original,
requested mask and raw/final images at matching source coordinates. Verify
automatic-fit/padding restoration before placing crop boxes. Preserve selective
computation, make additional artifact generation explicit, reuse source-only crop
rules, show literal prompts and identify pixel compositing. That integration
requires its own UI/browser check and is not implemented by this review.

See [review-ui.md](review-ui.md) for the standalone app and
[ddim-research-plan.md](ddim-research-plan.md) and
[calibration-diagnostic.md](calibration-diagnostic.md) for research gates. This
organizational work does not start full-trajectory calibration or attention editing.

## Repository and local evidence boundary

Commit reusable source, tests, the canonical blank registry/table, English reports
and the small historical renderer preview. Keep model caches/weights, null tensors,
raw local experiment trees and live rating histories in their existing ignored
paths. Do not copy human scores into the blank template or delete auxiliary trials.

Output-local numerical prototypes and their diagnostic tests are not promoted
into the installed package. Historical measurements do not imply a clean checkout
can rerun those experiments from these commits. Reports identify local artifacts.

## Verification performed

- Offline suite: **285 passed, 28 opt-in checks skipped**, 16.95 s.
- Focused review suite with `--run-demo`: **11 passed**, 5.40 s. Construction,
  save/navigation callbacks and persistence used temporary synthetic data.
- Canonical table: twelve `not_run` rows, blank scores, no reviewer/time fields.
  Existing human tables were not edited.
- Source formatting and local documentation links are checked before commit.
- No GPU/model run, browser end-to-end test or human rating was executed.

## Logical commit groups

1. Acceptance protocol, registration/table, boundary tooling, tests and historical
   renderer preview.
2. Optional development review demo, versioned persistence, prompt-figure tools,
   tests and interface documentation.
3. Research/reference findings, route/gates, repository instructions, documentation
   links and this assessment.

Push together after checking the remote. No force push or history rewrite is part
of this work.
