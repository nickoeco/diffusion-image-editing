# Editing acceptance protocol v1

Status: **canonical registry has 12 assets-pending task slots; local development
trials are recorded below; method not frozen; no holdout evaluation executed.** This protocol separates
editing success from reconstruction and preservation measurements. The earlier
Phase 7 results remain valid historical evidence but are not this acceptance set:
34 runs on two images do not constitute 12 independent cases.

## Dataset and independence

Use [the registered plan](../configs/acceptance/cases.yaml): four recoloring,
four object-replacement and four style-conversion cases. Each category has two
development and two holdout cases, giving **6 development + 6 holdout** cases.
Tasks include material edges, occlusions, thin structures and contact shadows.
All edits are localized; the allowed target region and protected content are
explicit for every case. Style conversion applies to the named region, not the
entire background. Replacement permits target-object shape change within a
predefined allowed region; it does not require the original object silhouette.

Each slot must receive a different original photograph. Reusing a photograph
with a different prompt, crop, mask, seed or augmentation is not independence.
Near-duplicates, the same capture session or source family cannot cross splits;
for this small initial set use a different source group for every case. The old
coffee and astronaut examples and their derivatives are already development
material and must not enter the new holdout set. Record provenance, usage rights,
original-file and oriented-RGB hashes, source group, dimensions and an aligned
requested mask. Check exact and visual near-duplicates before accepting assets.
Null assets in the plan are placeholders, not usable or evaluated samples.

Before tuning, fix each input, source/target prompt, mask, allowed shape/lighting
changes, protected regions, and case-specific criteria. Replacement masks must
provide room for new geometry such as a spout or rabbit ears; they cannot be
enlarged after seeing a failed result. Annotate a fixed requested region and
evaluate expanded feather support separately. Keep sufficient scene context to
judge preservation and lighting. Assets must be compatible with the configured
VAE spatial factor; do not silently resize during comparisons.

## Three independent ratings

For the local two-candidate development trials, use the [human review interface](review-ui.md)
to select ratings and type reasons beside the evidence. CSV is retained as the
storage/export format. Comparisons include the literal source and target prompts
from generation reports; the interface performs no automatic visual scoring.

Rate each dimension **0 = fail, 1 = partial, 2 = pass**. Blank means unreviewed,
never zero or a pass. Use the case-specific text in the manifest as the primary
checklist; record concrete observations and evidence paths for every rating.

| Dimension | 2: pass | 1: partial | 0: fail |
| --- | --- | --- | --- |
| Semantic success | Requested color, object or style is clearly realized throughout the intended region | Recognizable but incomplete, mixed, weak or ambiguous target | Target absent, incorrect or contradicted by residual original content |
| Unrelated-content preservation | Position, background and all protected structures remain consistent at normal view and in the registered details | Small localized drift without a major protected-content change | Moved/missing/duplicated objects, changed layout, identity loss when protected, or substantial background/structure changes |
| Boundary quality | No visible unintended truncation, double edges, halos, ghosting or abrupt lighting/texture seams in the registered views | Small localized artifact visible on detail inspection | Obvious clipping, disconnected structures, duplicate contour, ghost object, or conspicuous lighting/texture seam |

Semantic checks are category-specific:

- **Recoloring:** the target hue clearly dominates the intended material surface.
  Preserve plausible highlights, shading and texture; do not count the entire
  scene acquiring a color cast as a successful localized color change.
- **Replacement:** the new object is recognizable, singular and structurally
  coherent; the old object must not remain as a ghost. Placement/scale constraints
  are preserved while explicitly permitted target geometry may differ.
- **Style:** require the named style's registered visible traits. A grayscale
  filter alone does not establish graphite hatching, and a color shift alone does
  not establish watercolor or oil brushwork. Preserve the registered object
  identity, pose and connectivity when these are protected.

An intentional medium transition in a localized style edit is not automatically
a seam failure; judge unintended clipping, halos, incoherent texture or lighting
against the declared target. For unedited regions, original photographs are the
reference. VAE-only reconstruction helps identify codec loss but does not excuse
a visible acceptance failure. Exact pixels achieved by compositing do not excuse
an unsuccessful semantic change or a bad boundary.

## Verdicts and reporting

A case passes only with **(semantic, preservation, boundary) = (2,2,2)**. A complete
rating containing a 0 fails; other complete ratings are partial. Missing ratings
remain unreviewed, missing execution remains not_run, and execution errors remain
errors. Averages cannot compensate for a failed dimension. The strict initial
suite passes only if all six holdout cases pass and the freeze/evidence records
are complete. This is a declared acceptance target, not a claim the current
method meets it.

Report pass/partial/fail/unreviewed/not_run/error counts over the **fixed six**
holdout cases, with per-category denominators **two**. Also report each dimension
separately. Do not drop failures, replace difficult cases after inspecting output,
choose the best seed, or select a successful checkpoint post hoc. Report each
method/version separately with paired case results and actual total compute.
The sample is small: give counts and observations without significance or broad
generalization claims. Improvement requires better paired semantic/preservation/
boundary judgments and disclosed trade-offs; higher PSNR alone is insufficient.

Use [the blank 12-row review table](acceptance-review-template.csv). Preserve raw
ratings, run/freeze IDs, evidence, metric-report path, output
hash and failure tags. Suggested tags: wrong_color, wrong_object, weak_style,
residual_source, geometry_drift, background_change, truncation, ghosting, halo,
lighting_seam, texture_seam and execution_error. The table intentionally omits
per-row reviewer identity and review-time fields. Traceability uses versioned raw
ratings, run IDs and output hashes. If multiple independent assessments are made,
retain separate tables and resolve disagreements explicitly. Never silently
replace a raw rating with an adjudicated one.

## Development, freeze and holdout workflow

1. Assemble and audit all 12 independent assets. Finish eligibility checks and
   allocation before tuning. Dataset assembly may inspect input photos/masks to
   verify eligibility; it must not use holdout model outputs to adjust tasks.
2. Tune only on the six development cases. Log every trial, including unsuccessful
   outputs. Keep a declared rule for choosing one method/parameter profile from
   development results. If profiles vary by category, declare that mapping now;
   do not introduce per-holdout-case overrides.
3. Write an immutable freeze record **before generating holdout outputs**: dataset
   manifest and file hashes; split/source groups; rubric/version; prompts/masks;
   method/code identity; model ID/revision; all parameters, seeds, precision and
   compositing/feather switches; execution budget and selection rule; boundary
   anchor/crop rules; and the final development selection rationale. Hash and date
   the record. This draft plan is explicitly not a valid freeze record.
4. Execute the six holdout cases once under that frozen selection. Preserve every
   output/error. A retry is allowed only for a logged infrastructure failure with
   identical frozen inputs/settings, retaining the failed attempt. No tuning based
   on holdout success, pixel metrics or boundary screenshots is allowed.
5. Review all three dimensions, then attach existing preservation metrics and
   publish the complete per-case table. Review semantics before looking at metric
   rankings to reduce preservation-score anchoring. Compare methods with labelled
   matched cases and disclosed budgets.
6. A method change after holdout inspection makes those images exposed regression
   material. Further development can use them only with that label; a fresh unseen
   holdout set is needed for a new confirmation claim.

This protocol is binding workflow guidance, not an inference access-control
system. The existing Phase 7 runner is unchanged and does not enforce split/freeze
rules. The new plan is deliberately a separate acceptance registry, not an
executable Phase 7 sweep. No holdout run is authorized by a blank review template.

## Required evidence and boundary zooms

Retain original, requested mask, effective feather mask, VAE reconstruction, matched
source control, raw edit before compositing, final edit, prompts/settings, seed,
model revision, runtime, actual state IDs and compute budget. Continue saving
floating-tensor **PSNR, SSIM and background MAE** with their existing definitions.
Evaluate background against the same requested mask across methods; provide
effective-mask metrics separately. Never score masked perceptual similarity by
zeroing the rest of an image. No new automatic semantic score is claimed here.

For every case include full-frame original/raw/final comparisons, the mask and a
boundary locator, plus corresponding enlarged crops. The offline helper selects
top/right/bottom/left contour anchors from the **source requested mask only**,
centers ties toward the contour centroid, removes duplicate boxes and clips boxes
to image bounds. Defaults: 64×64 source pixels, 4× nearest-neighbor enlargement.
It never selects attractive regions based on output content and never smooths
away halos. Freeze the mask and crop rule so all candidate methods share boxes.
Pre-register extra case-specific critical-detail boxes if necessary; the four
automatic anchors alone cannot guarantee coverage of every seam or thin structure.
Pass pre-registered source-pixel centers with repeated `--extra-center X Y`; keep
the same centers across all compared methods. These coordinates are saved in the
review JSON and must be part of the freeze record.
Inspect the full contour at 100% in addition to the fixed panels, and log additional
diagnostic views rather than replacing registered views after seeing results.

```powershell
# Create a fresh blank table; refuses to overwrite an existing review:
.\.venv\Scripts\python.exe scripts/prepare_acceptance_review.py template --output outputs/acceptance/review.csv
# Render evidence from an already-recorded run; no inference or auto-grading:
.\.venv\Scripts\python.exe scripts/prepare_acceptance_review.py panels --original path/to/original.png --mask path/to/requested_mask.png --edited path/to/final.png --raw path/to/raw.png --output-dir outputs/acceptance/case-id/run-id
```

The helper saves `crop_locations.png`, `requested_mask.png`, `boundary_panel.png`
and `boundary_review.json` with exact boxes, zoom and image hashes. It enforces
alignment and a nonempty partial grayscale mask, and refuses to overwrite evidence.
Its panels do not replace manual ratings. Import metric values from the recorded
floating-tensor run; do not recompute acceptance metrics from enlarged crops or
JPEG previews. Historical images may demonstrate the renderer only and must not
be counted as new development or unseen holdout cases.

## Implementation verification (2026-11-16)

The offline regression suite passed **234 tests**, with **23 opt-in GPU/model/HTTP
checks skipped** (13.08 s). The nine focused acceptance tests also passed after
adding image dimensions to the renderer metadata. Tests cover balanced registration,
blank pending rows, independent score verdicts, duplicate declared sources, aligned
deterministic crops, nearest-neighbor zooms, extra detail centers and invalid-input/
overwrite rejection. No GPU/model/HTTP test was executed for this offline-only
addition, and no new semantic result is claimed.

A visually checked [historical renderer example](results/acceptance-tooling/README.md)
shows the locator and original/raw/final crops. It demonstrates evidence rendering
only. New-case asset acquisition/audit, development trials, a genuine freeze record,
and holdout execution/review were pending at initial registration, in that order.

## Review-field simplification and local development follow-up

At the user's request, current rating tables and the generator omit `reviewer`
and `reviewed_at`. Raw submitted scores/reasons are preserved in a versioned
snapshot; run IDs and output hashes retain the evidence association. No reviewer
identity or review date is required to proceed with development.

An isolated local working copy under
`outputs/acceptance/2026-11-16-side-validation/` assembled twelve audited real-photo
inputs and masks. Its first six development outputs received human scores:
one pass, two partial and three fail. All preservation and boundary ratings were
2, while the reported semantic failures were unchanged color, insufficient style
and a floating replacement vase. The canonical registry remains separate from
this local working copy.

Two pre-registered development-only profiles subsequently completed twelve
additional outputs on those same six sources: strength 1 at guidance 7.5, and
guidance 12 at strength 0.75. Each changes one baseline parameter; inputs, prompts,
masks, seed, remaining settings and boundary anchors are held fixed. The passed
orange case remains a regression control. These outputs require fresh human
ratings; baseline ratings are not copied to them. No improved acceptance rate is
claimed from pixel metrics or assistant observations. The holdout has no model
outputs and no method is frozen.

The field change passed the nine focused acceptance tests and the offline suite
(234 passed, 23 opt-in checks skipped, 6.47 s). Local trial verification checks
parameter deltas, input/output hashes, matched crop boxes and holdout exclusion;
it is distinct from a human semantic verdict. No diffusion algorithm, model
dependency or pretrained weight was changed.
