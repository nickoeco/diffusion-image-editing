# Selective editing demo controls

Date: 2026-11-16. This follow-up changes the editing demo, without changing the
acceptance review interface or recorded human ratings.

## Behavior

- **Operation** chooses Local edit, Global edit or Reconstruct source. Global and
  reconstruction modes hide the mask panel. Local mode also hides it when only
  reconstruction outputs are selected, because no mask is consumed.
- **Outputs to calculate** independently selects Edited image, VAE reconstruction,
  DDIM reconstruction and Source-prompt control where applicable. Editing defaults
  to Edited image only; Reconstruct source permits VAE/DDIM and defaults to DDIM.
  Switching operation resets the selection. At least one output is required.
- **Upload custom image** opens an explicit image chooser, EXIF-orients the image
  and converts to RGB. It replaces the original, clears uploaded/painted masks and
  clears old outputs/download links. The coffee example restores its own image,
  prompts and mask. Prompts for custom images remain user-editable.
- Model components are still lazily cached and inference stays serialized. VAE-only
  avoids all text/diffusion execution, although first use still loads the shared
  backend. The original November 16 controls retained strict dimensions. The
  [November 18 image-fitting follow-up](demo-image-fit.md) adds explicit automatic
  resizing/padding in the demo; the core pipeline remains strict.

The canvas remains mounted. Scoped CSS hides inactive controls, preserving the
initialized ImageEditor across operation changes. This avoids the prior lifecycle
problem in which remounting could remove paint tools. Reconstruction and target
settings are hidden when they do not apply to selected outputs. Uploading a new
image always resets painting, including while the mask panel is hidden.

## Actual computation

| Selected output | Work |
| --- | --- |
| VAE only | Posterior mode encode + one VAE decode; no U-Net or CLIP execution |
| Edited image only | Source inversion + target reverse path + final decode |
| Source control only | Source inversion + source-prompt reverse path + final decode |
| DDIM reconstruction only | Independent source reconstruction, optional residual correction/fallback, final decode |
| Combined outputs | Required branches only; a requested VAE baseline is not decoded twice across edit and DDIM runs |

For positive strength, inversion still spans all configured levels; reverse
selection uses the existing stored-level mapping. The source-control branch is
skipped when unselected. VAE baseline decoding is skipped unless requested or
needed as the final strength-zero output. Local latent blending and final pixel
compositing retain their existing independent switches and updated-level behavior.
Unselected outputs are None, omitted from metrics/archives and do not trigger
sampling merely to populate a hidden component. `computed_outputs` and nullable
source/target/reconstruction IDs document the executed selection.

Editing API switches `include_vae_reconstruction`, `include_source_control` and
`include_edit` default to true for backwards-compatible experiment/CLI behavior.
DDIM reconstruction's new VAE-baseline switch also defaults to true. The demo
explicitly requests its selected outputs, with Edit-only defaults. The DDIM residual
correction option affects the independent reconstruction only, not target editing.

## Verification

Focused offline checks verify actual U-Net and decoder calls rather than display
visibility alone. With the three-level synthetic schedule and strength 0.5:

| Selection | Observed U-Net forwards | VAE decodes |
| --- | --- | --- |
| Edit only | 5 (3 inversion + 2 target) | 1 |
| VAE only | 0 | 1 |
| Plain DDIM only | 6 (3 inversion + 3 reverse) | 1 |
| Source control only | 5 (3 inversion + 2 source control) | 1 |
| All four outputs, plain DDIM | 13 | 4 (one shared baseline and three requested branches) |

The focused tests also compare Edit-only pixels against the existing all-output
path, exercise zero-mask control preservation, validate empty/invalid output
selections, check no hidden PNGs are archived, and retain existing default-pipeline
regressions. HTTP checks exercise actual Gradio preprocessing for upload, draw,
selection/mode callbacks, selected-only run archives and useful errors.

Browser verification on the isolated loopback instance at port 7862 checked:
Global and Reconstruct hide masks; returning to Local restores paint tools;
painting works after mode changes; a custom astronaut PNG replaces the example
while masks are hidden; old mask/painting is cleared; the new image can be painted
after returning to Local; and VAE-only produces just the selected result with
download links. The real 512×512 VAE run displayed 0.61 s compute and 7.01 s first
model loading. No DDIM or target edit was requested in that browser run.

Observed regression results:

- Default offline suite: **250 passed, 24 skipped**, 5.52 s. Existing opt-in
  socket/CUDA/model checks remain skipped by default.
- Demo HTTP suite with `--run-demo`: **25 passed, 1 skipped**, 11.09 s; six
  Gradio/Starlette HTTP-422 deprecation warnings were retained. The skipped case
  requires the separate real-model/GPU opt-ins.
- Cached-weight demo check with `--run-integration --run-gpu`: **1 passed**, 13.52 s,
  with the existing tokenizer FutureWarning. The four-level partial edit used six
  U-Net forwards and the subsequent VAE-only request used zero. Other GPU suites
  were not rerun for this follow-up; no full-GPU-suite result is claimed here.

The browser artifact is
`outputs/demo-controls-validation/5d5a33139de341e6b2380b21034e72b4/`.
Its `computed_outputs` contains only VAE reconstruction, metrics contain only
the VAE stage, and source/target/DDIM timestep fields are all null. This run did
not modify any semantic acceptance inputs, review table or human rating.

All project documentation remains English. No model download or dependency change
is introduced. Existing servers are not stopped by this change; restart an older
process to load the new code. Validation uses a separate port/output directory to
avoid replacing an existing editing or acceptance-review session.
