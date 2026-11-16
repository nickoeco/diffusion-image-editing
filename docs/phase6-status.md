# Phase 6 status: local Gradio demo

Date: 2026-11-15. Phase 5 was logically committed and pushed first:
`3bb8253` adds mask-guided editing; `1329640` records tests, masks and evidence.
The user then authorized Phase 6. Phase 7 had not started at that phase's gate.

## Purpose, inputs and outputs

Provide a local interface to the existing custom pipelines, with uploaded or
painted masks, editing/reconstruction controls, useful errors and downloadable
results. This phase adds no replacement diffusion algorithm or new model weights.
Gradio 5.49.1 is an optional `demo` extra; its tested transitive versions are
recorded in `configs/constraints-windows-py311-cu124.txt`.

Inputs are an EXIF-oriented RGB image, source/target prompts, steps, strength,
guidance, seed, and optional aligned grayscale upload or painted RGBA layers.
Image sides are bounded to 8–1024 pixels and must satisfy the model's VAE spatial
factor; no resizing/cropping occurs. Paint-layer alpha union defines the mask,
independently of paint color or the photographic composite. Canvas backgrounds
are checked against the original to reject stale drawings.

Outputs separate original, VAE baseline, optional full DDIM reconstruction,
matched source-prompt editing control, final edit and effective mask. Optional
previews decode stored latents at explicit IDs; they are not clean predictions.
Each run saves a unique directory and ZIP with PNGs, original/entered masks,
exact float masks, a grid, JSON report and CSV metrics. Loading and pipeline
compute times are separate; compute excludes metrics, previews and file saving.

## Architecture and operation

- `demo/app.py`: Gradio controls, queue, errors and loopback-only launch entrypoint.
- `demo/inputs.py`: original-image normalization, canvas reset and mask extraction.
- `demo/service.py`: immutable request validation, lazy cached backend, serial
  inference, calls to our pipelines, metrics and isolated artifacts.
- `tests/test_demo.py`: input/service unit tests, explicit local HTTP checks and
  explicit cached-weight GPU service check. `--run-demo` enables socket checks;
  `--run-integration --run-gpu` enables the real model check.

App construction loads no weights. The service requires `local_files_only=true`,
loads the existing pinned cache on first valid use and reuses components thereafter.
A lock and the inference queue serialize GPU work. Full source reconstruction
uses guidance 1/1 and optional existing residual correction, independently of edit
guidance. Editing uses plain inversion. Both latent blending and final pixel
compositing remain explicit independent settings.

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,evaluation,demo,dev]"
.\.venv\Scripts\python.exe -m diffusion_editor.demo.app
```

The server listens on `http://127.0.0.1:7860`, with no public share link. Uploads
are limited to 20 MB. The output root is explicitly permitted for downloads.
Ctrl+C stops the process. The local demo has no authentication or remote-deployment
hardening and should remain on loopback. Cache weights using the already documented
explicit download workflow before launch; this UI never downloads them.

## Executed verification

On Windows/Python 3.11.9, RTX 4080 Laptop GPU, CUDA 12.4, with the pinned SD 1.5
revision and existing local cache:

| Check | Result |
| --- | --- |
| `python -m pip check` | No broken requirements |
| `python -m pytest -q` | 207 passed, 22 skipped, 3.77 s |
| `python -m pytest -q --run-demo --run-gpu --run-integration` | 229 passed, 9 warnings, 80.00 s |
| Local HTTP upload + draw + PNG/ZIP downloads + empty-paint error | Passed using CPU synthetic backend |
| Real cached-model UI service, 64×64 local edit, 4 steps | Passed; exact zero-mask background |
| Browser-painted 600×400 image, 50 steps, full independent reconstruction | Completed; images, timing and download links visibly rendered |

The 22 default skips are three CUDA-only tensor tests, eighteen integration checks
(one also CUDA-marked) and one HTTP demo check. All were explicitly enabled and
passed in the full run.
Six warnings are the existing Transformers tokenizer-default FutureWarning;
three are Gradio's use of a deprecated Starlette HTTP-422 constant. No warning was
suppressed or mistaken for a failed check.

Unit tests cover alpha union, ignored photographic composite, empty/stale/RGB/
misaligned layers and uploads, invalid requests, validation before loading,
backend reuse, all operation modes, independent reconstruction, masks, preview
IDs, unique artifacts, strict JSON, CSV inclusion and revision/seed recording.
HTTP checks exercise actual Gradio preprocessing and serialization with
`gradio_client`, including RGBA drawing payloads, file retrieval and actionable
error propagation. Mocks establish these contracts, not pretrained compatibility.

Browser testing caught a front-end issue that the HTTP tests could not: an
initially hidden ImageEditor raced its resize/background initialization, leaving
the result display stale despite completed model computation. The final UI sets
image controls explicitly interactive and keeps the paint canvas mounted and
visible across mask/operation changes. The original problematic run is retained
under `outputs/demo/31f650b916ab4ac49e006a4664cfbd49/`; it is not counted as a
successful browser result. The repaired browser flow showed paint tools, accepted
a stroke, displayed inference progress, then rendered outputs and download links.

## Measured browser run

Successful artifacts: `outputs/demo/4d0cef5917ee43e9978205bac8f61ff6/`, including
`comparison.png`, `report.json`, `metrics.csv`, `mask_tensors.pt` and `run.zip`.
The input is the existing CC0 coffee photograph. The drawn diagonal stroke has
15,008 edit pixels and 224,992 background pixels. Source/target prompts are the
coffee-photo/pencil-drawing prompts saved in the report. Settings: seed 42,
50 steps, strength 1, start ID 981, reverse guidance 7.5, inversion guidance 1,
hard mask, no feathering, latent blending and final pixel compositing enabled.
Full reconstruction uses 12 residual evaluations per level and guidance 1/1.

| Measurement | Value |
| --- | --- |
| First model loading | 7.535 s |
| Combined edit + independent reconstruction compute | 40.138 s |
| Peak CUDA allocation | 3,286,412,288 bytes (3.061 GiB) |
| VAE PSNR / SSIM vs original | 25.656064 dB / 0.7398561 |
| Full DDIM reconstruction PSNR / SSIM vs original | 25.656006 dB / 0.7398565 |
| Uncomposited edit background MAE / PSNR | 0.0334535 / 25.1093 dB |
| Final zero-mask background MAE | 0 (224,992 pixels) |
| Selected reconstruction method | residual_refined_ddim |
| Refined / plain clean-latent roundtrip MSE | 0.00000334379 / 0.1005837 |

The full reconstruction is close to its VAE baseline under these settings.
Refinement did not meet the strict 1e-6 local tolerance at any level; the warning
and per-level diagnostics are retained, and selection uses measured final latent
MSE. This is not a claim of exact inversion. Final background equality comes from
pixel compositing. The narrow stroke produced a local appearance/geometry change,
not a convincing pencil drawing; this browser run verifies UI wiring and
preservation, not semantic quality. Whole-image edited/control scores are retained
but are dominated by the preserved background.

## Review and remaining work

The UI changes no timestep indexing, DDIM/CFG equations, tensor/device policy or
matched-level blending. Model components are cached once, and trajectories are
retained only as required by inversion or explicit previews. No pipeline shortcut
is used. Input and model errors are logged and surfaced instead of swallowed.

CPU full-model performance was not benchmarked. Very large supported images or
high step counts can still be slow or exceed available memory. Full reconstruction
adds work, feathering expands support, tiny strokes may vanish at latent resolution,
and strong edits can distort geometry. Color-edit reliability and general
high-guidance reconstruction limitations remain. No general editing-quality gain,
formal ablation completion or full project completion is claimed.

Phase 6 implementation and tests were committed as `bb55617`, followed by the
documentation/evidence commit, before the user-authorized Phase 7 work began.
The executed evidence above remains the Phase 6 gate; Phase 7 has a separate report.
