# Automatic demo image fitting

Date: 2026-11-18. Scope: the Gradio editing demo and its service boundary.
Research inputs, calibration execution, acceptance assets/ratings and core
pipeline size contracts are unchanged. No dependency or weight download is added.

## Behavior and coordinate spaces

**Automatically resize and pad** defaults on. The demo EXIF-orients the upload
and converts it to RGB before calculating the fit:

1. If the longest side exceeds 1024, reduce both sides proportionally using
   Lanczos. Each content dimension uses integer half-up rounding, with a minimum
   of one pixel. Smaller images are not enlarged.
2. Round both content dimensions up to multiples of 8 by padding the right and
   bottom. Padding replicates the last content row/column; content is not cropped.
3. Show upload, resized-content and model-canvas sizes in **Image sizing**.
   Model outputs retain the fitted canvas. They are not cropped or upscaled back
   to the uploaded dimensions.

| Upload (width × height) | Resized content | Canvas/output |
| --- | --- | --- |
| 512 × 768 | 512 × 768 | 512 × 768 |
| 513 × 769 | 513 × 769 | 520 × 776 |
| 1501 × 1001 | 1024 × 683 | 1024 × 688 |
| 17 × 13 | 17 × 13 | 24 × 16 |
| 1 × 2 | 1 × 2 | 8 × 8 |

The 1024 limit is a demo input bound, not a GPU memory guarantee. Around 512 per
side remains recommended. The upload file-size limit remains 20 MB.

Uploaded grayscale masks may match the oriented original upload, resized content
or shown canvas. Original-size masks share the image's resize dimensions, using
nearest interpolation for binary mode or bilinear for soft mode. Canvas masks are
restricted to the content rectangle; entered mask padding is always zero. Other
dimensions are rejected rather than stretched. Binary thresholding, feathering,
latent blending and final pixel compositing retain their existing semantics.
Feathering can expand the effective mask into padded pixels, and exact background
pixel preservation still requires final pixel compositing.

Drawing uses alpha from layers aligned with the shown canvas. Direct service
calls may also supply a drawing on the original upload. Both require an exact
matching background. Stale backgrounds and padding-only painting are rejected.
An explicitly uploaded all-zero mask remains valid.

Changing the image or the fitting switch clears uploaded/painted masks, outputs
and download links. A per-session `PreparedUpload` retains the oriented original,
fit plan and pixel hashes. Reusing a preview preserves original provenance;
toggling fitting off restores original pixels, and toggling on fits those pixels
once. An unrelated image cannot reuse stale source state. Disabling fitting
restores strict size errors; it does not resize or pad. Operation/output changes
retain the current geometry and usable mounted canvas.

## Implementation and artifacts

`demo/image_fit.py` defines `ImageFit`, `PreparedUpload`, `prepare_upload` and
`resolve_fitted_mask`. The app prepares previews and stores source state;
`DemoService.run` applies the same fitting for direct API inputs and checks the
loaded VAE's actual spatial factor. The eight-pixel alignment is the supported
SD 1.5 demo policy. VAE wrappers, custom DDIM pipelines and experiment runners
remain strict and do not implicitly fit their images.

Reports include the `auto_fit` request, original/content/canvas dimensions,
padding, interpolation, coordinate-space metadata and source/canvas pixel hashes.
`original.png` is the fitted model input. A transformed upload is additionally
saved as `uploaded_original.png`. `input_mask.png` is the aligned entered mask;
`mask_before_fit.png` retains a mask whose pixels or geometry changed. These files
are included in the run ZIP. Effective masks and existing exact mask tensors are
still saved independently.

`metric_domain` explicitly names the fitted model canvas including padding.
PSNR/SSIM and regional denominators therefore include padded pixels. These are
demo preservation diagnostics, not original-resolution or semantic-quality
claims, and are not comparable to previous runs with different fit geometry.

The `/run` API appends an `auto_fit` boolean after the existing visible arguments;
`/upload_image` takes file, operation, selected outputs and fitting switch.
`/fit_image` takes current image, operation, selected outputs and fitting switch.
The source state is server-side and belongs to the client's Gradio session.

## Executed verification

- Offline suite: **272 passed, 24 skipped**, 5.31 s. Geometry tests cover
  unchanged content, replicated edges, portrait/landscape fitting, tiny/extreme
  aspect ratios, EXIF orientation, mask spaces/interpolation, stale backgrounds,
  padding-only paint and source-state reuse. Service tests cover fitting,
  preserved background pixels, retained raw assets and report geometry.
  A pre-push check excluding the separately uncommitted acceptance/review tests
  passed **253 tests, 23 skipped**, 6.82 s. The larger workspace total above
  includes those unrelated tests and is not the committed demo suite's size.
- Explicit demo HTTP suite: **27 passed, 1 skipped**, 14.72 s. The real-model case
  needs separate opt-ins. Ten existing Gradio/Starlette HTTP-422 deprecation
  warnings were retained. HTTP checks verify upload/draw/run behavior and a
  session round trip: 17 × 13 upload → 24 × 16 preview/output → disable fitting
  restores 17 × 13 → re-enable fits the original again. The report retains the
  17 × 13 source when inference receives the fitted preview.
- Cached-weight CUDA check: **1 passed**, 7.26 s, with the existing tokenizer
  FutureWarning. The 64 × 64 local-edit smoke uses six U-Net forwards. A separate
  57 × 49 VAE-only request produces a 64 × 56 output with zero U-Net forwards.
  Its measured VAE compute time is 0.0364 s (warm backend); this is a small-input
  smoke measurement, not a performance benchmark. No full-resolution editing
  or research experiment was run for this follow-up.

Real artifacts are retained under
`outputs/demo-size-validation/real-check/test_real_demo_service0/`:

- `9268b6d01b7944de829c5d061d9d558d/`: local-edit smoke.
- `9bc5bda8c03d432ba8c5e9a6043e7c26/`: odd-size VAE-only smoke, including raw and
  padded originals, reconstruction, report, metrics and archive.

**Browser verification remains pending.** Computer Use stopped before navigating
to the demo because it could not confidently determine the current Windows
browser URL for policy enforcement. No browser painting, progress or rendered
output claim is made for this size-fitting change. API checks do not establish
front-end correctness. An isolated updated service was started at
`http://127.0.0.1:7863`; existing services were not replaced. Restart older demo
processes to load the updated code.

Remaining browser checks: upload a large odd-size image and inspect the fitted
canvas; paint on it after switching operations; toggle fitting off/on and confirm
mask reset and restored raw dimensions; run VAE-only on a smaller odd-size image
and inspect progress, fitted output dimensions and download links. Use synthetic
validation images, not acceptance assets or real human ratings.
