# FLUX Gradio reference backend

The **FLUX** task page uses the locally prepared distilled
`black-forest-labs/FLUX.2-klein-4B` checkpoint at revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. The protected export has 100 INT8
Transformer linear layers and 252 INT8 Qwen3 linear layers. Sensitive conditioning
and projection groups, other unquantized floating-point layers and the VAE retain
BF16. Sampling uses four steps, guidance 1 and a local seeded generator. Original
Base weights and the rejected all-linear INT8 export remain available. See
[quantization evidence](flux-klein-int8.md).

## Launch and use

Keep the existing SD1.5 environment unchanged. FLUX requires a separate compatible
Python runtime; the currently available adjacent FLUX environment can be selected:

```powershell
.\.venv\Scripts\python.exe -m diffusion_editor.demo.app --port 7865 `
  --flux-python '..\FLUX\.venv\Scripts\python.exe'
```

Open `http://127.0.0.1:7865` and choose **FLUX**. The runtime can also be selected
with `DIFFUSION_EDITOR_FLUX_PYTHON`; `--flux-config` selects the local FLUX JSON
configuration. No model is loaded on navigation. Models are offline-only and
must have been prepared before launching a job. The UI never exports/updates
checkpoints or installs dependencies.

- **Generate:** enter a prompt, select output dimensions and seed, then run.
- **Edit image:** upload a reference or load the coffee example, describe the
  change, select output dimensions and run. This is instruction editing of the
  whole image; exact unrelated-pixel preservation is not guaranteed.
- **Fit reference to model:** optional aspect-preserving shrink/right-bottom
  edge padding to multiples of 16, at most 1024 pixels per side. The preview and
  source/content/canvas dimensions are visible before inference. Fitting defaults
  off; the explicitly labelled coffee-example action enables it. Its 600x400
  source becomes 608x400, with no content resizing and eight right-edge pixels.
- Download the output PNG, UI JSON or ZIP. The ZIP retains the request, worker
  report/log, prepared reference and oriented RGB original when fitting changes
  geometry. The selected output size is independent of reference size.

Masks, DDIM strength, negative prompts, source reconstruction and canvas margins
remain on the existing SD pages. They are not presented as supported FLUX options.

## Runtime and state boundaries

All routes share the existing inference concurrency group and `DemoService`
lock. Before a FLUX job, the SD backend/editor/reconstructor cache is released
and its allocator cache cleared. The worker uses the separate FLUX runtime,
computes prompt embeddings with INT8 Qwen3, releases that model, then loads the
INT8 Transformer. Workers exit after each request; repeated requests include
startup/loading overhead. Subsequent SD tasks reload their own backend lazily.

Structured worker progress is displayed for startup, text encoding, Transformer
loading and each sampling step. A bounded supervisor terminates only its own
worker on timeout. Errors retain raw logs/failure records and the previous
completed image/downloads. Changes to inputs or controls mark prior results as
stale without relabelling their prompt, run ID, hash or settings.

## Verification

Focused offline and opt-in HTTP checks passed **52 tests**, with six existing
Gradio/Starlette HTTP-422 deprecation warnings. They validate input constraints,
unused-image handling, literal prompt transport, explicit fitting/original-pixel
retention, revision/protection checks, SD cache release/reload, mutual exclusion,
worker progress/timeout/failure handling, output integrity, downloadable archives,
FLUX routing and legacy route compatibility. Synthetic checks do not prove model
compatibility.

Real local browser verification uses a fresh headless Chrome profile and the
prepared local weights. Evidence is saved under `docs/results/flux-ui/` and full
run artifacts under `outputs/flux-ui-verification/flux/`. The browser checks
generation, coffee-image editing, rendered progress/output, literal fitting sizes,
actual downloads, stale labels, invalid-prompt retention, FLUX tab result
retention and the existing painted SD canvas before/after navigation. The test
does not modify acceptance tables or fabricate review ratings.

The final browser check also ran cached real SD1.5 generation before and after
the FLUX jobs. Both SD reports identify the pinned SD revision and the latter
reports `model_first_load=true`, confirming cache eviction and subsequent reload.
The browser recorded all five FLUX progress stages, no JavaScript page errors,
512x512 rendered outputs with opacity 1, working downloads and retained paint.
Generation/edit UI wall times in that run were 26.7/21.7 seconds respectively;
they include per-request startup and loading and are not benchmark comparisons.

To reproduce the optional browser checks, install Playwright into a separate
test-tool target and use an existing Chrome executable. No browser download is
required:

```powershell
.\.venv\Scripts\python.exe -m pip install --target .cache/flux-ui-tools playwright
.\.venv\Scripts\python.exe -B scripts/verify_flux_ui.py --run-demo --run-gpu --check-sd-switch
```

`--probe` replaces `--run-gpu` for a model-free UI inspection. The optional
`--check-sd-switch` check also runs tiny cached real SD generations before/after
FLUX to verify backend eviction/reload. HTTP checks require `--run-demo`; browser
GPU checks require `--run-gpu`. The automation blocks non-local page requests.

These are compatibility and UI checks at 512x512, not a general memory/performance
benchmark. Larger images/reference batches have not been validated here. In the
coffee regression, both cup and saucer become blue and framing differs at the
selected square output size. This is not an exact-preservation claim or a pass
of the twelve-case semantic acceptance protocol.
