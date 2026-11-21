# Demo navigation

The current interface has four task entries: **Generate**, **Edit image**,
**Extend canvas** and **Diagnostics**. Edit image combines whole-image and selected-
region editing, with forward noising as its default method and DDIM inversion
under Advanced settings. See [workspace-ui.md](workspace-ui.md) for the current
layout, state/provenance contract and validation. The seven existing named run
APIs and their upload/fitting callbacks retain their parameter/output order in
hidden compatibility forms backed by the same service/cache/lock.

The sections below preserve historical layout and execution evidence. Their
seven-tab descriptions and result-reset rules apply to the compatibility forms,
not the new task workspace.

## Historical seven-tab layout

Before the task-first redesign, the main demo had seven persistent tabs. The original four replaced the visible
operation radio and separate txt2img accordion; three forward-noising tabs were
subsequently added at the user's request:

| Tab | Inputs and behavior | Run API |
| --- | --- | --- |
| Text to image | Generation/negative prompts, dimensions, steps, CFG and seed; no original/mask | `/txt2img` |
| Local edit | Its own original, upload/draw mask, prompts and editing controls | `/run` |
| Global edit | Its own original/prompts and editing controls; no drawing canvas | `/run_global` |
| Reconstruction | Its own original/source prompt; VAE/DDIM output choices | `/run_reconstruction` |
| Img2Img (noise) | Image, target prompt, floor-based strength and random-noise seed; no inversion | `/forward_img2img` |
| Inpainting (noise) | Image, upload/draw mask, target prompt and independent latent/pixel blending | `/forward_inpaint` |
| Outpainting (noise) | Image, margins, fill, overlap and target prompt; generates an extended canvas | `/forward_outpaint` |

Switching tabs retains each tab's uploaded image, prompts, parameters, selected
outputs, results and download links. Uploading/clearing an image, changing automatic
fitting or changing output choices clears that tab's results as before. Loading the coffee example resets
only that tab and keeps its operation. A hidden operation field preserves the
existing Local edit API signature; there is no visible operation radio to change.
Legacy `/run`, `/upload_image`, `/fit_image` and associated callback names remain
available. Global/reconstruction upload, fitting and callback APIs use matching
`_global`/`_reconstruction` suffixes.

The editor form is built by one shared UI helper, with independent Gradio state
and namespaced element IDs per tab. CSS for one operation cannot hide another
tab's controls. There are two paint canvases, in Local edit and Inpainting (noise).
Local edit is selected
initially so the canvas can initialize at a visible size; `render_children=True`
keeps all tab children mounted across switching. Changing this initialization
policy requires another paint/switch browser check.

All tabs share one `DemoService`, offline model cache, inference lock and queue
concurrency group. Tab selection loads no weights and triggers no inference.
The original generation, inversion editing, calibration and reconstruction math
is unchanged. New noising mathematics lives in its own route.
No model dependency or checkpoint was added. Existing running demo processes must
be restarted to load the revised interface; they were not restarted automatically.

## Original four-tab validation (2026-11-18, historical)

`pytest tests/test_generation.py tests/test_demo.py tests/test_demo_tabs.py --run-demo -q`
completed with **43 passed, 2 skipped, 15 warnings** in 39.06 s. Warnings are the
existing Gradio/Starlette HTTP-422 deprecation. Two pretrained/GPU checks were
not enabled. Tests verify the four labels/default selection, persistent mounting,
one canvas, unique element IDs, hidden operation fields, per-tab API routing,
independent output artifacts and one shared backend load. Existing image fitting,
mask upload/drawing, output-selection, txt2img and download checks pass.

A separate in-app-browser session used a temporary loopback server with synthetic
CPU components. It visited all four tabs and completed each operation. Different
prompt markers, sampling settings, completed images and download links were
retained in their own tabs. A local stroke survived switching through all other
tabs; its brush toolbar remained present, another stroke could be added, and local
execution completed using the retained drawing. No browser console errors were
captured in the local-execution check. The temporary service was stopped after
validation; no production/research server was restarted.

![Four-tab interface during CPU UI validation](results/demo-tabs/tab-layout.jpg)

The screenshot shows the Text to image controls and synthetic CPU test output,
not a real-model generation-quality example. This is an interface/lifecycle
smoke, not an acceptance rating, reconstruction/calibration gain or GPU benchmark.

## Forward-noising extension validation (2026-11-18)

The complete suite passes 337 tests with all HTTP/CUDA/model flags enabled.
New HTTP checks exercise all three APIs, uploaded/painted masks, downloads and
the shared cache. A real-model browser session drew an inpainting stroke, switched
to outpainting and back with its drawing retained, completed painted-mask editing,
then generated an expanded 728×400 canvas with rendered results and download links.
See [forward-noising.md](forward-noising.md). New tabs intentionally validate exact
input/canvas sizes rather than silently applying the legacy image-fitting policy.
