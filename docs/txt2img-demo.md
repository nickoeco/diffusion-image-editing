# Text-to-image demo interface

The main demo includes a dedicated **Text to image** tab and a
`/txt2img` Gradio API. It needs no uploaded original or mask. It uses the cached
Stable Diffusion components with our `DDIMSampler` and explicit CFG; no built-in
text-to-image/editing pipeline, new dependency or checkpoint is introduced.

Restart your existing demo process to load the new interface, then open its local
URL (default `http://127.0.0.1:7860`). This change does not automatically restart
another running server or alter research/acceptance artifacts.

```powershell
.\.venv\Scripts\python.exe -m diffusion_editor.demo.app
```

Select the Text to image tab, enter a generation prompt and optional negative prompt,
choose width/height, steps, guidance and seed, then click **Generate image**.
The generation controls are independent of editing and reconstruction controls.
Download the PNG or the ZIP containing the generated PNG and JSON report.

- UI dimensions are 64–1024 in increments of 64; the API accepts integer multiples
  of 8 in that range. The pipeline also validates the actual VAE spatial factor.
  No upload fitting, padding or cropping is involved.
- Steps are 1–200. Around 512×512 and 50 steps are reasonable initial UI settings;
  very low step counts can produce poor output, and large sizes may exceed VRAM.
- Guidance uses the existing explicit unconditional/conditional CFG implementation.
  Seed controls FP32 Gaussian noise from a local CPU generator; editing's original
  image, mask and inversion strength are not generation inputs.
- Model components and the inference lock/queue are shared with editing. Requests
  run serially and reuse the lazy offline cache. The UI never downloads weights.

## API

With the demo running and the optional demo dependencies installed:

```python
from gradio_client import Client

image_path, status, png_path, zip_path, report = Client(
    "http://127.0.0.1:7860"
).predict(
    "a red apple on a wooden table",  # prompt
    "",                             # negative prompt
    512, 512,                       # width, height
    50, 7.5, 42,                    # steps, guidance, seed
    api_name="/txt2img",
)
```

The request is validated before model loading. Output directories are unique
under `outputs/demo/<run-id>/`, containing `generated.png`, `report.json` and
`run.zip`. Reports retain the prompt/negative prompt, seed, dimensions, model
revision, actual state IDs including clean -1, scaled initial-noise hash and
standard deviation, runtime/device/dtype, software environment and output hash.
Model loading and pipeline compute times are separate; compute excludes saving.

## Numerical scope

`pipelines/generation.py::TextToImagePipeline` encodes the text, draws a Gaussian
prior scaled by the scheduler's `init_noise_sigma`, places it at the highest
actual selected schedule level, traverses our eta=0 updates/CFG and decodes once
with the FP32 VAE. The prior is an approximation at that selected level; it is not
claimed to be an inverted source latent or an exact terminal distribution. This
generation path does not invoke inversion, residual correction, null-text
calibration, source pivots or mask/pixel blending. Fixed seeds fix initial CPU
noise; bitwise image reproducibility across GPU/runtime versions is not promised.

There is no original reference, so PSNR, SSIM and background MAE are explicitly
not applicable (`metrics: null`). Generated images do not demonstrate editing
acceptance, reconstruction calibration gains or a passed attention-editing gate.

## Verification and limitations

On 2026-11-18, focused offline generation and existing demo tests passed:
**39 passed, 4 skipped**. Explicit local HTTP checks passed:
**41 passed, 2 skipped**, with 12 existing Gradio/Starlette deprecation warnings.
These checks used synthetic CPU components, no pretrained weights and no GPU.
They cover analytic constant-noise DDIM/CFG at scales 0/1/>1, model-configured
latent shape/scaling, local-seed repeatability/RNG isolation, invalid dimensions,
absence of source encoding/inversion, shared cache reuse, unique files, actual
Gradio serialization, PNG/ZIP downloads and actionable invalid-prompt errors.

Native Chrome verification was attempted against an isolated temporary CPU
validation server. Browser automation stopped because its current URL could not
be identified reliably; no completed browser generation or visual layout check is
claimed for that initial attempt. The temporary server was stopped. Existing demo
servers were not restarted. A later successful browser check of the four-tab
interface is recorded in [the tab layout report](demo-tabs.md); its generated
outputs also used synthetic CPU components, not pretrained weights.

A cached-weight GPU smoke is registered but **was not executed** for this side
request, to avoid competing with ongoing research work:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_generation.py --run-demo --run-integration --run-gpu -q
```

Synthetic tests do not prove pretrained generation quality or compatibility.
No real-model image, semantic-success score or new acceptance result is claimed.

## Tabbed operation layout

The demo now has Text to image, Local edit, Global edit and Reconstruction tabs.
Each owns its inputs, settings, results and downloads while sharing the cached
backend and inference queue. Select another tab to switch without clearing work.
The local tab opens initially to initialize its drawing canvas, and all tab
children remain mounted. See [tab behavior and API names](demo-tabs.md).
