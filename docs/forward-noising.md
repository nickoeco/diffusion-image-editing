# Forward-noising Img2Img, inpainting and outpainting

Date: November 18, 2026 (America/Los_Angeles).

The user requested this implementation after the custom-inversion diagnostic.
It is an independent inference route, not a new inversion candidate, calibration
result or attention editor. Earlier code, gates, failed diagnostics and reference
artifacts are retained. No new dependencies, training or pretrained weights are
introduced. All runs use the existing cached SD 1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.

## Equations and conventions

VAE posterior mode produces scaled z0 from RGB BCHW [0,1]. A local CPU generator
draws one FP32 unit Gaussian epsilon with the same latent shape; this does not
alter the global RNG. With the actual selected training alpha products:

```text
N = requested schedule levels
k = floor(strength * N)
t_start = kth ascending level, with k=0 at CLEAN=-1 (alpha=1)
z_start = sqrt(alpha[t_start])*z0 + sqrt(1-alpha[t_start])*epsilon
```

Strength endpoints are explicit: 0 performs no reverse updates; 1 uses all N
updates from the highest selected level. Positive strength below 1/N also yields
zero updates. At 50 steps, 0.85 selects k=42 and ID821; 1 selects k=50 and ID981.
The existing inversion editor still uses its independent ceil policy and stored
inversion states. This route does **not** invert or optimize null embeddings.
Posterior mode is intentional; it does not reproduce a stochastic VAE posterior
sample used by some Img2Img implementations.

Reverse updates and CFG are the existing custom eta=0 equations. Mask 1 edits,
0 preserves. After t->s, including the final clean update, inpainting computes:

```text
known_s = sqrt(alpha[s])*z0 + sqrt(1-alpha[s])*epsilon
z_s = M_latent*z_edit_s + (1-M_latent)*known_s
```

One epsilon couples all known-region forward marginals. These are not independently
resampled states or a simulated Markov forward trajectory. The known clean state
is exactly z0. No source pivot is substituted into this route: the noised input
is an explicit masked-sampling constraint, distinct from calibration's pivot-free
replay requirement. Only clean/noise and required current states are retained;
full histories are optional for Python API diagnostics.

The pretrained U-Net has four latent channels. This is masked sampling with a base
model, not a specialized nine-channel inpainting network. Existing binary/soft
mask validation, latent resizing and optional Gaussian feathering are reused.
VAE spatial coupling means latent blending alone cannot preserve exact pixels.
Final `M_pixel*decoded+(1-M_pixel)*original` compositing is independent, enabled
by default for masked paths. Both raw and composited images are retained. With
zero selected steps, decoding/compositing still occurs; this does not generally
give pixel identity within editable regions.

## Outpainting canvas

Margins are nonnegative source-pixel integers (left/right/top/bottom), with at
least one positive side. The source is pasted at `(left,top)` without resizing or
cropping. `edge` fill replicates its boundary colors; `neutral` fills RGB0.5.
The entire prepared canvas is VAE-encoded and forward-noised.

New exterior pixels have mask weight1. Default overlap0 keeps the original source
rectangle at mask0. Optional overlap gives a linear editable band *inside* only
the source edges with expansion: weight1 at the edge, falling to0 at the stated
depth. Overlap must be strictly less than half the smaller source dimension.
Feathering independently expands support. With overlap0, feather0 and compositing
enabled, all original source pixels are preserved exactly. Turning off compositing
or permitting overlap/feathering changes that guarantee.

The core pipeline uses the VAE's configured spatial factor and channel count.
Demo/CLI target the cached SD 1.5 model: final dimensions must be divisible by8,
at most1024 per side, source sides at least8. No silent image fitting is applied
in the new forward-noising tabs. Legacy inversion forms retain their fitting controls.
No reference image exists for the new borders: initialized-canvas PSNR/SSIM are
labelled preservation diagnostics, not outpainting quality or seam scores.

## Usage

Start the shared offline model demo from the repository root:

```powershell
.\.venv\Scripts\python.exe -m diffusion_editor.demo.app --port 7862
```

Open `http://127.0.0.1:7862` and choose **Img2Img (noise)**,
**Inpainting (noise)** or **Outpainting (noise)**. All existing named APIs remain
compatible. Inpainting accepts aligned grayscale uploads or paint-layer alpha.
Each form retains independent state; all share one model cache and inference
lock. Outpainting exposes numeric margins, initialization and editable overlap.
No weights load on navigation. Source changes clear that form's masks/results.

CLI examples are in the README. The same service is available through Python:

```python
from diffusion_editor.pipelines.forward_editing import ForwardEditingPipeline

pipeline = ForwardEditingPipeline(backend)  # existing SDBackend
global_edit = pipeline.img2img(image, "a pencil drawing", strength=0.75, seed=42)
local_edit = pipeline.inpaint(image, mask, "a ceramic vase", strength=0.85,
                              latent_blending=True, preserve_background=True)
extended = pipeline.outpaint(image, "a wider view of the same scene",
                             left=64, right=128, overlap=0, strength=1)
```

`/forward_img2img`, `/forward_inpaint` and `/forward_outpaint` are independent
Gradio APIs. UI/CLI save unique directories with original, initialized canvas,
VAE, raw and final images, effective/requested masks, float `masks.pt`, `noise.pt`,
grid, strict JSON/CSV metrics and `run.zip`. Reports record actual IDs/alpha,
settings/prompts, seed/noise hash, model revision, environment/device/dtype,
source-box/margins, timings and artifact hashes. Explicit random noise is retained
for replay; seeds do not promise cross-hardware bitwise equivalence.

## Executed checks

| Check | Observed result |
| --- | --- |
| Offline default suite | 307 passed, 30 skipped, 9.59 s |
| All HTTP/CUDA/model flags enabled | 337 passed, 28 warnings, 126.79 s |
| Focused noising + tab + real-model suite | 26 passed, 8 warnings, 45.25 s |
| CLI help | Passed without loading a model |
| Browser | Painted inpainting, switch/back persistence, progress, rendered results and outpainting verified |

Warnings are9 Transformers tokenizer-default FutureWarnings and19 existing
Gradio/Starlette HTTP-422 deprecations. No test is passed by suppressing a failure.
Unit tests cover equation coefficients/boundaries, FP32 arithmetic, floor strength,
seed/RNG isolation, guidance0/1/>1, no inversion, zero/full/soft/partial masks,
same-epsilon known states at every actual ID including clean, independent switches,
canvas placement, overlap, invalid inputs and shared-cache/artifact routing.
HTTP tests include uploaded and painted masks and PNG/ZIP retrieval. A separate
64×64 cached-weight GPU test executes all three modes. Old math, reconstruction,
generation, review and UI regression checks remain in the full suite.

Three larger historical-coffee smokes use seed42, 50 steps and guidance7.5.
The photo is an already exposed CC0 example, not a new acceptance/holdout input.

| Mode | Size | Strength / start ID / reverse steps | Pipeline compute |
| --- | --- | --- | --- |
| Img2Img | 600×400 | 0.85 / 821 / 42 | 4.196 s |
| Inpaint | 600×400 | 0.85 / 821 / 42 | 3.776 s |
| Outpaint, right128 | 728×400 | 1 / 981 / 50 | 5.316 s |

Inpainting's raw requested-background MAE was0.04063265 (171,241 pixels); final
compositing gave0. Outpainting's source-rectangle raw MAE was0.03076215
(240,000 pixels); final MAE was0 and SSIM1. These zeros establish the implemented
pixel constraint, not a semantic or boundary pass. Runtime excludes loading,
metric calculation and saving; it is descriptive, without matched-cost claims.
Raw files are indexed by `outputs/forward-editing/examples.json`; a portable
[example bundle](results/forward-noising/README.md) preserves the observations.

The real browser also completed a painted-mask edit and a right128 extension.
In that session the latter reported3.77 s and retained the source rectangle.
The [browser screenshot](results/forward-noising/browser-outpaint.jpg) shows actual
UI output, not a mock. Earlier services on other ports were preserved; this
feature demo uses its own loopback port7862.

## Remaining limitations

The masked cup example changes geometry and shows a cut boundary. Outpainting
generates wood-like texture but changes grain orientation and introduces a visible
board/texture seam. All outputs are retained. More context, another mask, overlap
or feathering may change results but have no verified quality guarantee here.
High strength can destroy source semantics. Base-model inpainting has no learned
mask-aware conditioning; no quality equivalence to a specialized checkpoint is
claimed. Zero-mask preservation cannot override these failures.

No human ratings were created, no holdout outputs were generated, no calibration
threshold was relaxed and no attention gate was advanced. The existing independent
semantic/preservation/boundary acceptance protocol still applies. This is a
working implementation with executed smoke/regression evidence, not a claim of
twelve-case semantic acceptance or production-ready image quality.
