# Diffusion-Based Image Editing

A Computer Vision course project using pretrained Stable Diffusion 1.5 with custom
DDIM inversion, reverse sampling, Classifier-Free Guidance (CFG), and mask-guided
latent blending. **Phase 7 adds reproducible ablations, automatic grids/plots and
a separately labelled Diffusers Img2Img baseline, alongside the local Gradio UI.**
All future changes must
follow [AGENTS.md](AGENTS.md).

Project documentation is maintained in English. Workflow communication may use
English, Chinese, or both.

For the current two-candidate development acceptance comparison, the separate
[human review interface](docs/review-ui.md) provides direct score/reason entry,
image enlargement, matched boundary zooms and exact source/target prompts. CSV
remains an export format; no model loading is needed for review.

Reconstruction defaults to budgeted residual-refined DDIM with an end-to-end latent-error
selection check. Plain DDIM remains available for teaching and controlled baselines.
See [the drift investigation](docs/reconstruction-drift.md) for measured evidence,
the computational trade-off and known failure cases.

The opt-in [advanced reconstruction extension](docs/advanced-reconstruction.md)
audits high-CFG fidelity and tests bounded per-image null-text calibration with
frozen model weights. Reconstruction improves on two development inputs, but the
predeclared latent-error/convergence gate remains open. Attention editing and
production-default adoption have not started; source reconstruction is not an
editing-success claim.

The active research direction is the **custom DDIM editor**. The
[evidence review and next-stage plan](docs/ddim-research-plan.md) summarizes all
tested approaches, unresolved reconstruction/editability gates and the bounded
deterministic full-trajectory budget comparison that comes next. Pretrained instruction editing
and direct color postprocessing remain auxiliary references.

The [calibration diagnostic](docs/calibration-diagnostic.md) has now completed:
deterministic repeated local controls pass, a larger budget helps early steps,
and FP32 execution alone does not resolve local convergence. The next experiment
is a separately registered full-trajectory budget comparison. Source and semantic
gates remain unmet; production defaults are unchanged.

## Installation (PowerShell, repository root)

Python 3.11 is recommended; supported versions are 3.10–3.12. Development and model
dependencies are installed separately:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
python scripts/check_environment.py --output outputs/environment.json
```

The current offline suite uses real CPU PyTorch tensors and small synthetic models;
it requires no GPU, pretrained weights, Diffusers, or Transformers. The `test` extra
installs tensor/image dependencies and scikit-image for metric tests. The original Phase 0 tests needed only the
lighter `dev` extra. Installing the project metadata alone does not verify the full
model dependency stack. CUDA and real-weight tests require explicit opt-in flags.

The model environment uses a pinned torch/torchvision pair and established library
versions. For NVIDIA GPU execution, install the CUDA wheels first instead of the
CPU wheel above, then install the complete model stack:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --output outputs/environment-model.json
```

For CPU environments, replace `cu124` with `cpu`. CPU execution is recommended for
lightweight tests; full model loading and generation are slow and require
substantial RAM. PyTorch CUDA wheels include the runtime libraries, so a separate
full CUDA Toolkit installation is not required for inference. Use the environment
script to check the driver and actual PyTorch CUDA availability.

Pinned versions include torch 2.5.1, torchvision 0.20.1, diffusers 0.35.1,
transformers 4.44.2, accelerate 1.0.1, safetensors 0.6.2, huggingface-hub 0.34.4,
and numpy 1.26.4. This combination passed Phase 1 CUDA component integration tests
on the recorded machine; DDIM/editing verification is recorded in the phase reports.
Direct dependencies are pinned. A [tested Windows/Python 3.11/CUDA 12.4 constraints
snapshot](configs/constraints-windows-py311-cu124.txt) records the installed versions.
Use it with `pip install -c configs/constraints-windows-py311-cu124.txt -r requirements.txt`
after installing the CUDA torch wheels. It is not a cross-platform lockfile.
Phase 3 adds scikit-image 0.25.2 for SSIM, Phase 6 adds optional Gradio 5.49.1,
and Phase 7 adds Matplotlib 3.10.7 to the evaluation extra. OpenCV is not required.

## Model setup and smoke checks

- ID: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- The [official model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  and [access/file metadata](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5)
  were checked on 2026-11-15: the repository was public and ungated, with the
  CreativeML OpenRAIL-M license. Use and distribution must comply with that license.
- The Diffusers-format checkpoint includes CLIP, a VAE, a four-channel U-Net,
  scheduler configuration, and FP16 safetensors. Only epsilon-prediction base
  models are supported; inpainting checkpoints are not compatible substitutes.
- Weight downloads and caching require several GB; reserve approximately 10 GB
  of disk space. An RTX 4080 Laptop GPU with 12 GB VRAM is a starting point for
  512×512 inference with batch size 1 and CUDA FP16. Measure actual peak memory.
- U-Net/CLIP precision follows the runtime policy; the VAE uses FP32. The cache is
  located at `.cache/huggingface`.

The default configuration is strictly offline. These commands require both the
model dependencies and cached weights:

```powershell
.\.venv\Scripts\python.exe scripts/check_model.py --forward
.\.venv\Scripts\python.exe scripts/test_generation.py --steps 20
```

After user confirmation, explicitly add `--allow-download` for the first download.
Do not use that flag in unit tests. `check_model.py` verifies component loading,
text encoding, and one U-Net forward pass. `test_generation.py` is a **built-in
text-to-image reference smoke check**, saving `outputs/generation/reference.png`.
It is not the final editing algorithm and retains the built-in safety checker.
Neither script downloads weights by default; disabling `local_files_only` in the
configuration also permits network access.

## Architecture and future usage

```text
configs/                  Explicit YAML configuration
src/diffusion_editor/
  config.py               Configuration validation
  models/                Component loading, VAE and CLIP wrappers (implemented)
  diffusion/              Schedule, inversion, reverse sampler, manual CFG
  editing/                Strength, binary/soft masks, latent blend, pixel composite
  pipelines/              Reconstruction, global and local source/target editing
  evaluation/             Metrics, strict study plans, batch runner and reports
  demo/                   Local Gradio UI, validated inputs and cached-model service
  utils/                  Environment diagnostics and logging
scripts/                  Environment checks, component forward, reference generation
tests/                    Offline CPU unit tests
docs/                     Architecture, mathematical contracts, roadmap, experiment records
assets/examples/          Example inputs and masks with recorded provenance
outputs/                  Runtime artifacts excluded from version control
```

Global and local editing are available through `scripts/run_edit.py`.
The Gradio demo and reproducible experiment commands are available below. See the
[roadmap](docs/roadmap.md) for development gates.

## Phase 1: VAE and text conditioning

```python
from PIL import Image
from diffusion_editor.config import load_config
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.models.vae import VAEWrapper
from diffusion_editor.models.text_encoder import TextEncoderWrapper

backend = SDBackend.load(load_config("configs/default.yaml"))  # cached weights
vae = VAEWrapper(backend.vae)
text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
with Image.open("input.png") as image:
    latent = vae.encode_image(image)
reconstruction = vae.decode_latents(latent)  # RGB BCHW [0,1], on VAE device
conditioning = text.encode_prompt_pair(["a red apple", "a blue apple"])
# [2B,L,D]: all negative/unconditional embeddings, then all positive embeddings
```

Image dimensions must be divisible by the configured VAE spatial factor. The
wrapper never silently resizes/crops; latent scaling comes from VAE configuration.
Posterior mode makes repeated encoding deterministic while preserving VAE loss.
Text prompts are padded/truncated to CLIP's token limit with a truncation warning.

```powershell
.\.venv\Scripts\python.exe scripts/check_conditioning.py
.\.venv\Scripts\python.exe scripts/check_conditioning.py --image input.png
.\.venv\Scripts\python.exe -m pytest -q --run-gpu
.\.venv\Scripts\python.exe -m pytest -q --run-integration --run-gpu
```

The first script defaults to a deterministic 512×512 synthetic test card, saving
`original.png`, `vae_reconstruction.png`, labelled `comparison.png`, and configuration,
device/dtype/shape/provenance/timing information in `report.json` under
`outputs/reconstruction/phase1/`. This is a VAE baseline, not a DDIM reconstruction
or a natural-image quality benchmark. Use `--image` to verify a real photograph.
The script defaults to cached weights; downloads require `--allow-download`.
Pytest integration checks always use local cache and never download weights.
Executed checks, numerical caveats, and output paths are recorded in the
[Phase 1 status report](docs/phase1-status.md).

## Phase 2: scheduler and DDIM inversion

```python
from diffusion_editor.diffusion.schedule import DDIMSchedule
from diffusion_editor.diffusion.inversion import DDIMInverter
from diffusion_editor.models.noise_predictor import UNetNoisePredictor

schedule = DDIMSchedule.from_scheduler(backend.scheduler, num_steps=50)
source = text.encode_prompt("a red apple")  # [B,L,D], conditional-only
result = DDIMInverter(schedule, UNetNoisePredictor(backend.unet)).invert(latent, source)
noisy_latent = result.final_latent
source_at_t = result.trajectory[result.timesteps[-1]]
```

The schedule preserves the pretrained training coefficients and actual inference
IDs. The separate clean ID is `-1` with alpha 1; a cloned scheduler explicitly uses
`set_alpha_to_one=True` for this boundary. The original scheduler is unchanged.
Snapshots are FP32 on CPU by default. At the clean state, the first U-Net query
uses the lowest selected training timestep; `-1` is never sent to the network.
Inversion is approximate. Its final state is not necessarily pure Gaussian noise.
Conditional B embeddings support guidance 1. For other scales, provide paired 2B
embeddings from `text.encode_prompt_pair`; inversion and sampling share manual CFG.

```powershell
# Reuse the Phase 1 test card, or pass --image with a valid input image.
.\.venv\Scripts\python.exe scripts/test_inversion.py --steps 50
```

This cached-weight script saves `outputs/inversion/phase2/trajectory.pt` and
`report.json`, including state/query IDs, alpha coefficients, runtime, peak allocated
CUDA memory and environment. Load the locally generated trajectory with
`torch.load(path, weights_only=True)`; integer IDs map to tensors. This performs
no reverse reconstruction. See the [Phase 2 report](docs/phase2-status.md) and
[equation conventions](docs/implementation.md).

## Phase 3: reconstruction and metrics

```python
from diffusion_editor.pipelines.reconstruction import ReconstructionPipeline
from diffusion_editor.evaluation.metrics import reconstruction_metrics

with Image.open("assets/examples/astronaut.png") as image:
    result = ReconstructionPipeline(backend).reconstruct(
        image, "a portrait of an astronaut wearing a spacesuit", num_steps=50,
        guidance_scale=1.0, inversion_guidance_scale=1.0, return_trajectory=True,
        inversion_refinement_steps=12, inversion_refinement_tolerance=1e-6)
metrics = reconstruction_metrics(result.original, result.ddim_reconstruction)
```

The VAE-only and DDIM images are separate outputs. Inversion and sampling share
exactly the same schedule, clean boundary and source prompt. Guidance settings
are independently configurable; start with 1/1. The sampler executes our eta=0
equations and CFG, not Diffusers' pipeline or step method.

```powershell
.\.venv\Scripts\python.exe scripts/test_reconstruction.py --image assets/examples/astronaut.png --source-prompt "a portrait of an astronaut wearing a spacesuit" --save-intermediates --output-dir outputs/reconstruction/phase3-astronaut
```

The image is a public-domain example with [recorded provenance](assets/examples/README.md).
The default `--image` still uses the Phase 1 generated card when that output exists.
The script saves original/VAE/DDIM PNGs, a labelled comparison, `report.json` and
`metrics.csv`. `--save-intermediates` also saves both trajectories and explicitly
labelled decoded-latent previews (not predicted clean images). It is offline and
uses only cached model weights. Quality is measured on floating RGB [0,1] before
uint8 saving; SSIM uses uniform 7×7 windows, channel averaging and sample covariance.
Exact-match PSNR is infinite in the library and encoded as null in strict JSON.
See [Phase 3 results](docs/phase3-status.md) for measured reconstruction drift.

The historical Phase 3 report describes the original approximate inversion.
The current reconstruction default refines each inversion step against the actual
reverse-step residual, up to 12 destination U-Net evaluations, with RMS tolerance
1e-6. It also runs plain inversion/reconstruction and selects the lower VAE-latent
roundtrip MSE; selection and candidate residual diagnostics are saved in the report.
Both candidates are genuinely reverse-sampled from their terminal latents. This
adds computation and does not eliminate the VAE bottleneck's loss.

Use `--refinement-steps 0` (or `inversion_refinement_steps=0` in Python) to reproduce
the original DDIM baseline. `--refinement-tolerance` controls early stopping.
Start with matching inversion/sampling guidance 1/1. A per-step correction is not
a guarantee for every grid or guidance setting, so the reconstruction selection
check remains enabled whenever refinement is requested. The global editing API
keeps its measured plain-DDIM default; `run_edit.py --refinement-steps 12` opts into
source refinement, without the reconstruction-only selection check or an editing
preservation guarantee when target conditioning differs.

## Phase 4: global text-guided editing

```python
from diffusion_editor.pipelines.editing_pipeline import EditingPipeline

with Image.open("assets/examples/astronaut.png") as image:
    result = EditingPipeline(backend).edit_global(
        image,
        source_prompt="a portrait of an astronaut wearing an orange spacesuit",
        target_prompt="a portrait of an astronaut wearing a blue spacesuit",
        strength=0.5, guidance_scale=5.0, num_steps=50,
        inversion_guidance_scale=1.0, return_trajectory=True)
edited = result.edited_image
control = result.source_reconstruction  # Same selected latent and reverse guidance
print(result.selection.start_timestep, result.selection.reverse_steps)
```

For N selected inference levels, strength chooses `k=ceil(strength*N)` and stored
inversion level k. Decimal input boundaries are handled without float-rounding
off-by-one errors. With N=50, strength 0.5 selects timestep 481 and 25 reverse steps;
0.75 selects timestep 741 and 38 steps. This is not linear interpolation of timestep
or noise variance. Strength 0 returns the VAE-only image and skips U-Net/inversion;
it does not promise pixel identity to the original. Strength 1 uses the deepest
stored level. Every positive strength uses a full N-step source inversion before
the source-control and target branches each run k reverse steps.

```powershell
.\.venv\Scripts\python.exe scripts/run_edit.py --strength 0.5 --guidance-scale 5 --save-trajectories
```

The offline script saves original, VAE-only, source-control and target-edit images,
a labelled grid, JSON/CSV preservation metrics and optional explicit-ID trajectories.
It records both prompts, strength, selected timestep/alpha/noise standard deviation,
reverse count, seed, model revision, guidance, runtime and environment. The default
output directory is `outputs/editing/phase4/`. There is no spatial mask or final
pixel composite. PSNR/SSIM indicate preservation, not target-prompt adherence.
The blue-spacesuit smoke did not reliably change the suit color; do not treat this
example command as a successful color-edit guarantee. See the
[Phase 4 report](docs/phase4-status.md) for actual results and limitations.

A verified global appearance example is coffee-photo to pencil drawing:

```powershell
.\.venv\Scripts\python.exe scripts/run_edit.py --image assets/examples/coffee.png --source-prompt "a photo of a white ceramic cup filled with coffee on a wooden table" --target-prompt "a black and white pencil drawing of a cup of coffee, detailed sketch, pencil shading on white paper" --strength 1 --guidance-scale 7.5 --output-dir outputs/editing/phase4-sketch
```

The target branch visibly changed to a pencil style in this run; the matched
source branch remained photographic. Both changed geometry/details relative to
the original. This is global editing, not localized background preservation.

## Phase 5: mask-guided local editing

```python
with Image.open("assets/examples/coffee.png") as image, Image.open("assets/examples/coffee_mask.png") as mask:
    result = EditingPipeline(backend).edit_local(
        image, "a photo of a cup of coffee on a wooden table",
        "a black and white pencil drawing of a cup of coffee", mask,
        strength=1, guidance_scale=7.5, num_steps=50,
        latent_blending=True, preserve_background=True,
        binary_mask=True, feather_radius=0)
local_image = result.edited_image
latent_only_image = result.uncomposited_image
```

Mask 1 edits and 0 preserves. PIL masks must be single-channel L/1; tensors may
be HW or B1HW, floating [0,1] or boolean. A single mask can broadcast over a batch.
Pixel dimensions must match the oriented image exactly; only latent conversion
resizes. Binary masks use threshold >=0.5 and nearest resizing. Soft/feathered
masks use bilinear resizing. Feather radius is Gaussian sigma in image pixels,
truncated at ceil(3*sigma); it expands the effective edit support.

```powershell
.\.venv\Scripts\python.exe scripts/run_edit.py --image assets/examples/coffee.png --mask assets/examples/coffee_mask.png --source-prompt "a photo of a white ceramic cup filled with coffee on a wooden table" --target-prompt "a black and white pencil drawing of a cup of coffee, detailed sketch, pencil shading on white paper" --strength 1 --guidance-scale 7.5 --save-trajectories --output-dir outputs/editing/phase5-hard
```

Use `--soft-mask` for soft input weights, `--feather-radius 4` for feathering,
`--no-latent-blending` and `--no-pixel-composite` to control the two operations
independently. Without `--mask`, the CLI still runs global editing.

Latent blending occurs **after** each update, against the source inversion state
at the updated ID, including clean -1. It also applies to the same-prompt control.
Final compositing applies the effective pixel mask to original pixels separately.
An empty mask with compositing enabled preserves the original exactly; with only
latent blending it returns VAE reconstruction. Full masks match global editing.
The editing default remains plain inversion; optional source refinement is explicit
and does not apply reconstruction's original-latent selection to target edits.

Local output includes `edited_uncomposited.png`, `effective_mask.png`, exact
`mask_tensors.pt`, `preservation_comparison.png`, and background metrics in the
JSON report, alongside the existing images/CSV/optional trajectories. Weighted
background MAE/PSNR use weights 1-M and denominator 3*sum(weights); literal M==0
pixels have a separate exact-preservation MAE. Empty regions are undefined.
With feathering, zero-support preservation refers to the expanded effective mask;
the report also evaluates the original requested mask. See the
[Phase 5 report](docs/phase5-status.md) for measured effects and limitations.

## Phase 6: local Gradio demo

Install the optional interface into the model environment, then launch from the
repository root after caching the pinned model weights:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[model,evaluation,demo,dev]"
.\.venv\Scripts\python.exe -m diffusion_editor.demo.app
```

Open http://127.0.0.1:7860. The app binds only to loopback, creates no public share
link, requires offline configuration and loads cached model components on the
first run. Later runs reuse them; inference is serialized. Stop with Ctrl+C.
Use `--port 7861`, `--config configs/default.yaml` or `--output-dir outputs/demo`
to override launch settings. No weights download automatically.

1. Choose **Generate**, **Edit image**, **Extend canvas** or **Diagnostics**.
   Each task retains independent inputs/results. Edit image defaults to forward
   noising; choose Whole image or Selected region, then enter a target prompt.
2. Upload an image or use the coffee example. Source/canvas dimensions appear
   before painting. Advanced settings provide explicit optional shrinking/padding;
   fitting defaults off. Enabling it clears both masks. It never crops or upscales
   content. Final model dimensions must be divisible by 8 and at most 1024 per side.
3. Selected-region editing uses one visible work surface: Upload displays the
   aligned grayscale mask as a cyan overlay; Draw activates the paint editor.
   Only the active mask is used. Changing the image or fitting clears both masks;
   changing tasks or mask mode retains paint layers and tools.
4. Strength is a basic control. Steps, CFG, seed, negative prompt, mask feathering
   and the independent latent guidance / final pixel compositing switches live
   under Advanced settings. DDIM inversion is an explicit editing-method option
   that reveals the source prompt. Diagnostics calculates only selected VAE,
   DDIM reconstruction or source-prompt controls.
5. Run and inspect **Result**, draggable **Compare** or **Boundary** zoom. Extend
   canvas previews dimensions, source placement, new area and editable overlap
   without resizing the source. Details retain literal prompts, actual requests,
   requested/effective masks and reports. Download PNG or the image/JSON/CSV ZIP.
   Changed settings mark the previous result; failed runs retain successful results.

Artifacts use isolated directories under `outputs/demo/<run-id>/`. The report
separates loading time from pipeline compute time; compute excludes metrics,
preview decoding, image saving and archive creation. PSNR/SSIM measure preservation,
not prompt alignment. Forward noising uses seeded noise; deterministic inversion
adds no random edit noise. Workspace runs also save a `workspace.json` association
manifest in the archive; backend `report.json` remains intact. Data and cached weights stay
on the local server. This demo has no authentication and is intended for local use.

HTTP checks require local sockets and are a separate opt-in from offline tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --run-demo
# Include cached real-model and CUDA checks:
.\.venv\Scripts\python.exe -m pytest -q --run-demo --run-integration --run-gpu
```

See [Phase 6 validation](docs/phase6-status.md) for executed evidence and limitations.
See [selective demo controls](docs/demo-controls.md) for current output semantics,
upload/reset behavior and the follow-up validation.

## Phase 7: reproducible evaluation

The checked-in [experiment plan](configs/experiments/phase7.yaml) contains 34 custom
runs: plain/refined reconstruction at 20/50/100 steps; global strength at
0.25/0.5/0.75/1; guidance at 3/5/7.5/10; and six mask-operation controls. Two public
images are used for reconstruction/global studies; the mask study uses the authored
coffee ellipse. Each study varies one declared axis while keeping other settings
and input pixels fixed. Mask presets provide explicit pairwise controls for latent
blending, feathering and pixel compositing.

```powershell
# Validate and inspect without loading ML libraries or downloading weights:
.\.venv\Scripts\python.exe scripts/run_experiments.py --dry-run
# Offline model execution; the evaluation extra includes Matplotlib:
.\.venv\Scripts\python.exe scripts/run_experiments.py --output-dir outputs/experiments/phase7
# Baseline only: same weights, target prompts, guidance and actual reverse grids:
.\.venv\Scripts\python.exe scripts/run_img2img_baseline.py outputs/experiments/phase7
# Rebuild tables, grids and PNG/SVG plots from verified saved artifacts:
.\.venv\Scripts\python.exe scripts/summarize_experiments.py outputs/experiments/phase7
```

Existing nonempty experiment directories are rejected unless `--resume` is given.
Resume requires the same suite, input hashes, source code, model config, packages
and runtime signature; completed artifacts are verified before reuse. Inputs are
frozen under `inputs/`. Each `runs/<case-id>/` contains original/VAE/output/control
PNGs, masks where applicable, a grid, `report.json` and `metrics.csv`. Reports retain
actual timestep IDs, source refinement diagnostics, selected method, U-Net forward
and batch-element counts, device/dtype, runtime and peak CUDA allocation. Failures
produce `error.json` and propagate; they are not counted as completed runs.

The labelled baseline samples the VAE posterior and adds random Gaussian noise,
then calls Diffusers' Img2Img pipeline with DDIM eta=0. It uses supported latent
input/output so the shared VAE remains FP32. It does not perform our source
inversion, latent blending or source-control branch. At strengths 0.5/1 and 50
steps, baseline and custom reverse grids match exactly; total U-Net work differs
and is reported explicitly. Baseline update precision also follows Diffusers'
inference dtype. Neither runtime superiority nor prompt-alignment improvement is
implied by this comparison. The script loads no new model/safety-checker weights
and is a fixed-input offline evaluation, not a deployment pipeline.

Metric definitions and executed evidence are maintained in
[experiments](docs/experiments.md) and the [Phase 7 report](docs/phase7-status.md).
The fixed **requested mask** provides a common regional denominator across mask
variants; effective-mask scores separately describe expanded feather support.
PSNR/SSIM measure preservation, not whether an edit matches its target prompt.
The two-image, single-seed study does not establish statistical generalization.

Executed evidence: **34 custom cases, four baseline cases and five completed
studies**, with full tests passing. Browse the [portable measured-result bundle](docs/results/phase7/README.md)
for CSV/JSON, vector plots and example grids.

## Semantic editing acceptance

The [acceptance protocol](docs/acceptance.md) registers **12 independent task slots**:
four each for recoloring, object replacement and style conversion, balanced into
six development and six holdout cases. The canonical registry remains an
assets-pending template; isolated local development trials are documented in the
protocol. This is not a twelve-case acceptance pass. Existing Phase 7 images are
historical evidence, not an unseen holdout set.

Each case independently rates semantic success, unrelated-content preservation
and boundary quality. A pass requires 2/2/2 on the registered 0–2 rubric; PSNR,
SSIM and background MAE remain supporting preservation metrics. The method,
parameters, prompts/masks and boundary review rules must be frozen before holdout
outputs. The [blank per-case table](docs/acceptance-review-template.csv) contains
no invented scores and requires no reviewer identity or review-time fields. An
offline helper creates aligned source/raw/final boundary
zooms, without model execution or automatic grading; see the protocol for commands.

## Known limitations

Offline synthetic tests verify tensor contracts and equations; actual pretrained
verification is tracked separately in the phase status records. DDIM inversion is
approximate, and the VAE introduces reconstruction error. Latent masks cannot
guarantee exact pixel preservation; final pixel compositing is independently
configurable. Project completion is governed by the definition of done in AGENTS.md.
The historical plain-DDIM natural-image smoke shows substantial reconstruction
drift. Residual refinement addresses the measured cases, with an explicit fallback
for adverse roundtrips; passing equation tests alone does not imply fidelity. No editing quality,
general editing-quality improvement is claimed. Executed Phase 7 ablations are
limited to the recorded inputs/settings and include changed geometry and failures.

A separately labelled [pretrained instruction-editing reference](docs/instruction-reference.md)
recolored the two exposed development photographs with InstructPix2Pix after an
authorized fixed-revision download. Background/material changes remain, so this is
evidence for another editing route, not a twelve-case acceptance pass or a change
to the custom DDIM pipeline.

## Forward-noising Img2Img, inpainting and outpainting

The user-directed traditional route is implemented separately from DDIM inversion.
The demo exposes Img2Img and inpainting through **Edit image** (Whole image /
Selected region), and outpainting through **Extend canvas**, sharing one offline
model cache and queue. Forward noising is the default editing method.
No built-in editing pipeline or new checkpoint is used. All three use our manual
CFG and eta=0 sampler after explicit seeded forward noising of VAE posterior-mode
latents. See [the equations, usage and validation](docs/forward-noising.md).

```powershell
.\.venv\Scripts\python.exe scripts/run_forward_edit.py img2img --image assets/examples/coffee.png --prompt "a pencil drawing of a cup of coffee" --strength 0.85
.\.venv\Scripts\python.exe scripts/run_forward_edit.py inpaint --image assets/examples/coffee.png --mask assets/examples/coffee_mask.png --prompt "a pencil drawing of a cup of coffee" --strength 0.85
.\.venv\Scripts\python.exe scripts/run_forward_edit.py outpaint --image assets/examples/coffee.png --prompt "a photo of a cup of coffee on a wooden table, wider view" --right 128 --strength 1
```

White/painted regions edit; zero-mask regions preserve. Outpainting adds explicit
left/right/top/bottom margins, with edge or neutral initialization and optional
editable source-edge overlap. Default overlap=0 and feather=0 plus pixel compositing
preserve the entire source rectangle exactly. Feathering or overlap deliberately
permits border changes. Strength uses **floor(strength × steps)** here, independently
of inversion editing's existing ceil rule. At zero selected steps the VAE is still
decoded and chosen compositing still applies. This is four-channel base-model
masked sampling, not a specialized nine-channel inpainting checkpoint. Seams and
geometry/semantic failures remain possible; [measured smoke examples](docs/results/forward-noising/README.md)
are not holdout acceptance results.

## Text-to-image interface

The [workspace integration review](docs/workspace-integration-review.md) explains
which acceptance tools are retained, why the human review demo stays separate,
and which experimental/reference methods remain outside production defaults.

The current four tasks are **Generate**, **Edit image**, **Extend canvas** and
**Diagnostics**. Each retains its own inputs and results. Generate and the
`/txt2img` API support prompt/negative prompt, dimensions, steps, CFG, seed, generated
image and PNG/ZIP downloads. It uses our custom DDIM sampler and cached components,
without an original image, mask or inversion. Restart the demo process to load
the new layout. See [usage, API and verification limits](docs/txt2img-demo.md)
and [workspace behavior and validation](docs/workspace-ui.md). The previous named
APIs remain compatible, including all three `/forward_*` routes; compatibility
forms are hidden from the visible navigation.
