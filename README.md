# Diffusion-Based Image Editing

A Computer Vision course project using pretrained Stable Diffusion 1.5 with custom
DDIM inversion, reverse sampling, Classifier-Free Guidance (CFG), and mask-guided
latent blending. **Phase 6 adds a local Gradio interface for uploaded or painted
masks, reconstruction, global/local editing and downloadable run artifacts.**
All future changes must
follow [AGENTS.md](AGENTS.md).

Project documentation is maintained in English. Workflow communication may use
English, Chinese, or both.

Reconstruction defaults to budgeted residual-refined DDIM with an end-to-end latent-error
selection check. Plain DDIM remains available for teaching and controlled baselines.
See [the drift investigation](docs/reconstruction-drift.md) for measured evidence,
the computational trade-off and known failure cases.

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
Phase 3 adds scikit-image 0.25.2 for SSIM. Phase 6 adds optional Gradio 5.49.1;
OpenCV and plotting libraries remain deferred.

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
  evaluation/             PSNR/SSIM; experiment orchestration planned
  demo/                   Local Gradio UI, validated inputs and cached-model service
  utils/                  Environment diagnostics and logging
scripts/                  Environment checks, component forward, reference generation
tests/                    Offline CPU unit tests
docs/                     Architecture, mathematical contracts, roadmap, experiment records
assets/examples/          Example inputs and masks with recorded provenance
outputs/                  Runtime artifacts excluded from version control
```

Global and local editing are available through `scripts/run_edit.py`.
The Gradio demo is available below; formal experiment commands remain pending. See the
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

1. Upload an RGB image or load the included coffee example. Dimensions must be
   divisible by the VAE factor (8 for this model), between 8 and 1024 pixels per
   side; around 512 is recommended. Inputs are never silently resized or cropped.
2. Choose Local edit, Global edit or Reconstruct source. Set source/target prompts,
   sampling steps, strength, guidance and seed.
3. For local editing, choose **Upload** for an aligned grayscale mask or **Draw**
   and paint over the original. White/painted regions edit. The canvas stays visible
   to avoid a Gradio hidden-canvas initialization issue; the unused mask is ignored.
   Brush, erase, undo and clear controls are available. Changing the original
   clears the uploaded mask and painting to prevent stale alignment.
4. Feathering, latent blending and original-background pixel compositing are
   independent controls. Advanced settings optionally calculate full source
   reconstruction and decoded-latent previews. Full reconstruction uses guidance
   1 independently of editing guidance and is slower with residual correction.
5. Run and inspect the original, VAE baseline, optional full DDIM reconstruction,
   edit, matched source control and effective mask. Download the result PNG or ZIP
   containing images, prompts/settings, JSON/CSV metrics and environment/timings.

Artifacts use isolated directories under `outputs/demo/<run-id>/`. The report
separates loading time from pipeline compute time; compute excludes metrics,
preview decoding, image saving and archive creation. PSNR/SSIM measure preservation,
not prompt alignment. Seed is recorded, but deterministic inversion adds no random
noise, so changing it alone need not change the image. Data and cached weights stay
on the local server. This demo has no authentication and is intended for local use.

HTTP checks require local sockets and are a separate opt-in from offline tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --run-demo
# Include cached real-model and CUDA checks:
.\.venv\Scripts\python.exe -m pytest -q --run-demo --run-integration --run-gpu
```

See [Phase 6 validation](docs/phase6-status.md) for executed evidence and limitations.

## Known limitations

Offline synthetic tests verify tensor contracts and equations; actual pretrained
verification is tracked separately in the phase status records. DDIM inversion is
approximate, and the VAE introduces reconstruction error. Latent masks cannot
guarantee exact pixel preservation; final pixel compositing is independently
configurable. Project completion is governed by the definition of done in AGENTS.md.
The historical plain-DDIM natural-image smoke shows substantial reconstruction
drift. Residual refinement addresses the measured cases, with an explicit fallback
for adverse roundtrips; passing equation tests alone does not imply fidelity. No editing quality,
general improvement or completed Phase 7 ablation is claimed.
