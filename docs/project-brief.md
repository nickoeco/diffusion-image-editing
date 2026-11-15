# Project: Diffusion-Based Image Editing with DDIM Inversion and Mask-Guided Sampling

## 1. Your Role

You are an experienced Computer Vision and Deep Learning Engineer specializing in diffusion models, PyTorch, and generative image editing.

Your task is to help me build a complete, well-structured Computer Vision final project:

**Diffusion-Based Image Editing with DDIM Inversion and Mask-Guided Sampling**

This is an implementation-focused university Computer Vision project.

The goal is to develop a functional image editing system that uses a pretrained Stable Diffusion model while independently implementing the core diffusion inversion, sampling, guidance, and mask-based editing algorithms.

The final project should demonstrate:

1. A solid understanding of diffusion model mathematics.
2. Correct implementation of DDIM inversion and reverse sampling.
3. Localized text-guided image editing.
4. Modular and maintainable PyTorch code.
5. Quantitative evaluation and ablation studies.
6. An interactive image editing demo.

Prioritize **correctness, implementation depth, reproducibility, and explainability**.

Do not simply wrap existing Diffusers image editing pipelines.

---

# 2. Project Objectives

Develop a system that takes the following inputs:

- An original image.
- A source text prompt describing the original image.
- A target text prompt describing the desired modification.
- A user-defined binary or soft mask specifying the region to edit.
- Sampling parameters, such as inversion steps, editing strength, guidance scale, and random seed.

The system should output:

1. The original image.
2. A reconstructed image obtained through DDIM inversion and reverse sampling.
3. An edited image reflecting the target prompt.
4. Optional visualizations of intermediate diffusion states.

The primary objective is to modify the masked region while preserving the original content outside the mask.

Example editing tasks:

- Change a red car into a blue car.
- Replace an apple with an orange.
- Change the color of a person's shirt.
- Replace an object while preserving the surrounding scene.
- Modify the appearance of a selected object.

Focus initially on simple localized edits. More complex transformations are optional extensions.

---

# 3. Technical Stack

Use the following technologies:

### Programming Language

Python 3.10 or newer, using a version compatible with the selected dependencies.

### Deep Learning Framework

PyTorch.

### Pretrained Model

Stable Diffusion 1.5 or another compatible Stable Diffusion model if necessary.

Prefer a well-supported checkpoint compatible with the Hugging Face Diffusers library.

Verify the model identifier, access requirements, license, and compatibility before implementation.

### Libraries

- torch
- torchvision
- diffusers
- transformers
- accelerate
- safetensors
- Pillow
- numpy
- opencv-python
- matplotlib
- scikit-image
- gradio
- pytest

Optional:

- lpips
- open_clip_torch
- pandas
- seaborn

Use a compatible, reproducible dependency set. Do not blindly install the latest version of every package.

### Development Requirements

- Support CUDA when available.
- Provide CPU fallback for lightweight tests.
- Use FP16 inference on supported CUDA devices.
- Use FP32 where numerical stability requires it.
- Support deterministic random seeds where practical.
- Avoid unnecessary GPU memory consumption.
- Do not assume a GPU is available.

At the beginning, inspect the environment and report:

- Operating system.
- Python version.
- PyTorch version.
- CUDA availability.
- GPU model.
- Available GPU memory.
- Relevant installed dependencies.

If GPU resources are insufficient, explain the limitation and propose a compatible execution strategy.

---

# 4. Important Implementation Constraints

This project is implementation-oriented.

You may use pretrained neural networks provided by Stable Diffusion:

- VAE Encoder
- VAE Decoder
- CLIP Tokenizer
- CLIP Text Encoder
- U-Net Denoising Network

However, the following algorithms must be implemented independently:

1. DDIM inversion.
2. DDIM reverse sampling.
3. Classifier-Free Guidance.
4. Mask-guided latent blending.
5. Editing strength and timestep selection.
6. Image reconstruction pipeline.
7. Evaluation and ablation experiment logic.

The Diffusers library may be used for:

- Loading pretrained model weights.
- Model configuration.
- Tokenization.
- Model execution.
- Scheduler configuration and reference calculations.
- Baseline comparison.

Do not use a complete Stable Diffusion Img2Img or Inpainting Pipeline as the main implementation of our editing algorithm.

Existing pipelines may be used as baselines or for verifying correctness.

I want to understand and demonstrate the mathematical operations behind each custom component.

Do not copy large existing implementations without explaining their functionality.

---

# 5. Mathematical Background and Correctness

The system operates in the latent space of Stable Diffusion.

## 5.1 VAE Encoding

Given an input image \(x_0\), use the pretrained VAE encoder to obtain a clean latent:

\[
z_0 = E_{\mathrm{VAE}}(x_0)
\]

Use the correct image normalization and VAE latent scaling factor from the model configuration.

For deterministic reconstruction experiments, prefer the posterior mean or mode rather than random posterior sampling.

Explain how this choice affects reconstruction.

## 5.2 DDIM Inversion

Starting from the clean latent \(z_0\), implement approximate deterministic DDIM inversion.

At timestep \(t\), predict the noise:

\[
\hat{\epsilon}_t =
\epsilon_\theta(z_t,t,c_{\mathrm{src}})
\]

where:

- \(\epsilon_\theta\) is the pretrained Stable Diffusion U-Net.
- \(c_{\mathrm{src}}\) is the source prompt embedding.
- \(z_t\) is the current latent.

Estimate the clean latent:

\[
\hat{z}_0 =
\frac{
z_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_t
}{
\sqrt{\bar{\alpha}_t}
}
\]

Then estimate the latent at the next higher noise level \(s\):

\[
z_s =
\sqrt{\bar{\alpha}_s}\hat{z}_0
+
\sqrt{1-\bar{\alpha}_s}\hat{\epsilon}_t
\]

where \(s>t\) denotes a higher-noise timestep in the selected inversion schedule.

Important requirements:

- Implement the equations explicitly.
- Use the actual scheduler timesteps rather than assuming adjacent integer timesteps.
- Handle timestep boundaries correctly.
- Explicitly define the initial and terminal noise levels.
- Ensure inversion and reverse sampling use compatible noise schedules.
- Store the intermediate latent trajectory with its associated timesteps.
- Do not assume DDIM inversion reconstructs real images perfectly.

Implement timestep traversal carefully.

Avoid off-by-one errors between inversion and reverse sampling.

## 5.3 DDIM Reverse Sampling

Implement deterministic DDIM sampling from a noisy latent toward a clean latent.

Given \(z_t\), predict:

\[
\hat{\epsilon}_t =
\epsilon_\theta(z_t,t,c_{\mathrm{tgt}})
\]

Estimate:

\[
\hat{z}_0 =
\frac{
z_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_t
}{
\sqrt{\bar{\alpha}_t}
}
\]

Then update toward a lower noise level \(s<t\):

\[
z_s =
\sqrt{\bar{\alpha}_s}\hat{z}_0
+
\sqrt{1-\bar{\alpha}_s}\hat{\epsilon}_t
\]

Implement the deterministic case with \(\eta=0\) first.

Stochastic sampling is optional.

## 5.4 Classifier-Free Guidance

Implement CFG manually.

Given unconditional and conditional noise predictions:

\[
\epsilon_{\mathrm{uncond}}
\]

and

\[
\epsilon_{\mathrm{cond}}
\]

compute:

\[
\epsilon_{\mathrm{guided}} =
\epsilon_{\mathrm{uncond}}
+
w
(
\epsilon_{\mathrm{cond}}
-
\epsilon_{\mathrm{uncond}}
)
\]

where \(w\) is the guidance scale.

Requirements:

- Support adjustable guidance scale.
- Support empty unconditional prompts.
- Support negative prompts as an optional extension.
- Use the correct embedding batch order.
- Handle tensor shapes correctly.
- Support efficient batched U-Net evaluation.

Document how guidance affects reconstruction fidelity and editing strength.

## 5.5 Mask-Guided Latent Blending

Implement localized editing using a spatial mask.

Let:

- \(M\): editing mask.
- \(z_t^{\mathrm{edit}}\): edited latent.
- \(z_t^{\mathrm{src}}\): source latent at the corresponding noise level.

Perform:

\[
z_t =
M\odot z_t^{\mathrm{edit}}
+
(1-M)\odot z_t^{\mathrm{src}}
\]

Requirements:

- Mask value 1 means the region is editable.
- Mask value 0 means the region should be preserved.
- Resize the mask to latent resolution.
- Preserve mask alignment with the input image.
- Support hard binary masks initially.
- Support soft masks as an extension.
- Blend source and edited latents at matching noise levels.
- Avoid incorrect timestep indexing.
- Handle the final denoising step explicitly.

A crucial requirement:

The source trajectory and editing trajectory must use compatible timesteps.

Do not use an arbitrary inverted latent from a different timestep.

Store trajectory states using explicit timestep identifiers.

Also recognize that latent-space blending does not guarantee exact pixel preservation because VAE decoding introduces spatial coupling.

Optionally implement final pixel-space compositing to preserve pixels outside the editing mask.

Keep latent blending and final pixel compositing as separately configurable operations so their effects can be evaluated.

---

# 6. Software Architecture

Create a modular project.

Suggested structure:

```text
diffusion-image-editing/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   └── diffusion_editor/
│       ├── __init__.py
│       ├── models/
│       │   ├── sd_backend.py
│       │   ├── vae.py
│       │   └── text_encoder.py
│       ├── diffusion/
│       │   ├── schedule.py
│       │   ├── inversion.py
│       │   ├── sampler.py
│       │   └── guidance.py
│       ├── editing/
│       │   ├── mask.py
│       │   ├── blending.py
│       │   └── editor.py
│       ├── pipelines/
│       │   └── editing_pipeline.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── experiments.py
│       ├── utils/
│       │   ├── image_utils.py
│       │   ├── visualization.py
│       │   └── reproducibility.py
│       └── demo/
│           └── app.py
├── scripts/
│   ├── test_generation.py
│   ├── test_inversion.py
│   ├── test_reconstruction.py
│   ├── run_edit.py
│   └── run_experiments.py
├── tests/
│   ├── test_schedule.py
│   ├── test_inversion.py
│   ├── test_sampler.py
│   ├── test_guidance.py
│   ├── test_mask.py
│   └── test_pipeline.py
├── assets/
│   └── examples/
├── outputs/
│   ├── reconstruction/
│   ├── editing/
│   └── experiments/
└── docs/
    ├── architecture.md
    ├── implementation.md
    └── experiments.md
```

You may simplify the structure if necessary, but keep the following concerns separated:

- Model loading.
- Diffusion mathematics.
- Inversion.
- Sampling.
- Text conditioning.
- Editing.
- Evaluation.
- User interface.

Avoid unnecessary abstractions.

Use type hints and concise docstrings for important functions.

---

# 7. Development Phases

Implement the project incrementally.

Do not attempt to implement the entire project in one pass.

Each phase must produce a working, testable deliverable.

## Phase 0 — Environment and Project Initialization

Objectives:

1. Inspect the current environment.
2. Create the project structure.
3. Configure dependencies.
4. Implement configuration management.
5. Initialize basic logging.
6. Create the README.
7. Verify Stable Diffusion model loading.

Deliverables:

- Project structure.
- Dependency configuration.
- Environment verification script.
- Model loading code.
- Basic generation smoke test.

Acceptance criteria:

- The project can be installed.
- The model can be loaded in the available environment.
- CUDA is detected correctly when available.
- A basic inference call works if the required compute resources are available.
- Installation and execution commands are documented.

If the environment cannot run the model, complete the CPU-compatible structural tests and document the remaining GPU verification steps.

Do not proceed to complex editing code before the model-loading interface is established.

## Phase 1 — VAE and Text Conditioning

Implement:

### VAE Wrapper

Functions:

```python
encode_image(image) -> latents
decode_latents(latents) -> image
```

Requirements:

- Correct image preprocessing.
- Correct normalization.
- Correct latent scaling.
- Correct device handling.
- Correct tensor dimensions.

### Text Encoder Wrapper

Functions:

```python
encode_prompt(prompt) -> embeddings
encode_prompt_pair(prompt, negative_prompt) -> embeddings
```

Requirements:

- Proper tokenization.
- Correct attention and embedding dimensions.
- Batch support.
- Device compatibility.

Tests:

1. Image encode/decode shape consistency.
2. VAE reconstruction quality.
3. Prompt embedding shape.
4. CPU/GPU device consistency.

Save an example showing:

Original Image | VAE Reconstruction

This provides a baseline for distinguishing VAE reconstruction error from DDIM inversion error.

## Phase 2 — DDIM Scheduler and Inversion

Implement:

### Noise Schedule

Create a module that handles:

- Beta values.
- Alpha values.
- Cumulative alpha products.
- Inference timesteps.
- Forward and reverse timestep ordering.
- Boundary conditions.

Use the selected pretrained model's scheduler configuration.

Do not create a different noise schedule incompatible with its training.

### DDIM Inverter

Suggested interface:

```python
class DDIMInverter:
    def invert(
        self,
        latents,
        source_embeddings,
        num_steps,
        guidance_scale=1.0,
    ):
        ...
```

Return a structured result containing:

- Initial clean latent.
- Final inverted latent.
- Intermediate latent trajectory.
- Corresponding timestep identifiers.
- Inversion configuration.

Implement explicit inversion equations.

Tests:

1. Verify timestep ordering.
2. Verify cumulative alpha indexing.
3. Verify output tensor shapes.
4. Verify no NaN or Inf values.
5. Verify deterministic behavior under a fixed configuration.
6. Compare individual DDIM step calculations with trusted reference equations.

Do not proceed until the inversion module passes unit tests.

## Phase 3 — DDIM Reverse Sampling and Reconstruction

Implement a custom DDIM sampler.

Suggested interface:

```python
class DDIMSampler:
    def sample(
        self,
        initial_latent,
        prompt_embeddings,
        trajectory_timesteps,
        guidance_scale=1.0,
        return_trajectory=False,
    ):
        ...
```

Implement CFG separately from the sampling update.

Then implement image reconstruction:

```python
reconstruct(
    image,
    source_prompt,
    num_steps,
    guidance_scale,
)
```

Pipeline:

1. VAE encode.
2. DDIM inversion.
3. Reverse DDIM sampling with source prompt.
4. VAE decode.
5. Save reconstruction.

Acceptance criteria:

- Reverse sampling uses exactly the intended timestep schedule.
- Reconstruction can run end-to-end.
- Output dimensions match the input.
- Reconstruction metrics can be computed.
- Intermediate results can be visualized.

Evaluate:

- VAE-only reconstruction.
- DDIM inversion + reverse reconstruction.

Use PSNR and SSIM initially.

Optionally add LPIPS.

Do not assume that visual similarity alone proves mathematical correctness.

## Phase 4 — Text-Guided Image Editing

Implement text-guided editing without spatial masks first.

Pipeline:

1. Encode input image.
2. Invert using source prompt.
3. Select a starting noise level.
4. Reverse sample using target prompt.
5. Decode the resulting latent.

Suggested interface:

```python
edit_global(
    image,
    source_prompt,
    target_prompt,
    strength,
    guidance_scale,
    num_steps,
)
```

Implement editing strength by selecting a valid timestep from the stored inversion trajectory.

Clearly define the relationship between:

- Editing strength.
- Selected starting timestep.
- Noise level.
- Number of reverse denoising steps.

Acceptance criteria:

- Source and target prompts are handled independently.
- Editing strength changes the starting noise level.
- Same-prompt reconstruction and target-prompt editing can be compared.
- Results are reproducible under fixed settings.

Start with simple semantic modifications.

Do not attempt complex object replacement until this phase works reliably.

## Phase 5 — Mask-Guided Editing

Implement localized editing.

Suggested interface:

```python
edit_local(
    image,
    source_prompt,
    target_prompt,
    mask,
    strength,
    guidance_scale,
    num_steps,
    feather_radius=0,
    preserve_background=True,
)
```

Implement:

1. Binary mask preprocessing.
2. Mask resizing.
3. Optional Gaussian feathering.
4. Latent-space blending.
5. Source trajectory lookup.
6. Optional final pixel-space compositing.

Important:

At every reverse sampling step, correctly align the edited latent with the source latent at the same noise level.

Define whether blending occurs before or after the denoising update.

If blending occurs after the update, use the source latent corresponding to the updated timestep.

Pay special attention to the terminal step.

Acceptance criteria:

- Mask shape is validated.
- Empty mask preserves the original image when background preservation is enabled.
- Full mask behaves like global editing.
- Small masks enable localized changes.
- Background preservation is measurable.
- The system supports both binary and soft masks.

Add unit tests with synthetic tensors before testing real images.

## Phase 6 — Interactive Gradio Demo

Build an interactive Gradio interface.

Inputs:

- Original image.
- Source prompt.
- Target prompt.
- Editable mask.
- Number of sampling steps.
- Editing strength.
- Guidance scale.
- Random seed.

Outputs:

- Original image.
- VAE reconstruction.
- DDIM reconstruction.
- Edited image.
- Optional editing visualization.

Support:

- Image upload.
- Mask drawing or mask upload.
- Parameter adjustment.
- Image generation.
- Image downloading.
- Error reporting.
- Basic execution-time measurement.

Keep the UI clean and understandable.

Do not prioritize visual complexity over functionality.

Ensure the demo calls the custom editing pipeline rather than the built-in Diffusers inpainting pipeline.

## Phase 7 — Experiments and Evaluation

Create reproducible scripts for the following experiments.

### Experiment A: Reconstruction Quality

Compare:

1. Original image.
2. VAE reconstruction.
3. DDIM reconstruction.

Test different inversion step counts:

- 20
- 50
- 100

Metrics:

- PSNR.
- SSIM.
- LPIPS, if available.

Analyze reconstruction error separately from VAE encoding/decoding error.

### Experiment B: Editing Strength

Test:

- 0.25
- 0.50
- 0.75
- 1.00

Measure:

- Editing effectiveness.
- Background preservation.
- Reconstruction similarity.
- Inference time.

### Experiment C: Guidance Scale

Test:

- 3.0
- 5.0
- 7.5
- 10.0

Evaluate the trade-off between prompt alignment and image preservation.

### Experiment D: Mask-Guided Editing

Compare:

1. Global editing without a mask.
2. Mask-guided latent blending.
3. Mask-guided latent blending with feathering.
4. Optional final pixel-space compositing.

Evaluate:

- Background mean absolute error.
- Background PSNR or region-restricted similarity.
- Target-prompt CLIP similarity.
- Visual editing quality.

For masked-region perceptual metrics, avoid misleading calculations caused by simply zeroing all pixels outside the region.

Document exactly how each region-specific metric is computed.

### Experiment E: Baseline Comparison

Compare the custom method with at least one existing baseline, such as:

- Stable Diffusion Img2Img.
- Stable Diffusion Inpainting.

Use comparable input images, prompts, and computational budgets where possible.

Explain that the baseline and custom algorithm may use different conditioning mechanisms, so the comparison is not necessarily a controlled comparison of a single algorithmic component.

### Experimental Reproducibility

For every experiment:

- Record input image.
- Record source and target prompts.
- Record mask.
- Record random seed.
- Record model identifier.
- Record inference steps.
- Record guidance scale.
- Record editing strength.
- Record runtime.
- Save output images.
- Save metrics as CSV or JSON.

Generate visualization grids and summary plots automatically.

Do not fabricate results.

Do not claim improvement unless supported by actual measurements.

---

# 8. Testing Requirements

Testing is mandatory throughout development.

Implement unit tests for:

### Scheduler

- Alpha calculations.
- Timestep selection.
- Boundary handling.
- Ascending and descending traversal.

### DDIM Inversion

- Mathematical update.
- Latent shape preservation.
- Trajectory length.
- Timestep alignment.
- Deterministic behavior.

### DDIM Sampling

- Reverse update formula.
- Correct coefficient selection.
- Boundary behavior.

### CFG

- Guidance scale 0 produces the unconditional prediction.
- Guidance scale 1 produces the conditional prediction.
- Higher guidance scales follow the implemented linear combination.
- Correct batching and tensor shapes.

### Mask Editing

- All-zero mask.
- All-one mask.
- Partial binary mask.
- Soft mask.
- Incorrect mask dimensions.
- Timestep alignment.

Use lightweight synthetic data and mock models for unit tests that should not require downloading Stable Diffusion.

Separate these from GPU-dependent integration tests.

Do not mark GPU integration tests as passed unless they actually run successfully.

---

# 9. Code Quality Requirements

Follow these engineering practices:

1. Use modular Python classes and functions.
2. Include type hints.
3. Write meaningful docstrings.
4. Avoid duplicated code.
5. Use consistent tensor conventions.
6. Manage device and dtype explicitly.
7. Avoid hardcoding image or latent dimensions.
8. Use model configuration values instead of unexplained constants.
9. Handle invalid inputs gracefully.
10. Keep experimental scripts separate from library code.

For numerical code:

- Explain important tensor dimensions.
- Document timestep ordering.
- Avoid unnecessary CPU/GPU transfers.
- Check for NaN and Inf values where relevant.
- Handle numerical precision carefully.

Do not silently suppress exceptions.

Avoid creating unnecessary dependencies.

---

# 10. Documentation Requirements

Maintain the following documentation throughout development.

### README.md

Include:

- Project overview.
- Main features.
- System architecture.
- Installation instructions.
- Model setup.
- Usage examples.
- Demo instructions.
- Experiment instructions.
- Known limitations.

### docs/implementation.md

Explain:

- VAE encoding and decoding.
- DDIM inversion equations.
- DDIM reverse sampling equations.
- Classifier-Free Guidance.
- Mask-guided blending.
- Timestep alignment.
- Editing strength selection.

For each major equation, identify the function implementing it.

### docs/experiments.md

Include:

- Experimental setup.
- Evaluation metrics.
- Dataset or example image selection.
- Reproducibility parameters.
- Results.
- Limitations.

Only add numerical results after experiments have actually been executed.

---

# 11. Definition of Done

Consider the project complete only when:

- [ ] Stable Diffusion components load successfully.
- [ ] VAE encode/decode works.
- [ ] Custom DDIM inversion works.
- [ ] Custom DDIM reverse sampling works.
- [ ] Same-prompt reconstruction works.
- [ ] Classifier-Free Guidance works.
- [ ] Global text-guided editing works.
- [ ] Mask-guided local editing works.
- [ ] Interactive Gradio demo works.
- [ ] Unit tests pass.
- [ ] At least two ablation experiments are completed.
- [ ] Evaluation metrics are generated.
- [ ] Example results are saved.
- [ ] README and implementation documentation are complete.

The core project should not depend on optional advanced features.

---

# 12. Stretch Goals

If the core implementation is complete, consider:

1. Soft mask and feathering.
2. Negative prompt support.
3. Comparison between different inversion depths.
4. Intermediate denoising visualization.
5. Improved inversion methods.
6. Prompt-to-Prompt editing.
7. Null-text inversion.
8. More advanced background preservation methods.

Do not implement stretch goals until the core pipeline is working and tested.

---

# 13. Development Workflow

Follow an iterative development process.

For each phase:

### Step 1 — Explain

Before implementing a major component, briefly explain:

- Its purpose.
- Required inputs.
- Expected outputs.
- Mathematical assumptions.
- Important implementation risks.

### Step 2 — Implement

Write the necessary code.

Keep changes focused on the current phase.

### Step 3 — Test

Run relevant tests.

Report:

- Tests executed.
- Tests passed.
- Tests failed.
- Errors encountered.
- Remaining limitations.

### Step 4 — Review

Inspect the implementation for:

- Mathematical correctness.
- Timestep indexing errors.
- Tensor shape mismatches.
- Device inconsistencies.
- Numerical instability.
- Unnecessary complexity.

### Step 5 — Summarize

After completing each phase, provide:

1. Files created or modified.
2. Main functionality implemented.
3. Important implementation decisions.
4. Verification results.
5. Known limitations.
6. Recommended next steps.

Do not claim success without verification.

If a technical issue prevents progress, explain the actual blocker and propose a reasonable solution.

Do not make major architectural changes without explaining the reason.

---

# 14. Important Collaboration Instructions

I want to learn from this project, not just obtain working code.

Therefore:

- Explain important mathematical implementations.
- Point out non-obvious design decisions.
- Keep the core algorithms readable.
- Avoid hiding the entire implementation behind high-level library calls.
- Distinguish pretrained model functionality from our custom algorithms.
- Explicitly identify where DDIM equations are implemented.
- Explain how mask blending interacts with reverse sampling.
- Document assumptions and limitations.

Avoid excessive explanation of trivial Python syntax.

Focus explanations on diffusion mathematics, software architecture, and engineering decisions.

When relevant, reference the corresponding equations from DDIM or diffusion literature.

---

# 15. Current Task — Start with Phase 0

For your first response, do the following:

1. Inspect the current workspace and environment.
2. Identify existing project files and preserve them.
3. Check Python, PyTorch, CUDA, and relevant dependencies.
4. Propose the final project architecture.
5. Identify likely technical risks.
6. Create a concise, actionable implementation roadmap.
7. Initialize the project structure and basic configuration.
8. Implement the pretrained Stable Diffusion model-loading module.
9. Implement a lightweight environment verification script.
10. Add basic tests that do not require GPU execution.

Then run the available checks.

Do not begin DDIM inversion, mask editing, Gradio development, or experimental evaluation in this first phase.

Before making major dependency or model choices, explain their compatibility and resource requirements.

Do not download very large model weights until the hardware environment and model choice have been confirmed.

After Phase 0, stop and provide a progress report.

Wait for my instruction before proceeding to Phase 1.

**Begin by inspecting the environment and presenting the implementation plan.**