# Architecture

Phase 0 establishes a small pretrained boundary: `SDBackend.load(ProjectConfig)`
returns tokenizer, text encoder, VAE, U-Net, scheduler, device, dtype and configuration.
It loads components directly, freezes parameters and enters eval mode. It performs
no DDIM updates and makes no assumptions about future trajectory storage.

Configuration loading and host inspection do not import heavyweight ML dependencies.
Logging is configured by script entrypoints, not as an import side effect. Relative
cache paths resolve against the process working directory; run from repository root.
An explicit immutable model revision accompanies every load call.

Future data flow: RGB image -> deterministic VAE encode -> source-conditioned
custom inversion -> trajectory indexed by explicit noise-level ID -> custom reverse
sampling with target CFG -> aligned optional mask blending -> VAE decode -> optional
pixel composite. The clean boundary needs its own identity, distinct from training
timestep zero. Phase 2 will implement and test this contract before any editing code.

Diffusion mathematics must not depend on UI or disk I/O. Pipeline orchestration
combines the modules; evaluation and Gradio call that pipeline. Full Diffusers
pipelines are confined to labelled reference/baseline scripts. No circular UI/model
dependencies or generic plugin framework is needed.

Phase 1 adds `VAEWrapper` and `TextEncoderWrapper` on top of this boundary.
Image conversion lives in `utils/image_utils.py`; it does not resize or crop inputs.
The wrappers accept pretrained components directly, allowing synthetic PyTorch
models in offline tests without importing Diffusers or Transformers.
`scripts/check_conditioning.py` orchestrates a real-weight smoke check and saving
results, while model wrappers remain independent of file paths and UI state.

Runtime policy: CUDA auto uses FP16 U-Net/CLIP, CPU uses FP32, VAE always uses FP32.
The VAE wrapper casts its inputs explicitly; CLIP token IDs stay integer while
output embeddings use the model dtype. Coefficients/critical math will use FP32.
Trajectory memory retention and CPU offload remain decisions for the inversion phase,
after measuring actual GPU memory rather than assuming unlimited resources.
