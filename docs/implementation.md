# Implementation status and mathematical contracts

Phases 1–2 implement VAE/text conditioning, the noise grid and custom DDIM inversion.
`models/sd_backend.py::SDBackend.load` loads model configuration and weights;
`resolve_runtime` selects device/dtype. Scheduler configuration is loaded with
`DDIMScheduler.from_pretrained`; it is not yet used to execute custom sampling.

## VAE encoding and decoding (implemented)

`utils/image_utils.py::image_to_tensor` accepts PIL images or floating RGB BCHW
tensors in [0,1]. PIL inputs are EXIF-oriented and converted to RGB. Tensor layout,
range and finite values are validated. There is no implicit resize or crop.

`models/vae.py::VAEWrapper.encode_image` implements:

```text
x_normalized = 2*x - 1
z = mode(q_VAE(z | x_normalized)) * vae.config.scaling_factor
```

The model-configured spatial factor is `2**(len(block_out_channels)-1)`. Input
dimensions must be divisible by that factor; output is [B,latent_channels,H/f,W/f].
The posterior mode is also the Gaussian posterior mean. It removes random posterior
sampling variation but cannot undo the VAE bottleneck's information loss.

`VAEWrapper.decode_latents` implements:

```text
x_decoded = VAE.decode(z / vae.config.scaling_factor)
x_rgb = clamp(x_decoded / 2 + 0.5, 0, 1)
```

Both methods run in inference mode on the VAE device in FP32, validate model output
shape/finite values and return tensors on that device. Input tensors can originate
on CPU or use another floating dtype; conversion is explicit at the model boundary.
The wrapper requires an FP32 VAE. `tensor_to_images` performs a deliberate CPU
transfer and uint8 rounding only for saving/display, not for numerical evaluation.

See the pinned [AutoencoderKL documentation](https://huggingface.co/docs/diffusers/v0.35.1/en/api/models/autoencoderkl)
for the pretrained latent scaling convention.

## Text conditioning (implemented)

`models/text_encoder.py::TextEncoderWrapper.encode_prompt` accepts a string or a
nonempty sequence of strings and returns [B,L,D] `last_hidden_state`. This is the
token sequence consumed by the U-Net, not CLIP's pooled projection. L comes from
the tokenizer's maximum length and D from the text encoder configuration.

Token IDs stay integer and move to the model device. Padding is to maximum length;
long inputs are truncated with an explicit warning. An attention mask is passed
only if `text_encoder.config.use_attention_mask` requests it. Model execution uses
eval/inference mode and the backend's CLIP dtype.

`encode_prompt_pair(prompt, negative_prompt="")` returns [2B,L,D] ordered as
`[negative_0,...,negative_B-1, positive_0,...,positive_B-1]`. A scalar negative prompt
broadcasts over B; a sequence must match B exactly. Empty negatives provide
unconditional conditioning. This method prepares the CFG batch but does not yet
implement CFG arithmetic. The CLIP contract follows the pinned
[Transformers reference](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/clip).

## Verification

Synthetic model tests check normalization endpoints, nondefault scaling, inverse
scaling, posterior mode (never sample), batches, shapes, deterministic repetition,
invalid inputs, token/embedding order, truncation, attention masks and device/dtype.
GPU tensor tests require `--run-gpu`; real cached-weight tests require
`--run-integration`. Neither default pytest nor integration tests download weights.
The opt-in `scripts/check_conditioning.py` saves a labelled original/VAE comparison
and reproducibility metadata; it supports an explicitly supplied image or a generated
test card. Real-model results are recorded separately in the Phase 1 status report.

On the tested GPU, FP16 CLIP embeddings changed by up to 0.01953125 in absolute
value between combined batch size 4 and split batch size 2 execution. Different
kernel batching can change rounding; do not require bitwise equivalence across
batch shapes. The real order check compares the wrapper with direct pretrained
execution using identical token order and batch size. Synthetic tests independently
verify that the first B entries are unconditional and the remaining B are conditional.

## Noise schedule and clean boundary (implemented)

`diffusion/schedule.py::DDIMSchedule.from_scheduler` clones the pretrained DDIM
scheduler configuration, reads its beta vector and actual inference IDs, and
checks that FP32 cumulative products match the scheduler. Training coefficients
and the original scheduler are unchanged. No adjacent-integer or fixed-stride
assumption is used by custom traversal, so nonuniform grids are supported.

```text
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_0, ..., alpha_t)
clean ID = -1; alpha_bar_clean = 1
```

The checkpoint's `set_alpha_to_one=false` would use alpha_bar_0 at the denoising
boundary. The VAE latent is actually clean, so our cloned DDIM view explicitly
sets `set_alpha_to_one=True`. Inversion and future reverse sampling must share
this choice. Training timestep 0 remains distinct from clean. Unsupported epsilon
singularities, clipping/thresholding, non-epsilon models and invalid grids fail.

For the pinned checkpoint and 50 leading steps, inversion levels are
`[-1, 1, 21, ..., 981]`; reverse levels will be `[981, ..., 21, 1, -1]`.
The terminal inverted level is 981, not an assumed 999 or pure Gaussian sample.
Requesting 1000 leading steps with offset 1 produces an out-of-range ID and is
rejected rather than silently clamped.

## DDIM inversion (implemented)

`diffusion/inversion.py::ddim_inversion_step` implements the equations below for
the actual current/next higher-noise pair (t,s), where alpha_bar_s < alpha_bar_t:

```text
x0_hat = (z_t - sqrt(1-alpha_bar_t)*epsilon) / sqrt(alpha_bar_t)
z_s = sqrt(alpha_bar_s)*x0_hat + sqrt(1-alpha_bar_s)*epsilon
```

All coefficient/update arithmetic is FP32, with shape/device/finiteness checks.
No stochastic noise, clipping or library step is used. `DDIMInverter.invert`
records N+1 snapshots including clean, by explicit state ID, with query IDs.
The first noise prediction uses the lowest selected training timestep on the clean
latent; subsequent predictions use the current state timestep. At 50 steps the
query IDs are `[1, 1, 21, ..., 961]`. The clean bookkeeping ID never reaches the
U-Net. Noise prediction at the current state approximates the unknown higher-noise
state; this is not an exact inverse of future denoising.

`models/noise_predictor.py::UNetNoisePredictor` casts model inputs to U-Net runtime
dtype and returns FP32 epsilon. Only guidance 1 is supported; other scales fail
explicitly until Phase 3 implements manual CFG separately.

Tests use independent NumPy float64 equations and an analytic constant-epsilon
trajectory, with irregular gaps, boundaries and deterministic repetition. Optional
reference tests compare fixed-epsilon updates with `DDIMInverseScheduler` at
1/20/50/100 steps, plus leading/linspace/trailing grid preservation. These verify
coefficient math, not equality between full neural pipelines with different
query-time conventions. Production never calls the reference step. See the
[DDIM paper](https://arxiv.org/abs/2010.02502) and pinned
[scheduler reference](https://huggingface.co/docs/diffusers/v0.35.1/en/api/schedulers/ddim_inverse).

## Later phases (not implemented)

- Reverse DDIM sampling will traverse the same levels in descending order, using
  the explicit eta=0 equation and the same clean boundary (Phase 3).
- CFG: eps = eps_uncond + w*(eps_cond-eps_uncond); embedding and U-Net batch order
  must match. Larger w can improve target adherence but damage reconstruction.
- Local blend: z_s = M*z_edit_s + (1-M)*z_source_s after updating t -> s (Phase 5).
  Look up source **s**, including the clean terminal state. Do not blend against t.
- Strength must select a stored level and define endpoints and actual reverse step
  count (Phase 4). Mask resizing must preserve original spatial alignment.

Each later phase must replace these planned descriptions with actual function references,
tested timestep conventions, tensor contracts and evidence.

References: [DDIM paper](https://arxiv.org/abs/2010.02502),
[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598).
