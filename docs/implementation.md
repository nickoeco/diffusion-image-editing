# Implementation status and mathematical contracts

Phase 1 implements deterministic VAE and token-level text conditioning wrappers.
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

## Later phases (not implemented)

- DDIM: x0_hat = (z_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t);
  z_s = sqrt(alpha_bar_s)*x0_hat + sqrt(1-alpha_bar_s)*eps.
  The same deterministic update works toward higher noise for inversion (Phase 2)
  and lower noise for sampling (Phase 3). Inversion approximates the unknown future
  noise prediction; real-image reconstruction is not exact.
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
