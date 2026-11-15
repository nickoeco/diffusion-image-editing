# Implementation status and mathematical contracts

Only configuration, diagnostics and the pretrained loading interface are implemented.
`models/sd_backend.py::SDBackend.load` loads model configuration and weights;
`resolve_runtime` selects device/dtype. Scheduler configuration is loaded with
`DDIMScheduler.from_pretrained`; it is not yet used to execute custom sampling.

The following equations are **planned**, with no implementing functions yet:

- VAE: normalize RGB to [-1,1]; z = posterior.mode() * vae.config.scaling_factor.
  Decode with the inverse scale. Deterministic posterior mode avoids sampling
  variation, but cannot remove lossy VAE reconstruction error (Phase 1).
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

Each phase must replace these planned descriptions with actual function references,
tested timestep conventions, tensor contracts and evidence.

References: [DDIM paper](https://arxiv.org/abs/2010.02502),
[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598).
