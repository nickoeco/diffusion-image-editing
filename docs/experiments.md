# Experiments

No experiments have been executed. No reconstruction/editing metrics are available.

Phase 1 produced an unscored VAE-only smoke comparison at
`outputs/reconstruction/phase1/comparison.png`, using the real pretrained VAE and a
deterministic synthetic test card. Its configuration and environment are recorded
in the adjacent `report.json`. This is a structural baseline rather than an
ablation or a natural-image quality benchmark; no PSNR/SSIM or improvement claims
are reported. See `phase1-status.md` for executed verification.

Phase 7 will evaluate reconstruction at 20/50/100 steps; strength at
0.25/0.50/0.75/1.0; guidance at 3/5/7.5/10; global versus latent blending versus
feathering versus optional pixel compositing; and a clearly labelled Diffusers
Img2Img/Inpainting baseline with comparable budgets. At least two executed
ablations are required before claiming project completion.

Use simple localized edits first and record example provenance. Separate VAE-only
loss from inversion+reverse loss. PSNR and SSIM start with RGB float [0,1] and
explicit data range 1. For background MAE use an explicitly defined background
mask and divide summed absolute error by the number of background channel values.
An empty evaluation region is undefined and must not produce a misleading zero.
Do not compute regional perceptual scores by zeroing out other pixels.

Every run must save image, mask, prompts, seed, model ID/revision, software versions,
hardware/dtype, step schedule, guidance, strength, blending/composite settings,
runtime, generated outputs and JSON/CSV metrics. CLIP/LPIPS are optional and must
record their model identifiers. Actual results, grids and plots will be added only
after execution; pixel compositing must not be credited as a latent-algorithm gain.
