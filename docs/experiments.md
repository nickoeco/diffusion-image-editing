# Experiments

Reconstruction smoke metrics are available from Phase 3. No formal ablation suite,
editing evaluation or baseline comparison has been completed.

Phase 1 produced an unscored VAE-only smoke comparison at
`outputs/reconstruction/phase1/comparison.png`, using the real pretrained VAE and a
deterministic synthetic test card. Its configuration and environment are recorded
in the adjacent `report.json`. This is a structural baseline rather than an
ablation or a natural-image quality benchmark; no PSNR/SSIM or improvement claims
are reported. See `phase1-status.md` for executed verification.

Phase 2 saved a real 50-step inversion trajectory and execution metadata under
`outputs/inversion/phase2/`. This is a correctness smoke run, not reconstruction,
an ablation or an editing-quality result. Scheduler reference tests at different
step counts do not count as completed experimental ablations.

## Phase 3 measured reconstruction checks

All runs use the pinned SD1.5 revision, seed 42, 50 steps, clean alpha=1,
FP32 latent arithmetic and FP16 U-Net/CLIP. VAE-only and DDIM outputs are measured
against the original RGB image with data range 1. SSIM uses uniform 7×7 windows,
channel averaging and sample covariance. These are single-image smoke measurements.

| Input | Inversion/sampling guidance | VAE PSNR (dB) | VAE SSIM | DDIM PSNR (dB) | DDIM SSIM |
| --- | --- | --- | --- | --- | --- |
| Generated test card | 1 / 1 | 29.066 | 0.8367 | 10.990 | 0.6174 |
| Natural astronaut example | 1 / 1 | 26.799 | 0.8706 | 18.711 | 0.6283 |
| Natural astronaut example | 3 / 3 | 26.799 | 0.8706 | 12.888 | 0.4405 |

The generated card uses source prompt `a red apple on a colorful background`.
The natural example uses `a portrait of an astronaut wearing a spacesuit` and has
[documented public-domain provenance](../assets/examples/README.md). Negative prompts
are empty. VAE-only metrics are not DDIM inversion scores. The DDIM outputs exhibit
visible appearance drift; guidance 3 worsens preservation in this check. These
observations do not establish a general guidance relationship or improvement.

PNG comparisons, optional trajectories/previews, JSON reports and CSV metrics are
saved under `outputs/reconstruction/phase3/`, `phase3-astronaut/` and `phase3-cfg3/`.
Reports record input hashes, prompts, model revision, both guidance scales, seed,
exact grids, environment, runtime and peak allocated memory. No mask or editing
strength is used for reconstruction. This does not fulfill the two-ablation
completion requirement. See `phase3-status.md` for validation details.

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
