# Development roadmap

| Phase | Working deliverable | Gate |
| --- | --- | --- |
| 0 | Environment, package, config/logs, component loader, smoke scripts | Offline tests, installation, honest integration status; stop for user |
| 1 | VAE and prompt wrappers | Shape/device tests; saved VAE reconstruction baseline |
| 2 | Schedule and approximate DDIM inversion | Tested equations, boundaries, deterministic trajectory with explicit timestep IDs |
| 3 | Custom reverse DDIM + CFG + reconstruction | Unit tests, actual same-prompt reconstruction, PSNR/SSIM |
| 4 | Global target-prompt editing | Stored-level strength selection; reproducible simple edits |
| 5 | Local mask editing | Synthetic mask tests, aligned levels, measured background preservation |
| 6 | Gradio demo | Upload/mask/parameters/outputs/download/timing/errors calling custom pipeline |
| 7 | Reproducible evaluation and baseline | At least two ablations executed, metrics, outputs, plots and limitations |

Every phase follows explain -> implement -> test -> review -> summarize. Do not
implement later-phase placeholders as if functional. Phase 1 requires a new user
instruction. See AGENTS.md for binding constraints and test acceptance details.
