# Pretrained instruction-editing reference

## Question and scope

The current custom SD 1.5 text-swap experiments have not established reliable
recoloring. A separately labelled InstructPix2Pix reference will test whether a
pretrained image-and-instruction editing route can recolor the same exposed apple
and car photographs. This changes weights, conditioning, scheduler, precision and
compute. It cannot isolate the cause to model weights, certify the custom DDIM
implementation, or establish general semantic acceptance from two cases.

This reference is confined to an output-local baseline script. It is not an
installed editor mode and does not replace inversion, sampling or CFG in the core
implementation. The reconstruction gate and existing ratings remain unchanged.
No holdout image is generated.

## Registered setup

- Model: `timbrooks/instruct-pix2pix`, immutable revision
  `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`.
- Public metadata checked on November 18, 2026: ungated, `mit` license label.
  See the [model card](https://huggingface.co/timbrooks/instruct-pix2pix) and
  [authors' implementation](https://github.com/timothybrooks/instruct-pix2pix).
- The pinned configuration declares an eight-channel U-Net and an
  `EulerAncestralDiscreteScheduler`, using epsilon prediction and 1,000 training
  levels with scaled-linear betas. This is a stochastic reference scheduler,
  separate from our deterministic DDIM method.
- The installed Diffusers 0.35.1 supplies the reference pipeline. No package
  upgrade or new dependency is needed. Loading and both real-weight GPU runs
  completed successfully after authorization.
- Load only the FP16 safetensors components and small required configuration/tokenizer
  files. Four weight files total 2,740,646,990 bytes, about 2.74 GB decimal or
  2.55 GiB. This includes the safety checker, which remains enabled. Downloads
  use the project cache and require new explicit authorization.
- Host inspection found an RTX 4080 Laptop GPU with 12,282 MiB total and 10,299 MiB
  free. Availability is a point-in-time observation, not a guaranteed memory budget.
- Two registered, exposed development inputs retain their original resolution,
  provenance, image/mask hashes and source-fixed boundary coordinates.
- Literal instructions: `Make the apple green.` and `Make the car yellow.`
  These are generation instructions. Historical source and target descriptions
  appear as comparison context, but are not supplied as a source/target pair to
  InstructPix2Pix. The negative prompt is empty.
- One output per case: seed 42, 50 steps, text guidance 7.5, image guidance 1.5.
  U-Net, text encoder, VAE and safety checker use CUDA FP16. The stock pipeline
  performs scheduler arithmetic. This differs from the custom FP32 VAE/updates.
- The stock three-branch image/text guidance should perform 50 U-Net forwards and
  150 batch elements per image, with no inversion, calibration or backward pass.
  Actual counts and visited timestep values are recorded and checked. Previous
  custom/calibration work has different budgets and is not presented as matched.
- Requested masks select evaluation regions and boundary crops only. They are not
  used for generation, blending or final pixel compositing.

## Evidence and stopping rule

The baseline retains decoded floating RGB, PNGs, final latents, literal instruction
tokens, model/config hashes, software/device information, runtime, peak allocated
memory, actual U-Net work, safety flags and per-output reports. Comparisons and
boundary supplements show the generation instruction and historical prompt
context. PSNR, SSIM and requested-background MAE quantify preservation only.

There is no automatic semantic grade. The three independent human dimensions
remain semantic success, unrelated-content preservation and boundary quality.
New review rows start unreviewed; prior scores are never overwritten. Any safety
flag, numerical failure or missing artifact must remain visible in the report.

Generate exactly the two registered outputs before assessing them. Retain failures;
do not hide prompt/seed sweeps or select per-case winners. If results are useful,
the next proposal must distinguish recoloring success from structural preservation
and separately register any wider development study. Two successful examples
would still not pass the twelve-case acceptance protocol.

## Local execution and current status

The local experiment directory is
`outputs/acceptance/2026-11-16-side-validation/development-round-10-instruct-pix2pix-reference/`.
It contains `plan.json`, frozen source, the baseline-only `run_reference.py` and
six offline contract tests. The output tree is ignored by Git and is not promised
to exist in a clean checkout.

Registration and all six offline tests passed. Checks cover frozen identities,
overwrite rejection, selective offline/pinned loading, preserved safety checking,
frozen model parameters, tamper rejection, three-branch work accounting and honest
infinite-metric representation. Mocks do not prove real-weight compatibility.

At preparation time, only the original SD 1.5 weights were cached. The user then
explicitly authorized the additional download and two GPU runs; both completed.
The public metadata request initially
encountered an expired saved Hugging Face token; an unauthenticated public request
succeeded, without changing login or account settings.

The completed commands, run from the repository root after authorization, were:

```powershell
$reference = 'outputs/acceptance/2026-11-16-side-validation/development-round-10-instruct-pix2pix-reference/run_reference.py'
.venv\Scripts\python.exe $reference download --authorized-download
.venv\Scripts\python.exe $reference run --run-gpu
```

The download command fetches only the registered files at the fixed revision.
Inference is cache-only. It rejects changed inputs/code, missing weights,
unexpected architecture/scheduler and existing execution directories. A failed
attempt is retained; rerunning with changed settings requires a separately
registered attempt, rather than overwriting evidence.

## Executed result

The apple is visibly green and the car body visibly yellow. Unrelated changes
remain: the apple background/contact region becomes green, while the car's dark
roof, windows/trim, wheel details and foliage also change. This is direct evidence
that another pretrained editing route can recolor these inputs; it does not
isolate a model-only cause or establish semantic acceptance. Human scores remain
blank. The source convergence gate, default editor and holdout remain unchanged.

| Input | PSNR vs original | SSIM | Requested-background MAE | U-Net forwards / batch elements | Runtime |
|---|---:|---:|---:|---:|---:|
| Apple | 12.4486 dB | 0.715619 | 0.065615 | 50 / 150 | 4.365 s |
| Car | 11.7893 dB | 0.457481 | 0.107070 | 50 / 150 | 3.086 s |

Total new inference used 100 forwards / 300 batch elements with no inversion,
calibration or backward pass. Runtime includes inference and safety checking,
excluding loading/hashing/artifact work; these are single runs. Peak allocated
CUDA memory was 3.125/3.031 GiB. Both safety flags were false. Pixel metrics
cannot establish color success. Historical calibrated targets use different
reverse and preparation budgets, so this is not a compute-matched comparison.

The completed evidence audit verified 24 recorded artifacts, frozen code/inputs
and prior output hashes, actual work, finite final latent shapes, unchanged
source-selected crop boxes and two unreviewed rating rows. The six offline tests
were rerun and passed. Start with `FINDINGS.md` and `REVIEW.md` in the local
experiment directory; raw results, exact instructions, floating tensors, weight
hashes, boundary enlargements and per-case reports are retained there.

A separate follow-up, `development-round-11-localization/`, compared fixed raw
outputs with hard-mask and inward-feather composites. Both restore requested-mask
exterior pixels exactly, while red apple rims and blue car strips remain. Seven
CPU tests pass; four human rows remain unreviewed. Browser verification did not
complete because the tool could not reliably identify the current Chrome URL.
No model inference or backward work was added by that postprocessing study.

The user subsequently selected the [custom DDIM route](ddim-research-plan.md) as
the active research direction. Preserve these references and failures; further
pretrained-editor or composite polishing is no longer the next deliverable.
