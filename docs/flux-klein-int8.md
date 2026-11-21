# Distilled FLUX.2 Klein 4B INT8 reference

The selected FLUX inference target is `black-forest-labs/FLUX.2-klein-4B`,
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`. It replaces Base 4B
for this standalone reference. Original Base files and the custom SD1.5 editor
remain available. The model ships under Apache 2.0.

Configuration: `configs/flux-klein-4b-int8.json`. The distilled model uses four
sampling steps and guidance 1. Both Transformer and Qwen3 linear layers use
bitsandbytes INT8. Transformer time conditioning, shared modulation, input/output
projections and the final normalization projection remain BF16, as explicitly
listed in `transformer_skip_modules`. Other floating-point layers, including
embeddings and the VAE, remain BF16. INT8 is not an all-tensor eight-bit conversion.

## Runtime and preparation

The project's pinned SD1.5 environment does not provide FLUX.2 or Qwen3.
Use a separate compatible runtime; do not upgrade `.venv` in place. The existing
adjacent FLUX runtime can be used without changing its installed packages:

```powershell
$fluxPython = '..\FLUX\.venv\Scripts\python.exe'
& $fluxPython -B scripts/prepare_flux_klein_int8.py
& $fluxPython -B scripts/run_flux_klein_int8_reference.py --run-gpu
```

If a full-file CDN transfer stalls, preparation also supports resumable bounded
HTTP ranges: `scripts/prepare_flux_klein_int8.py --http-range`. The complete file
must still pass the pinned upstream SHA-256 check before it is usable.

Preparation downloads only the pinned Diffusers component layout. Identical
existing Base files are verified against upstream hashes and hardlinked into
`models/FLUX.2-klein-4B`; never edit shared weights in place. The distilled
Transformer is distinct from the Base Transformer. A local source manifest
records file hashes and provenance. No existing Base file is replaced or deleted.

The runner is offline and requires an explicit GPU opt-in. It exports quantized
components to `models/FLUX.2-klein-4B-int8-protected` and verifies INT8 layers after
reloading. It computes prompt embeddings with the INT8 text encoder, releases
that encoder before loading the Transformer, and uses component CPU offload and
VAE tiling for generation. This requires a compatible NVIDIA
GPU, sufficient host memory and free disk space. The first run includes export;
subsequent runs reuse exported components. CUDA allocated/reserved peaks exclude
other processes, display allocations and some driver memory.

The available runtime currently reports Python 3.12.13, PyTorch 2.11.0+cu128,
Diffusers 0.40.0.dev0, Transformers 5.15.0, bitsandbytes 0.50.1 and Accelerate
1.14.0. The runner records actual installed versions with each output. BF16 is
the floating-point module dtype; bitsandbytes INT8 kernels may internally cast
activations to FP16. No package installation into either environment is performed
by these scripts.

Optional instruction editing:

```powershell
& $fluxPython -B scripts/run_flux_klein_int8_reference.py --run-gpu `
  --image assets/input.png --prompt 'Change the mug to blue.'
```

This is a labelled built-in pipeline reference, not an implementation of the
project's custom DDIM algorithm, masked inpainting or outpainting. Its output
does not establish semantic acceptance. Each run saves an image and JSON with
parameters, revision, software, device, timing, quantized layer counts, output
hash and CUDA memory peaks under `outputs/flux-klein-int8/`.

## Verification results

The pinned source files passed upstream SHA-256 or Git blob hash checks. The
active export contains 100 INT8 Transformer linear layers and 252 INT8 Qwen3
linear layers. The three weight files total **8.064 GiB**. Their hashes and tensor
dtypes are recorded in the local `quantization_manifest.json`.

On the RTX 4080 Laptop GPU, a 512x512 text-to-image smoke used four steps,
guidance 1 and seed 42. The literal prompt was:

> A photograph of a red ceramic mug on a wooden table, soft daylight, a plain neutral background.

| Run | Configuration | Observed result | CUDA allocated peak |
| --- | --- | --- | --- |
| `diagnostic-bf16` | Distilled BF16 reference | A red mug is visible | 7.803 GiB |
| `20261119-073056` | All-linear INT8, both components | Abnormal texture; rejected | 11.185 GiB |
| `diagnostic-int8-transformer` | All-linear INT8 Transformer, BF16 text encoder | Abnormal texture; rejected | See local run JSON |
| `20261119-074245` | Protected INT8, both components loaded together | A red mug is visible | 11.004 GiB |
| `20261119-074610` | Protected INT8, sequential text-encoder lifecycle | A red mug is visible | **7.407 GiB** |

The final run reserved at most **8.156 GiB** in the CUDA allocator. Its output
hash matches the preceding successful protected INT8 run exactly in this runtime.
Sampling, decoding and PNG saving took 2.995 seconds after prompt encoding and
model loading. Total runner time after imports and initial validation was 13.112
seconds with existing exports. These timings are observations, not a matched
performance benchmark or a guarantee at other resolutions.

Keeping the listed control/projection groups in BF16 restored a recognizable
smoke output. This group intervention does not establish which individual layer
needs protection. The rejected all-linear export remains under
`models/FLUX.2-klein-4B-int8`, explicitly marked `VALIDATION_FAILED.json`.

Two exploratory mixed-precision attempts exited before saving output and without
a Python traceback: protected INT8 Transformer with BF16 text encoding, and
INT8 text encoding with BF16 Transformer. Their cause is unconfirmed. Incomplete
attempt records are retained; they do not establish a quality result.

The scripts also passed syntax checks, lightweight CLI help checks, the explicit
GPU opt-in guard and rejection of a Base configuration. Unit-like CLI checks
used the existing SD environment without importing model dependencies. No model
packages were installed or upgraded in either environment.

This smoke establishes local execution and a visually recognizable single
example. It does not pass the twelve-case semantic acceptance protocol, establish
general quality parity with BF16, or validate masked inpainting/outpainting.
The existing SD1.5 Gradio routes retain their custom algorithms. Gradio now also
exposes this reference through a separate FLUX page; see `flux-gradio.md`.

## Repository evidence and local model cache

The portable [evidence archive](results/flux-klein-int8/README.md) retains source
and quantization manifests, protected/rejected validation records, original smoke
outputs and incomplete attempt records with file hashes. These files were copied
from existing local evidence without rerunning or changing the experiments.

The root `models/` directory is a local cache and is excluded from Git, including
downloaded tokenizer/configuration copies and weight files. This leaves original
and rejected exports on disk while avoiding cache duplication in the repository.
Fresh installations use the pinned preparation scripts; the evidence archive is
not an inference checkpoint.
