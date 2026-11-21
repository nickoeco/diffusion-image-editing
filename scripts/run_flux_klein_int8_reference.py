"""Explicit GPU-only built-in FLUX reference; separate from custom DDIM editing.

Quantize Transformer and Qwen3 linear layers to bitsandbytes INT8, retain BF16
for other floating-point layers, and offload components between inference stages.
No downloads are allowed. Exported checkpoints are reused on subsequent runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flux-klein-4b-int8.json")
    parser.add_argument("--run-gpu", action="store_true", help="Explicitly opt into real-weight GPU inference")
    parser.add_argument("--prompt", help="Override the configured generation or editing prompt")
    parser.add_argument("--image", type=Path, help="Optional reference image for instruction editing")
    parser.add_argument("--run-dir", type=Path, help="Write one run to this new directory")
    parser.add_argument("--progress-json", action="store_true", help="Emit structured progress for the local UI")
    parser.add_argument("--require-export", action="store_true", help="Use existing exports without modifying model files")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("Real-weight checks require --run-gpu")
    if args.image and not args.image.is_file():
        parser.error(f"Reference image does not exist: {args.image}")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if (cfg["is_distilled"] is not True or cfg["num_inference_steps"] != 4
            or cfg["guidance_scale"] != 1.0 or cfg["quantization"] != "bitsandbytes-int8"):
        raise ValueError("This reference requires distilled Klein 4B, four steps, guidance=1 and INT8")

    def progress(fraction: float, message: str) -> None:
        if args.progress_json:
            print("FLUX_PROGRESS " + json.dumps({"fraction": fraction, "message": message}), flush=True)

    progress(0.02, "Starting the isolated FLUX runtime")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/flux-int8/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/flux-int8/inductor"))
    import gc
    import torch
    import bitsandbytes as bnb
    from diffusers import BitsAndBytesConfig, Flux2KleinPipeline, Flux2Transformer2DModel
    from transformers import BitsAndBytesConfig as TextQuantConfig, Qwen3ForCausalLM
    from PIL import Image

    class CastNoticeOnce(logging.Filter):
        """Keep the expected INT8 activation-cast notice without repeating it."""
        seen = False

        def filter(self, record: logging.LogRecord) -> bool:
            if record.getMessage().startswith("MatMul8bitLt: inputs will be cast"):
                if self.seen:
                    return False
                self.seen = True
            return True

    logging.getLogger("bitsandbytes.autograd._functions").addFilter(CastNoticeOnce())

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This reference requires a CUDA GPU supporting BF16")
    source = ROOT / cfg["source_dir"]
    quantized = ROOT / cfg["quantized_dir"]
    manifest = json.loads((source / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != cfg["model_id"] or manifest["revision"] != cfg["revision"]:
        raise ValueError("Prepared model does not match the configured distilled revision")
    if json.loads((source / "model_index.json").read_text()).get("is_distilled") is not True:
        raise ValueError("Refusing to run a Base checkpoint as a distilled model")
    quantized.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    counts = {}
    for name, cls, quant_cls in (("transformer", Flux2Transformer2DModel, BitsAndBytesConfig),
                                 ("text_encoder", Qwen3ForCausalLM, TextQuantConfig)):
        destination = quantized / name
        ready = destination / "export_complete.json"
        skip = cfg.get("transformer_skip_modules", []) if name == "transformer" else []
        if ready.exists():
            previous = json.loads(ready.read_text())
            if (previous["revision"] != cfg["revision"] or previous["bits"] != 8
                    or previous.get("skip_modules", []) != skip):
                raise ValueError(f"Incompatible quantized checkpoint: {destination}")
            counts[name] = previous["int8_linear_layers"]
            continue
        if args.require_export:
            raise ValueError(f"Prepared INT8 export missing: {destination}. Prepare the local export first.")
        print(f"Quantizing {name} to INT8...", flush=True)
        model = cls.from_pretrained(
            source, subfolder=name, local_files_only=True, torch_dtype=torch.bfloat16,
            quantization_config=quant_cls(load_in_8bit=True, llm_int8_skip_modules=skip or None), device_map={"": 0},
        )
        count = sum(isinstance(layer, bnb.nn.Linear8bitLt) for layer in model.modules())
        if not count:
            raise RuntimeError(f"INT8 conversion did not create quantized layers in {name}")
        model.save_pretrained(destination, safe_serialization=True)
        counts[name] = count
        ready.write_text(json.dumps({"model_id": cfg["model_id"], "revision": cfg["revision"],
                                     "bits": 8, "int8_linear_layers": count,
                                     "skip_modules": skip}, indent=2) + "\n")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    # The UI uses an immutable prepared export; only preparation may create links.
    for directory in (() if args.require_export else ("vae", "tokenizer", "scheduler")):
        for original in (source / directory).rglob("*"):
            if original.is_file():
                destination = quantized / original.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    os.link(original, destination)
    for filename in (() if args.require_export else ("model_index.json", "LICENSE.md")):
        if not (quantized / filename).exists():
            os.link(source / filename, quantized / filename)
    previous_manifest = quantized / "quantization_manifest.json"
    export_metadata = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    export_metadata.update({
        "model_id": cfg["model_id"], "revision": cfg["revision"], "is_distilled": True,
        "quantization": cfg["quantization"], "int8_linear_layers": counts,
        "transformer_skip_modules": cfg.get("transformer_skip_modules", []),
        "unquantized_floating_point_dtype": "bfloat16",
        "software": {p: version(p) for p in ("torch", "diffusers", "transformers", "bitsandbytes")},
    })
    if not args.require_export:
        previous_manifest.write_text(json.dumps(export_metadata, indent=2) + "\n", encoding="utf-8")
    if not args.require_export and not (quantized / "README.md").exists():
        (quantized / "README.md").write_text(
            "# Local distilled FLUX.2 Klein 4B INT8 export\n\n"
            "Transformer and Qwen3 linear layers use bitsandbytes INT8. Other floating-point "
            "layers and VAE use BF16. This is not the Base or a four-bit checkpoint.\n\n"
            "See quantization_manifest.json for upstream revision and runtime versions. "
            "The upstream Apache 2.0 license is included in LICENSE.md. "
            "Run scripts/run_flux_klein_int8_reference.py --run-gpu from the project root "
            "using the documented compatible runtime.\n", encoding="utf-8")
    print("Encoding the prompt with the exported INT8 text encoder...", flush=True)
    progress(0.10, "Loading the INT8 text encoder")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        quantized / "text_encoder", local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    def check_int8(name: str, model: torch.nn.Module) -> None:
        actual = sum(isinstance(layer, bnb.nn.Linear8bitLt) for layer in model.modules())
        if actual != counts[name] or not actual:
            raise RuntimeError(f"Reloaded INT8 layer count differs in {name}: {actual}")
        model.eval().requires_grad_(False)

    check_int8("text_encoder", text_encoder)
    prompt = args.prompt or cfg["prompt"]
    conditioning = Flux2KleinPipeline.from_pretrained(
        source, transformer=None, vae=None, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    with torch.inference_mode():
        prompt_embeddings, _ = conditioning.encode_prompt(prompt=prompt, device=torch.device("cuda"))
    progress(0.30, "Prompt encoded; releasing the text encoder")
    del conditioning, text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("Text encoder released; loading the exported INT8 Transformer...", flush=True)
    progress(0.40, "Loading the INT8 Transformer")
    transformer = Flux2Transformer2DModel.from_pretrained(
        quantized / "transformer", local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    check_int8("transformer", transformer)
    pipe = Flux2KleinPipeline.from_pretrained(
        source, transformer=transformer, text_encoder=None,
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.eval().requires_grad_(False)
    pipe.vae.enable_tiling()
    extra = {}
    if args.image:
        with Image.open(args.image) as original:
            extra["image"] = original.convert("RGB")
    output = args.run_dir.resolve() if args.run_dir else ROOT / cfg["output_dir"] / time.strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    if args.image:
        extra["image"].save(output / "input.png")
    inference_start = time.perf_counter()

    def on_step(_pipeline, index, _timestep, tensors):
        progress(0.55 + 0.35 * (index + 1) / cfg["num_inference_steps"],
                 f"Sampling step {index + 1} / {cfg['num_inference_steps']}")
        return tensors

    progress(0.55, "Generating the image")
    with torch.inference_mode():
        result = pipe(prompt_embeds=prompt_embeddings, height=cfg["height"], width=cfg["width"],
                      num_inference_steps=cfg["num_inference_steps"], guidance_scale=cfg["guidance_scale"],
                      generator=torch.Generator(device="cpu").manual_seed(cfg["seed"]),
                      callback_on_step_end=on_step, **extra)
    torch.cuda.synchronize()
    image_path = output / "output.png"
    result.images[0].save(image_path)
    report = {**cfg, "label": "built-in reference; not the custom DDIM editor",
              "run_directory": str(output),
              "prompt": prompt, "input_image": str(args.image.resolve()) if args.image else None,
              "saved_input_sha256": hashlib.sha256((output / "input.png").read_bytes()).hexdigest() if args.image else None,
              "int8_linear_layers": counts, "gpu": torch.cuda.get_device_name(),
              "text_encoder_lifecycle": "INT8 prompt encoding, then released before Transformer loading",
              "inference_seconds": time.perf_counter() - inference_start,
              "total_seconds": time.perf_counter() - started,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
              "output_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "software": {p: version(p) for p in ("torch", "diffusers", "transformers", "bitsandbytes", "accelerate")}}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress(1.0, "Image and run report saved")
    print(json.dumps({"output": str(image_path), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
