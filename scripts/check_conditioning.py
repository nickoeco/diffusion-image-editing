"""Opt-in real-weight VAE/text validation and a saved reconstruction comparison."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from diffusion_editor.config import load_config
from diffusion_editor.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--image", type=Path, help="Use an image with dimensions divisible by the VAE spatial factor")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction/phase1"))
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from diffusion_editor.models.sd_backend import SDBackend
    from diffusion_editor.models.text_encoder import TextEncoderWrapper
    from diffusion_editor.models.vae import VAEWrapper
    from diffusion_editor.utils.image_utils import image_to_tensor, save_comparison, tensor_to_images
    from diffusion_editor.utils.environment import inspect_environment

    config = load_config(args.config)
    if args.allow_download:
        config = replace(config, local_files_only=False)
    configure_logging(config.log_level)
    started = time.perf_counter()
    backend = SDBackend.load(config)
    vae = VAEWrapper(backend.vae)
    text = TextEncoderWrapper(backend.tokenizer, backend.text_encoder)
    if args.image:
        with Image.open(args.image) as opened:
            original = image_to_tensor(opened)
        provenance = str(args.image)
    else:
        # A deterministic synthetic test card; no external image or hidden dataset.
        y, x = np.mgrid[0:512, 0:512]
        rgb = np.stack((x / 511, y / 511, ((x // 32 + y // 32) % 2)), axis=-1)
        image = Image.fromarray((rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.ellipse((128, 128, 384, 384), fill=(210, 40, 35))
        draw.rectangle((210, 92, 300, 144), fill=(30, 140, 40))
        original = image_to_tensor(image)
        provenance = "Generated deterministic 512x512 gradient/checkerboard test card with red circle; no external assets"
    latent = vae.encode_image(original)
    repeated = vae.encode_image(original)
    torch.testing.assert_close(latent, repeated, rtol=0, atol=0)
    reconstructed = vae.decode_latents(latent)
    prompts = ["a red apple on a colorful background", "a blue apple on a colorful background"]
    embeddings = text.encode_prompt_pair(prompts)
    # Compare against direct pretrained execution at the SAME batch size. FP16
    # kernels can give different rounding for a split batch; that is not an order bug.
    tokens = backend.tokenizer(["", ""] + prompts, padding="max_length",
                               max_length=backend.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
    kwargs = ({"attention_mask": tokens.attention_mask.to(backend.device)}
              if getattr(backend.text_encoder.config, "use_attention_mask", False) else {})
    with torch.inference_mode():
        reference = backend.text_encoder(tokens.input_ids.to(backend.device), **kwargs).last_hidden_state
    torch.testing.assert_close(embeddings, reference, rtol=0, atol=0)
    split = torch.cat((text.encode_prompt(["", ""]), text.encode_prompt(prompts)))
    batch_difference = (embeddings - split).abs().max().item()
    assert reconstructed.shape == original.shape
    if backend.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_images(original)[0].save(args.output_dir / "original.png")
    tensor_to_images(reconstructed)[0].save(args.output_dir / "vae_reconstruction.png")
    save_comparison(original, reconstructed, args.output_dir / "comparison.png")
    report = {"config": config.to_dict(), "image_source": provenance,
              "image_shape": list(original.shape), "latent_shape": list(latent.shape),
              "embedding_shape": list(embeddings.shape), "vae_scaling_factor": vae.scaling_factor,
              "vae_spatial_factor": vae.spatial_factor, "vae_dtype": str(latent.dtype),
              "embedding_dtype": str(embeddings.dtype), "device": backend.device,
              "posterior": "mode", "prompts": prompts, "negative_prompt": "",
              "cfg_batch_order": "unconditional batch, conditional batch",
              "deterministic_repeat": "passed", "cfg_order_check": "passed",
              "embedding_split_batch_max_abs_difference": batch_difference,
              "runtime_seconds_including_model_load": elapsed, "environment": inspect_environment()}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
