"""Opt-in component loading and one real U-Net forward pass (not DDIM)."""
import argparse
from dataclasses import replace

from diffusion_editor.config import load_config
from diffusion_editor.models.sd_backend import SDBackend
from diffusion_editor.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--allow-download", action="store_true", help="May download several GB of model weights")
    parser.add_argument("--forward", action="store_true", help="Execute text encoder and U-Net")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.allow_download:
        config = replace(config, local_files_only=False)
    configure_logging(config.log_level)
    backend = SDBackend.load(config)
    if args.forward:
        import torch
        generator = torch.Generator(device=backend.device).manual_seed(config.seed)
        with torch.inference_mode():
            tokens = backend.tokenizer("a red apple on a wooden table", padding="max_length",
                                       max_length=backend.tokenizer.model_max_length,
                                       truncation=True, return_tensors="pt")
            kwargs = ({"attention_mask": tokens.attention_mask.to(backend.device)}
                      if getattr(backend.text_encoder.config, "use_attention_mask", False) else {})
            context = backend.text_encoder(tokens.input_ids.to(backend.device), **kwargs)[0]
            size = backend.unet.config.sample_size
            spatial = (size, size) if isinstance(size, int) else tuple(size)
            latent = torch.randn((1, backend.unet.config.in_channels, *spatial),
                                 generator=generator, device=backend.device, dtype=backend.dtype)
            noise = backend.unet(latent, backend.scheduler.config.num_train_timesteps - 1,
                                 encoder_hidden_states=context).sample
            if noise.shape != latent.shape or not torch.isfinite(noise).all():
                raise RuntimeError("U-Net forward produced invalid noise prediction")
            print(f"Forward OK: shape={tuple(noise.shape)}, dtype={noise.dtype}, device={noise.device}")
    print("Component loading OK; this does not verify DDIM inversion or reconstruction.")


if __name__ == "__main__":
    main()
