"""Controlled cached-model reconstruction probes; does not change production defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from diffusion_editor.config import load_config
from diffusion_editor.utils.logging import configure_logging

CASES = {"baseline50": (50, "current", "fp16"), "destination50": (50, "destination", "fp16"),
         "current100": (100, "current", "fp16"), "current200": (200, "current", "fp16"),
         "fp3250": (50, "current", "fp32"), "refined50": (50, "refined", "fp16")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", type=Path, default=Path("assets/examples/astronaut.png"))
    parser.add_argument("--source-prompt", default="a portrait of an astronaut wearing a spacesuit")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--local-residuals", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/reconstruction"))
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusion_editor.diffusion.inversion import ddim_inversion_step
    from diffusion_editor.diffusion.sampler import DDIMSampler, ddim_sampling_step
    from diffusion_editor.diffusion.schedule import DDIMSchedule
    from diffusion_editor.diffusion.refinement import refine_inversion_step
    from diffusion_editor.evaluation.metrics import reconstruction_metrics
    from diffusion_editor.models.sd_backend import SDBackend, resolve_runtime
    from diffusion_editor.models.vae import VAEWrapper
    from diffusion_editor.models.text_encoder import TextEncoderWrapper
    from diffusion_editor.models.noise_predictor import UNetNoisePredictor
    from diffusion_editor.utils.environment import inspect_environment
    from diffusion_editor.utils.image_utils import image_to_tensor, tensor_to_images
    from diffusion_editor.utils.visualization import save_image_grid

    config = load_config(args.config)
    if not config.local_files_only:
        parser.error("diagnostics require cached weights")
    configure_logging(config.log_level)
    if resolve_runtime(config, torch)[0] != "cuda":
        parser.error("these FP16/FP32 precision probes require CUDA; offline unit tests remain CPU-compatible")
    torch.manual_seed(config.seed)
    backend = SDBackend.load(config)
    vae = VAEWrapper(backend.vae)
    with Image.open(args.image) as image:
        original = image_to_tensor(image)
    clean = vae.encode_image(original)
    vae_image = vae.decode_latents(clean)
    embeddings = TextEncoderWrapper(backend.tokenizer, backend.text_encoder).encode_prompt(args.source_prompt)
    predictor = UNetNoisePredictor(backend.unet)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, images, labels = [], [original, vae_image], ["Original", "VAE only"]
    for name in args.cases:
        steps, query, precision = CASES[name]
        # FP32 probe widens the SAME FP16-quantized weights; no new checkpoint.
        backend.unet.to(dtype=torch.float32 if precision == "fp32" else torch.float16)
        schedule = DDIMSchedule.from_scheduler(backend.scheduler, steps)
        levels = schedule.inversion_timesteps
        coefficients = {t: schedule.alpha(t, device=clean.device) for t in levels}
        if clean.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        states, queries, refinements = {-1: clean.cpu().clone()}, [], []
        z = clean.clone()
        with torch.inference_mode():
            for t, s in zip(levels, levels[1:]):
                model_t = s if query == "destination" or t == -1 else t
                eps = predictor(z, model_t, embeddings)
                source = z
                z = ddim_inversion_step(source, eps, coefficients[t], coefficients[s])
                if query == "refined":
                    z, diagnostics = refine_inversion_step(source, z, coefficients[t], coefficients[s],
                                                           s, predictor, embeddings, 1,
                                                           max_iterations=12, tolerance=1e-6)
                    from dataclasses import asdict
                    refinements.append({"timestep": s, **asdict(diagnostics)})
                states[s] = z.cpu().clone()
                queries.append(model_t)
            sampled = DDIMSampler(schedule, predictor).sample(z, embeddings)
            reconstructed = vae.decode_latents(sampled.final_latent)
        if clean.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        row = {"name": name, "steps": steps, "query": query, "unet_compute_dtype": precision,
               "guidance_scale": 1, "timesteps": levels, "prediction_timesteps": queries,
               "original_metrics": reconstruction_metrics(original, reconstructed),
               "vae_metrics": reconstruction_metrics(vae_image, reconstructed),
               "latent_mse": float((sampled.final_latent - clean).square().mean()), "runtime_seconds": elapsed}
        if query == "refined":
            row.update(refinement_max_iterations=12, refinement_tolerance=1e-6, refinement_diagnostics=refinements)
        if args.local_residuals:
            local = []
            with torch.inference_mode():
                for t, s in zip(levels, levels[1:]):
                    noisy = states[s].to(clean.device)
                    eps = predictor(noisy, s, embeddings)
                    recovered = ddim_sampling_step(noisy, eps, coefficients[s], coefficients[t])
                    delta = recovered.cpu() - states[t]
                    local.append({"from": s, "to": t, "mse": float(delta.square().mean()),
                                  "max_abs": float(delta.abs().max())})
            row["teacher_forced_one_step_residuals"] = local
        rows.append(row)
        images.append(reconstructed)
        labels.append(name)
        tensor_to_images(reconstructed)[0].save(args.output_dir / f"{name}.png")
        report = {"config": config.to_dict(), "image": str(args.image),
                  "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  "source_prompt": args.source_prompt, "vae_only": reconstruction_metrics(original, vae_image),
                  "fp32_probe_note": "FP32 arithmetic on the same FP16-quantized UNet weights; CLIP embeddings held fixed",
                  "cases": rows, "environment": inspect_environment()}
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        save_image_grid(images, labels, args.output_dir / "comparison.png", columns=3)
        print(json.dumps({key: row[key] for key in ("name", "original_metrics", "vae_metrics", "latent_mse", "runtime_seconds")}), flush=True)


if __name__ == "__main__":
    main()
