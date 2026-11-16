"""Shared experiment artifacts and measured inference budgets (no pipeline shortcut)."""
from contextlib import contextmanager
from collections.abc import Iterator
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from diffusion_editor.evaluation.experiment_config import file_hash
from diffusion_editor.evaluation.metrics import background_metrics, reconstruction_metrics
from diffusion_editor.evaluation.records import write_json
from diffusion_editor.utils.image_utils import tensor_to_images
from diffusion_editor.utils.visualization import save_image_grid

METRIC_DEFINITIONS = {
    "whole_image": "RGB float [0,1]; PSNR data_range=1; SSIM uniform 7x7, channel mean, sample covariance.",
    "background": "Weights 1-M; denominator 3*sum(1-M). zero_mask_mae divides by 3*count(M==0). Empty region undefined.",
    "requested_vs_effective": "Requested mask is the fixed evaluation region across modes; effective mask includes feathering.",
    "null": "Infinite scores become null in JSON; nonfinite_metrics disambiguates infinity. Empty regions are null without infinity flag.",
    "meaning": "PSNR/SSIM/MAE measure preservation, not prompt alignment. No zero-background perceptual metrics.",
}


@contextmanager
def count_unet(model: torch.nn.Module) -> Iterator[dict[str, int]]:
    """Count actual forwards and batch elements, including CFG's doubled batch."""
    counts = {"unet_calls": 0, "unet_batch_elements": 0}
    def before(module, args):
        counts["unet_calls"] += 1
        counts["unet_batch_elements"] += args[0].shape[0]
    handle = model.register_forward_pre_hook(before)
    try:
        yield counts
    finally:
        handle.remove()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def calculate_metrics(original: torch.Tensor, images: dict[str, torch.Tensor],
                      requested_mask: torch.Tensor | None = None,
                      effective_mask: torch.Tensor | None = None) -> dict[str, float | int | None]:
    """Single RGB BCHW [0,1] images; masks B1HW, measured before quantization."""
    metrics = {}
    for name, image in images.items():
        if name != "original" and "mask" not in name:
            metrics.update({f"{name}_{key}": value for key, value in reconstruction_metrics(original, image).items()})
    metrics.update({f"output_vs_vae_{key}": value for key, value in reconstruction_metrics(images["vae"], images["output"]).items()})
    for region, mask in (("requested", requested_mask), ("effective", effective_mask)):
        if mask is not None:
            for stage in ("output", "uncomposited"):
                if stage in images:
                    metrics.update({f"{stage}_{region}_{key}": value for key, value in background_metrics(original, images[stage], mask).items()})
    return metrics


def save_case(directory: Path, images: dict[str, torch.Tensor], metrics: dict[str, float | int | None],
              report: dict[str, Any], *, masks: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    """Metrics use tensors before PNG quantization; hashes bind saved artifacts."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in images.items():
        tensor_to_images(value)[0].save(directory / f"{name}.png")
    save_image_grid(list(images.values()), [name.replace("_", " ") for name in images], directory / "comparison.png")
    if masks:
        torch.save({name: value.detach().cpu() for name, value in masks.items()}, directory / "masks.pt")
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(metrics.items())
    report = dict(report, status="complete", metrics=metrics, metric_definitions=METRIC_DEFINITIONS,
                  nonfinite_metrics={key: "positive_infinity" if value > 0 else "negative_infinity"
                                     for key, value in metrics.items() if isinstance(value, float) and not math.isfinite(value)})
    report["artifacts"] = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p.name not in ("report.json", "error.json")}
    write_json(directory / "report.json", report)
    return json.loads((directory / "report.json").read_text(encoding="utf-8"))
