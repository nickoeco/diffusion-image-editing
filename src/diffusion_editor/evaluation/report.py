"""Rebuild tables, PNG/SVG plots and grids from completed measured artifacts."""
import csv
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _grid(paths: list[Path], labels: list[str], target: Path, columns=3):
    cells = []
    for path in paths:
        with Image.open(path) as image:
            cells.append(image.convert("RGB"))
    width, height = cells[0].size
    if any(cell.size != (width, height) for cell in cells):
        raise ValueError("Experiment grid requires equal-size saved images")
    columns = min(columns, len(cells))
    canvas = Image.new("RGB", (width * columns, (height + 44) * math.ceil(len(cells) / columns)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (cell, label) in enumerate(zip(cells, labels)):
        x, y = index % columns * width, index // columns * (height + 44)
        draw.multiline_text((x + 8, y + 5), "\n".join(textwrap.wrap(label, max(12, width // 7)))[:160], fill="black")
        canvas.paste(cell, (x, y + 44))
    canvas.save(target)


def summarize(output_dir: str | Path) -> dict:
    """Read reports only; never load weights or recompute metrics from PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffusion_editor.evaluation.records import validate_completed, write_json
    from diffusion_editor.evaluation.experiment_config import fingerprint

    root = Path(output_dir).resolve()
    manifest = _read(root / "manifest.json")
    reports, missing = [], []
    for case in manifest["planned_cases"]:
        expected = fingerprint({"manifest": manifest["fingerprint"], "case": case})
        report = validate_completed(root / "runs" / case["id"], expected)
        if report is None:
            missing.append(case["id"])
        else:
            reports.append(report)
    summaries = root / "summary"
    summaries.mkdir(exist_ok=True)
    rows = []
    for report in reports:
        case = report["case"]
        rows.append({"case_id": case["id"], "study": case["study"], "sample": case["sample"],
                     "axis": case["axis"], "value": case["value"], "method": report["method"],
                     "runtime_seconds": report["runtime_seconds"], **report["budget"], **report["metrics"]})
    if rows:
        names = list(rows[0]) + sorted(set().union(*(set(row) for row in rows)) - set(rows[0]))
        with (summaries / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
    studies = {}
    for study in manifest["suite"]["studies"]:
        selected = [r for r in reports if r["case"]["study"] == study["id"]]
        if not selected:
            continue
        studies[study["id"]] = {"completed": len(selected), "planned": len(study["samples"]) * len(study["values"])}
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        panel_metrics = (("output_psnr_db", "Preservation PSNR (dB)"), ("output_ssim", "Preservation SSIM"),
                         ("runtime_seconds", "Pipeline runtime (s)"),
                         ("output_requested_background_mae", "Fixed requested-region background MAE"))
        for sample in study["samples"]:
            group = [r for r in selected if r["case"]["sample"] == sample]
            if not group:
                continue
            categorical = study["axis"] == "mask_mode"
            positions = [study["values"].index(r["case"]["value"]) if categorical else r["case"]["value"] for r in group]
            for ax, (key, title) in zip(axes.flat, panel_metrics):
                values = [r["runtime_seconds"] if key == "runtime_seconds" else r["metrics"].get(key) for r in group]
                if any(value is not None for value in values):
                    ax.plot(positions, [value if value is not None and math.isfinite(value) else math.nan for value in values],
                            marker="o", label=sample)
                if study["operation"] == "reconstruct" and key in ("output_psnr_db", "output_ssim"):
                    ax.axhline(group[0]["metrics"][key.replace("output", "vae", 1)], linestyle="--", alpha=0.5,
                               label=f"{sample} VAE only")
                ax.set(title=title, xlabel=study["axis"])
                ax.set_xticks(range(len(study["values"])) if categorical else study["values"],
                              [str(v).replace("_", "\n") for v in study["values"]])
                ax.grid(alpha=0.25)
            paths = [root / "runs" / group[0]["case"]["id"] / name for name in ("original.png", "vae.png")]
            labels = ["Original", "VAE-only baseline"]
            for report in group:
                paths.append(root / "runs" / report["case"]["id"] / "output.png")
                labels.append(f"{study['axis']}={report['case']['value']}; {report['method']}")
            _grid(paths, labels, summaries / f"{study['id']}--{sample}.png")
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No defined region/metric", ha="center", transform=ax.transAxes)
        figure.suptitle(f"{study['id']} — single-variable study; preservation is not prompt alignment\nInfinite/undefined points omitted; timings include source control where applicable", fontsize=11)
        figure.savefig(summaries / f"{study['id']}.png", dpi=160)
        figure.savefig(summaries / f"{study['id']}.svg")
        plt.close(figure)
    baseline_rows = []
    for path in sorted((root / "baseline").glob("*/report.json")):
        baseline = _read(path)
        match = next((r for r in reports if r["case"]["id"] == baseline["matched_case_id"]), None)
        if match is None or baseline["matched_case_fingerprint"] != match["case_fingerprint"]:
            raise ValueError(f"Baseline does not match a completed custom case: {path}")
        validate_completed(path.parent, fingerprint(baseline["baseline_identity"]))
        if baseline["baseline_identity"]["matched_case_fingerprint"] != match["case_fingerprint"]:
            raise ValueError(f"Baseline identity does not match custom case: {path}")
        if baseline["sampling_timesteps"] != match["sampling_timesteps"][:-1]:
            raise ValueError(f"Baseline and custom reverse grids differ: {path}")
        for method, report in (("custom_inversion", match), ("diffusers_img2img_baseline", baseline)):
            baseline_rows.append({"case_id": match["case"]["id"], "method": method, "sample": match["case"]["sample"],
                                  "strength": match["case"]["settings"]["strength"], "runtime_seconds": report["runtime_seconds"],
                                  **report["budget"], **report["metrics"]})
        _grid([root / "runs" / match["case"]["id"] / "original.png",
               root / "runs" / match["case"]["id"] / "output.png", path.parent / "output.png"],
              ["Original", "Custom DDIM inversion edit", "BASELINE: Diffusers DDIM Img2Img (random noise)"],
              summaries / f"baseline--{match['case']['id']}.png")
    if baseline_rows:
        names = sorted(set().union(*(set(row) for row in baseline_rows)))
        with (summaries / "baseline_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(baseline_rows)
    summary = {"planned_cases": len(manifest["planned_cases"]), "completed_cases": len(reports), "missing_cases": missing,
               "studies": studies, "baseline_cases": len(baseline_rows) // 2,
               "note": "Descriptive per-image results. No prompt-alignment metric or statistical significance claim; do not credit pixel compositing as a latent gain."}
    write_json(summaries / "summary.json", summary)
    return summary
