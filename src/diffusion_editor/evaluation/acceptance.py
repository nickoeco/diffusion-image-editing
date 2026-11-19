"""Offline acceptance planning and source-mask-based boundary review panels."""
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import yaml

CATEGORIES = ("recolor", "replacement", "style")
SPLITS = ("development", "holdout")
REVIEW_FIELDS = (
    "case_id", "category", "split", "run_id", "freeze_id",
    "semantic_score", "preservation_score", "boundary_score", "verdict",
    "semantic_notes", "preservation_notes", "boundary_notes", "failure_tags",
    "psnr_db", "ssim", "background_mae", "metrics_report", "boundary_panel", "output_sha256",
)


def load_acceptance_plan(path: str | Path) -> dict[str, Any]:
    """Validate balanced registration; pending assets are not execution-ready cases."""
    plan = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Acceptance plan requires schema_version 1")
    cases = plan.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(c, dict) or not isinstance(c.get("id"), str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", c["id"]) for c in cases):
        raise ValueError("Case IDs must be lowercase names with letters, digits and underscores")
    if len(cases) != 12 or len({case["id"] for case in cases}) != 12:
        raise ValueError("Register exactly 12 unique case IDs")
    for case in cases:
        if case.get("category") not in CATEGORIES or case.get("split") not in SPLITS:
            raise ValueError("Unknown acceptance category or split")
        for field in ("task", "semantic_target", "preserve", "boundary_checks"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Missing case-specific criterion: {field}")
        for field in ("image", "mask", "source_group", "provenance"):
            if case.get(field) is not None and (not isinstance(case[field], str) or not case[field].strip()):
                raise ValueError(f"{field} must be nonempty text or null while pending")
    for category in CATEGORIES:
        for split in SPLITS:
            if sum(c["category"] == category and c["split"] == split for c in cases) != 2:
                raise ValueError("Each category requires two development and two holdout cases")
    for field in ("image", "source_group"):
        values = [c.get(field) for c in cases if c.get(field) is not None]
        if len(values) != len(set(values)):
            raise ValueError(f"Cases must use independent sources; repeated {field}")
    return plan


def write_review_template(plan: dict[str, Any], path: str | Path) -> None:
    """No fabricated grades or metric zeros; unexecuted rows remain not_run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Review table exists; choose a new path: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({"case_id": c["id"], "category": c["category"], "split": c["split"], "verdict": "not_run"}
                         for c in plan["cases"])


def case_verdict(semantic: int | None, preservation: int | None, boundary: int | None) -> str:
    """All three dimensions must independently pass; pixel metrics cannot vote."""
    scores = (semantic, preservation, boundary)
    if any(score is not None and (type(score) is not int or score not in (0, 1, 2)) for score in scores):
        raise ValueError("Scores must be 0, 1, 2 or None")
    if any(score is None for score in scores):
        return "unreviewed"
    if 0 in scores:
        return "fail"
    return "pass" if scores == (2, 2, 2) else "partial"


def boundary_boxes(mask: Image.Image, crop_size: int = 64) -> list[tuple[int, int, int, int]]:
    """Deterministic top/right/bottom/left anchors from source mask only (L/1)."""
    if mask.mode not in ("L", "1") or type(crop_size) is not int or crop_size < 8:
        raise ValueError("Use a grayscale mask and integer crop_size >=8")
    hard = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
    a = np.asarray(hard)
    if not (a == 0).any() or not (a > 0).any():
        raise ValueError("Boundary review requires a nonempty partial mask")
    edge = np.asarray(hard.filter(ImageFilter.MaxFilter(3))) != np.asarray(hard.filter(ImageFilter.MinFilter(3)))
    y, x = np.nonzero(edge)
    cx, cy = float(x.mean()), float(y.mean())
    candidates = (np.flatnonzero(y == y.min()), np.flatnonzero(x == x.max()),
                  np.flatnonzero(y == y.max()), np.flatnonzero(x == x.min()))
    anchors = [indices[np.argmin(abs((x if i % 2 == 0 else y)[indices] - (cx if i % 2 == 0 else cy)))]
               for i, indices in enumerate(candidates)]
    width, height = mask.size
    cw, ch = min(crop_size, width), min(crop_size, height)
    boxes = []
    for index in anchors:
        left = min(max(int(x[index]) - cw // 2, 0), width - cw)
        top = min(max(int(y[index]) - ch // 2, 0), height - ch)
        box = (left, top, left + cw, top + ch)
        if box not in boxes:
            boxes.append(box)
    return boxes


def save_boundary_review(original: Image.Image, edited: Image.Image, mask: Image.Image,
                         output_dir: str | Path, *, raw: Image.Image | None = None,
                         crop_size: int = 64, zoom: int = 4,
                         extra_centers: tuple[tuple[int, int], ...] = ()) -> dict[str, Any]:
    """Aligned RGB panels, source-mask anchors and nearest-neighbor detail zooms.

    This renders evidence only. It neither edits model outputs nor assigns scores.
    At default zoom each 64px source crop is displayed as 256px, without smoothing.
    """
    original, edited, mask = (ImageOps.exif_transpose(image) for image in (original, edited, mask))
    raw = ImageOps.exif_transpose(raw) if raw is not None else None
    if edited.size != original.size or mask.size != original.size or (raw is not None and raw.size != original.size):
        raise ValueError("Original, mask and all outputs must have identical dimensions")
    if type(zoom) is not int or not 1 <= zoom <= 8:
        raise ValueError("zoom must be an integer in [1,8]")
    boxes = boundary_boxes(mask, crop_size)
    width, height = original.size
    cw, ch = min(crop_size, width), min(crop_size, height)
    for center in extra_centers:
        if len(center) != 2 or any(type(v) is not int for v in center) or not (0 <= center[0] < width and 0 <= center[1] < height):
            raise ValueError("Extra centers must be integer source-image coordinates inside the image")
        left = min(max(center[0] - cw // 2, 0), width - cw)
        top = min(max(center[1] - ch // 2, 0), height - ch)
        box = (left, top, left + cw, top + ch)
        if box not in boxes:
            boxes.append(box)
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Boundary output directory is nonempty; preserve the existing evidence")
    directory.mkdir(parents=True, exist_ok=True)
    stages = [("Original", original.convert("RGB"))]
    if raw is not None:
        stages.append(("Before pixel compositing", raw.convert("RGB")))
    stages.append(("Final edit", edited.convert("RGB")))
    locator = original.convert("RGB").copy()
    draw = ImageDraw.Draw(locator)
    for index, box in enumerate(boxes, 1):
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline="yellow", width=2)
        draw.text((box[0] + 3, box[1] + 3), str(index), fill="yellow", stroke_width=1, stroke_fill="black")
    locator.save(directory / "crop_locations.png")
    mask.convert("L").save(directory / "requested_mask.png")
    cw, ch = boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]
    panel = Image.new("RGB", (cw * zoom * len(stages), (ch * zoom + 42) * len(boxes)), "white")
    labels = ImageDraw.Draw(panel)
    for row, box in enumerate(boxes):
        for col, (name, image) in enumerate(stages):
            x, y = col * cw * zoom, row * (ch * zoom + 42)
            labels.text((x + 5, y + 4), f"Boundary {row + 1}: {name}", fill="black")
            labels.text((x + 5, y + 20), f"Source box {box}", fill="black")
            panel.paste(image.crop(box).resize((cw * zoom, ch * zoom), Image.Resampling.NEAREST), (x, y + 42))
    panel.save(directory / "boundary_panel.png")
    digest = lambda image: hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    report = {"image_size_wh": original.size, "boxes_xyxy": boxes, "extra_centers_xy": extra_centers,
              "crop_size_requested": crop_size, "zoom": zoom, "resampling": "nearest",
              "selection": "source requested mask only; fixed order top/right/bottom/left; duplicate boxes removed",
              "original_rgb_sha256": digest(original), "edited_rgb_sha256": digest(edited),
              "mask_l_sha256": hashlib.sha256(mask.convert("L").tobytes()).hexdigest(),
              "raw_rgb_sha256": digest(raw) if raw is not None else None,
              "note": "Evidence only; semantic/preservation/boundary ratings require separate review. Metrics must come from the recorded floating-tensor run."}
    (directory / "boundary_review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
