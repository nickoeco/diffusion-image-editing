"""Offline rubric, balanced registration and honest boundary evidence contracts."""
from copy import deepcopy
import csv

import numpy as np
from PIL import Image, ImageDraw
import pytest
import yaml

from diffusion_editor.evaluation.acceptance import (
    boundary_boxes, case_verdict, load_acceptance_plan, save_boundary_review, write_review_template,
)


def test_balanced_pending_plan_and_blank_table(tmp_path):
    plan = load_acceptance_plan("configs/acceptance/cases.yaml")
    assert plan["freeze_status"] == "not_frozen"
    assert sum(c["split"] == "holdout" for c in plan["cases"]) == 6
    assert all(c["image"] is None for c in plan["cases"])
    target = tmp_path / "review.csv"
    write_review_template(plan, target)
    with target.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert "reviewer" not in rows[0] and "reviewed_at" not in rows[0]
    assert len(rows) == 12 and {row["verdict"] for row in rows} == {"not_run"}
    assert all(row["semantic_score"] == row["psnr_db"] == "" for row in rows)
    with pytest.raises(FileExistsError):
        write_review_template(plan, target)


@pytest.mark.parametrize("scores,expected", [((2, 2, 2), "pass"), ((1, 2, 2), "partial"),
    ((2, 0, 2), "fail"), ((2, 2, 0), "fail"), ((0, 2, 2), "fail"), ((None, 2, 2), "unreviewed")])
def test_no_metric_or_average_can_override_a_dimension(scores, expected):
    assert case_verdict(*scores) == expected


def test_invalid_scores_and_unbalanced_or_duplicate_sources(tmp_path):
    with pytest.raises(ValueError):
        case_verdict(True, 2, 2)
    plan = load_acceptance_plan("configs/acceptance/cases.yaml")
    for field, value in (("split", "holdout"), ("source_group", "same_capture"), ("image", "same.png")):
        changed = deepcopy(plan)
        changed["cases"][0][field] = value
        if field != "split":
            changed["cases"][1][field] = value
        path = tmp_path / "invalid.yaml"
        path.write_text(yaml.safe_dump(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_acceptance_plan(path)


def test_boundary_alignment_fixed_source_boxes_and_unfiltered_zoom(tmp_path):
    mask = Image.new("L", (128, 96))
    ImageDraw.Draw(mask).rectangle((32, 24, 95, 71), fill=255)
    original = Image.fromarray(np.arange(96 * 128 * 3, dtype=np.uint8).reshape(96, 128, 3))
    final = Image.new("RGB", original.size, "red")
    boxes = boundary_boxes(mask, 24)
    assert len(boxes) == 4
    first = save_boundary_review(original, final, mask, tmp_path / "one", raw=original, crop_size=24)
    second = save_boundary_review(original, original, mask, tmp_path / "two", crop_size=24)
    assert first["image_size_wh"] == (128, 96)
    assert first["boxes_xyxy"] == second["boxes_xyxy"] == boxes
    with Image.open(tmp_path / "one/boundary_panel.png") as panel:
        assert panel.size == (24 * 4 * 3, (24 * 4 + 42) * 4)
        expected = original.crop(boxes[0]).resize((96, 96), Image.Resampling.NEAREST)
        np.testing.assert_array_equal(np.asarray(panel.crop((0, 42, 96, 138))), np.asarray(expected))
    assert first["edited_rgb_sha256"] != second["edited_rgb_sha256"]
    extra = save_boundary_review(original, final, mask, tmp_path / "extra", crop_size=24, extra_centers=((100, 80),))
    assert len(extra["boxes_xyxy"]) == 5 and extra["extra_centers_xy"] == ((100, 80),)
    with pytest.raises(ValueError, match="Extra centers"):
        save_boundary_review(original, final, mask, tmp_path / "bad_center", extra_centers=((-1, 5),))
    with pytest.raises(FileExistsError):
        save_boundary_review(original, final, mask, tmp_path / "one")
    with pytest.raises(ValueError, match="identical dimensions"):
        save_boundary_review(original, Image.new("RGB", (32, 32)), mask, tmp_path / "bad")
    for bad in (Image.new("L", mask.size), Image.new("L", mask.size, 255), mask.convert("RGB")):
        with pytest.raises(ValueError):
            boundary_boxes(bad)
