"""Offline persistence/evidence tests; optional review UI construction."""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from diffusion_editor.evaluation.acceptance import REVIEW_FIELDS
from diffusion_editor.evaluation.review_store import ReviewStore, RATING_FIELDS
from diffusion_editor.evaluation.review_figures import append_prompt_footer, annotate_prompt_file, _wrap


@pytest.fixture
def review_store(tmp_path):
    rows = []
    for case in ("one", "two"):
        for profile in ("development-pilot-01", "a", "b"):
            directory = tmp_path / profile / case
            directory.mkdir(parents=True)
            Image.new("RGB", (96, 64), "red").save(directory / "output.png")
            Image.new("RGB", (96, 64), "blue").save(directory / "original.png")
            report = {"case": {"id": case, "split": "development", "task": "Change red to green",
                      "semantic_target": "Green", "preserve": "Shape", "boundary_checks": "Contour"},
                      "prompts": {"source": "a red object", "target": "a green object"},
                      "settings": {"strength": 1, "guidance": 7.5}}
            (directory / "report.json").write_text(json.dumps(report))
            if profile != "development-pilot-01":
                row = dict.fromkeys(REVIEW_FIELDS, "")
                row.update(case_id=case, category="recolor", split="development", run_id=f"{profile}-{case}",
                           metrics_report=str(directory / "report.json"), verdict="unreviewed",
                           output_sha256=hashlib.sha256((directory / "output.png").read_bytes()).hexdigest())
                rows.append(row)
    table = tmp_path / "review.csv"
    with table.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return ReviewStore(table, tmp_path)


def rating(score=2):
    return dict(zip(RATING_FIELDS, [score, 2, 2, "Changed to green", "Unchanged", "Natural", ""]))


def test_save_zero_and_notes_without_changing_other_rows(review_store):
    store = review_store
    before = store.table.read_bytes()
    other = store.case("two")["variants"][0]["row"].copy()
    item = rating(0)
    item["semantic_notes"] = 'Not changed, "still red"\nLine 2; 原因'
    store.save("one", [item, rating()], store.case("one")["revision"], require_complete=True)
    saved = store.case("one")
    assert saved["variants"][0]["row"]["semantic_score"] == "0"
    assert saved["variants"][0]["row"]["semantic_notes"] == item["semantic_notes"]
    assert saved["variants"][0]["row"]["verdict"] == "fail"
    assert saved["variants"][1]["row"]["verdict"] == "pass"
    assert saved["completed_outputs"] == 2
    assert store.case("two")["variants"][0]["row"] == other
    assert next((store.table.parent / "rating_history").glob("review-*.csv")).read_bytes() == before
    assert len(list((store.table.parent / "rating_history").glob("change-*.json"))) == 1


def test_partial_draft_and_full_completion_requirement(review_store):
    store = review_store
    draft = rating(None)
    draft["boundary_notes"] = ""
    revision = store.case("one")["revision"]
    with pytest.raises(ValueError, match="all three scores"):
        store.save("one", [draft, rating()], revision, require_complete=True)
    assert store.case("one")["revision"] == revision
    store.save("one", [draft, rating()], revision)
    assert store.case("one")["variants"][0]["row"]["verdict"] == "unreviewed"


def test_stale_csv_and_changed_outputs_rejected(review_store):
    store = review_store
    case = store.case("one")
    store.table.write_bytes(store.table.read_bytes() + b"\r\n")
    raw = store.table.read_bytes()
    with pytest.raises(ValueError, match="changed outside"):
        store.save("one", [rating(), rating()], case["revision"])
    assert store.table.read_bytes() == raw
    Image.new("RGB", (96, 64), "green").save(case["variants"][0]["image"])
    with pytest.raises(ValueError, match="Output hash changed"):
        store.save("one", [rating(), rating()], hashlib.sha256(raw).hexdigest())
    assert store.table.read_bytes() == raw


@pytest.mark.parametrize("value", [-1, 3, True, 1.5, "2"])
def test_invalid_scores_leave_csv_unchanged(review_store, value):
    store = review_store
    raw = store.table.read_bytes()
    with pytest.raises(ValueError, match="Scores must"):
        store.save("one", [rating(value), rating()], store.case("one")["revision"])
    assert store.table.read_bytes() == raw


def test_path_escape_and_prompt_mismatch_rejected(review_store):
    store = review_store
    with pytest.raises(ValueError, match="escapes"):
        store._inside(store.root.parent / "outside.json")
    path = Path(store.case("one")["variants"][0]["directory"]) / "report.json"
    report = json.loads(path.read_text())
    report["prompts"]["target"] = "different target"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Prompts differ"):
        store.case("one")


def test_prompt_footer_preserves_pixels_and_export_is_idempotent(tmp_path):
    from PIL import ImageFont
    source, target = "source " * 40, "target " * 40
    original = Image.new("RGB", (180, 60), "blue")
    rendered = append_prompt_footer(original, source, target)
    assert rendered.crop((0, 0, *original.size)).tobytes() == original.tobytes()
    assert rendered.height > original.height + 100
    font = ImageFont.load_default(size=20)
    assert "".join(_wrap(source, font, 148)) == source
    assert all(font.getlength(line) <= 148 for line in _wrap(target, font, 148))
    path = tmp_path / "comparison.png"
    original.save(path)
    before = path.read_bytes()
    annotate_prompt_file(path, source, target)
    first = path.read_bytes()
    annotate_prompt_file(path, source, target)
    assert path.read_bytes() == first
    assert path.with_name("comparison.without-prompts.png").read_bytes() == before


@pytest.mark.demo
def test_review_app_construction_without_models(review_store):
    from diffusion_editor.demo.review_app import create_review_app
    app = create_review_app(review_store)
    assert len(app.config["dependencies"]) == 6
    text = str(app.config)
    assert "Source prompt (exact generation text)" in text
    assert "Save complete & next" in text
    callbacks = list(app.fns.values())
    first = callbacks[0].fn()
    assert first[0]["case_id"] == "one"
    assert first[4:6] == ["a red object", "a green object"]
    assert first[8:11] == [None, None, None]
    values = [rating()[k] for k in RATING_FIELDS] * 2
    next_page = callbacks[3].fn(first[0], "one", *values)
    assert next_page[0]["case_id"] == "two"
    assert "2/4 outputs fully rated" in next_page[2]
    previous_page = callbacks[1].fn(next_page[0], "two", *([None] * 3 + [""] * 4) * 2)
    assert previous_page[0]["case_id"] == "one"
    assert previous_page[8:11] == [2, 2, 2]
    app.close()
