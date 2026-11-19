"""Append literal generation prompts to derived two-candidate review figures."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

from diffusion_editor.evaluation.review_store import ReviewStore, digest
from diffusion_editor.evaluation.review_figures import annotate_prompt_file, append_prompt_footer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    store = ReviewStore(args.table, args.run_root)
    directory = store.table.parent
    table_hash = digest(store.table)
    records = []
    overview_rows = []
    overview = directory / "overview.png"
    overview_backup = directory / "overview.without-prompts.png"
    if overview.exists() and not overview_backup.exists():
        shutil.copyfile(overview, overview_backup)
    for case_id in store.case_ids:
        case = store.case(case_id)
        prompts = case["prompts"]
        for key in ("comparison", "boundaries"):
            path = Path(case[key])
            if path.exists():
                annotate_prompt_file(path, prompts["source"], prompts["target"])
                records.append({"case_id": case_id, "figure": str(path.relative_to(store.root)),
                                "sha256": digest(path), "prompts": prompts})
        # These figures are part of hashed execution evidence: create supplements,
        # preserving every recorded comparison.png and report.json unchanged.
        for parent in [Path(case["baseline"]).parent, *(Path(v["directory"]) for v in case["variants"])]:
            original = parent / "comparison.png"
            annotated = parent / "comparison-with-prompts.png"
            if original.exists():
                with Image.open(original) as image:
                    append_prompt_footer(image, prompts["source"], prompts["target"]).save(annotated)
                records.append({"case_id": case_id, "figure": str(annotated.relative_to(store.root)),
                                "sha256": digest(annotated), "original_sha256": digest(original), "prompts": prompts})
        if overview_backup.exists():
            with Image.open(overview_backup) as image:
                if image.height % len(store.case_ids):
                    raise ValueError("Overview must have one equally sized row per registered case")
                height = image.height // len(store.case_ids)
                row = image.crop((0, case["index"] * height, image.width, (case["index"] + 1) * height))
                overview_rows.append(append_prompt_footer(row, prompts["source"], prompts["target"], font_size=17))
    if overview_rows:
        canvas = Image.new("RGB", (overview_rows[0].width, sum(row.height for row in overview_rows)), "white")
        y = 0
        for row in overview_rows:
            canvas.paste(row, (0, y))
            y += row.height
        canvas.save(overview)
    (directory / "prompt-annotations.json").write_text(json.dumps({
        "prompt_source": "Original, unchanged report.json from the completed generation run",
        "shared_prompt_check": "Source and target match baseline and both candidates exactly",
        "figures": records}, indent=2), encoding="utf-8")
    if digest(store.table) != table_hash:
        raise RuntimeError("Rating table changed during annotation; review the external change")
    print(f"Annotated {len(records)} derived figures and the overview; ratings and raw execution artifacts unchanged.")


if __name__ == "__main__":
    main()
