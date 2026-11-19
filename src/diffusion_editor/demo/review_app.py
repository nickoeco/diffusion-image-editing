"""Local, model-free human review of two development candidates per source."""
import argparse
from pathlib import Path
from typing import Any

from diffusion_editor.evaluation.review_store import ReviewStore, RATING_FIELDS, SCORE_FIELDS


def create_review_app(store: ReviewStore):
    """Build the optional Gradio interface without loading diffusion models."""
    import gradio as gr

    def render(case_id: str, notice: str = "") -> list[Any]:
        case = store.case(case_id)
        criteria = case["case"]
        baseline = case["baseline_score"]
        baseline_text = "No saved baseline scores."
        if baseline:
            baseline_text = "Baseline scores (semantic / preservation / boundary): " + " / ".join(
                baseline[k] for k in SCORE_FIELDS) + f". Verdict: {baseline['verdict']}."
        gallery = [(case["original"], "Original"), (case["baseline"], "Baseline · strength 0.75 / guidance 7.5")]
        for label, variant in zip(("A", "B"), case["variants"]):
            settings = variant["report"]["settings"]
            gallery.append((variant["image"], f"{label} · strength {settings['strength']} / guidance {settings['guidance']}"))
        result = [
            {"case_id": case_id, "revision": case["revision"]}, case_id,
            f"**Case {case['index'] + 1}/{case['total_cases']} · {case['completed_outputs']}/{case['total_outputs']} outputs fully rated.** {notice}",
            f"### {case_id}: {criteria['task']}\n\n"
            f"**Semantic target:** {criteria['semantic_target']}\n\n"
            f"**Preserve:** {criteria['preserve']}\n\n"
            f"**Boundary checks:** {criteria['boundary_checks']}\n\n{baseline_text}",
            case["prompts"]["source"], case["prompts"]["target"], gallery, case["boundaries"],
        ]
        for variant in case["variants"]:
            row = variant["row"]
            result.extend(int(row[k]) if row[k].strip() else None for k in SCORE_FIELDS)
            result.extend(row[k] for k in RATING_FIELDS[3:])
            result.append(f"Saved verdict: **{row['verdict']}** · run `{row['run_id']}`")
        result.append(str(store.table))
        return result

    def action(kind: str, state: dict, selected: str, *values):
        try:
            current = state["case_id"]
            if kind == "reload":
                return render(current, "Reloaded CSV; unsaved form changes discarded.")
            ratings = [dict(zip(RATING_FIELDS, values[offset:offset + 7])) for offset in (0, 7)]
            store.save(current, ratings, state["revision"], require_complete=kind == "next")
            index = store.case_ids.index(current)
            destination = selected if kind == "select" else current
            if kind == "previous":
                destination = store.case_ids[max(index - 1, 0)]
            elif kind == "next":
                destination = store.case_ids[min(index + 1, len(store.case_ids) - 1)]
            return render(destination, "Saved. " + ("Last case reached." if kind == "next" and index == len(store.case_ids) - 1 else ""))
        except (ValueError, OSError, KeyError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Editing acceptance review", analytics_enabled=False) as app:
        gr.Markdown("# Editing acceptance review\n"
                    "Development cases only. Compare original, baseline, A and B; click an image to enlarge. "
                    "Scores are independent: **0 = fail, 1 = partial, 2 = pass**. "
                    "Enter a reason for each dimension. No reviewer identity or review time is collected.")
        state = gr.State()
        with gr.Row():
            selector = gr.Dropdown(store.case_ids, label="Case", interactive=True)
            download = gr.File(label="Download current rating table", interactive=False)
        progress = gr.Markdown()
        criteria = gr.Markdown()
        with gr.Row():
            source = gr.Textbox(label="Source prompt (exact generation text)", interactive=False, lines=3)
            target = gr.Textbox(label="Target prompt (exact generation text)", interactive=False, lines=3)
        gallery = gr.Gallery(label="Original / baseline / candidate A / candidate B", columns=4,
                             height=400, object_fit="contain", show_download_button=True)
        with gr.Accordion("Inspect matched boundary zooms (same source coordinates)", open=False):
            boundaries = gr.Image(label="Boundary comparison with prompts", interactive=False, type="filepath")
        fields = []
        verdicts = []
        with gr.Row():
            for label in ("A", "B"):
                with gr.Column():
                    gr.Markdown(f"## Candidate {label}")
                    scores = [gr.Radio([("0 · Fail", 0), ("1 · Partial", 1), ("2 · Pass", 2)],
                                       label=dimension, value=None, interactive=True)
                              for dimension in (f"{label}: Semantic success", f"{label}: Unrelated-content preservation", f"{label}: Boundary quality")]
                    notes = [gr.Textbox(label=f"{label}: {name} reason", lines=2, max_lines=6)
                             for name in ("Semantic", "Preservation", "Boundary")]
                    tags = gr.Textbox(label=f"{label}: Failure tags (optional)")
                    fields.extend([*scores, *notes, tags])
                    verdicts.append(gr.Markdown())
        with gr.Row():
            previous = gr.Button("Save draft & previous")
            draft = gr.Button("Save draft")
            next_case = gr.Button("Save complete & next", variant="primary")
            reload_csv = gr.Button("Reload CSV (discard unsaved form)")
        gr.Markdown("Save draft allows incomplete ratings. Save complete & next requires all scores and reasons for A and B. "
                    "Changing the case saves a draft first. Close Excel before saving. "
                    "External CSV changes require Reload CSV; old table versions are retained in `rating_history/`. "
                    "The baseline is read-only. Holdout outputs are not generated by this tool.")
        outputs = [state, selector, progress, criteria, source, target, gallery, boundaries,
                   *fields[:7], verdicts[0], *fields[7:], verdicts[1], download]
        inputs = [state, selector, *fields]
        app.load(lambda: render(store.case_ids[0]), outputs=outputs)
        for button, kind in ((previous, "previous"), (draft, "draft"), (next_case, "next"), (reload_csv, "reload")):
            button.click(lambda state, selected, *values, kind=kind: action(kind, state, selected, *values),
                         inputs=inputs, outputs=outputs, concurrency_id="review-save", concurrency_limit=1,
                         api_name=False)
        selector.input(lambda state, selected, *values: action("select", state, selected, *values),
                       inputs=inputs, outputs=outputs, concurrency_id="review-save", concurrency_limit=1, api_name=False)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Isolated development output root")
    parser.add_argument("--table", type=Path, required=True, help="Two-candidate development review.csv")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port from 1024 to 65535")
    store = ReviewStore(args.table, args.run_root)
    app = create_review_app(store)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False,
               allowed_paths=[str(store.root)], show_error=True)


if __name__ == "__main__":
    main()
