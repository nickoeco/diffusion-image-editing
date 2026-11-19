# Local human review interface

The model-free Gradio review app replaces manual CSV entry for the current
two-candidate development comparison. It uses the existing `demo` dependency,
loads no weights, performs no inference and cannot generate holdout outputs.
The initial version expects exactly two completed development rows per source
and a saved `development-pilot-01` baseline. It is not a general holdout runner.

```powershell
.\.venv\Scripts\python.exe -m diffusion_editor.demo.review_app --run-root outputs/acceptance/2026-11-16-side-validation --table outputs/acceptance/2026-11-16-side-validation/development-round-02/review.csv --port 7861
```

Open `http://127.0.0.1:7861`. The server binds to loopback with public sharing
disabled. It is separate from the image editor on port 7860. Run one review
server per table; concurrent independent writer processes are unsupported.
The runtime needs local socket access, independently of GPU/model opt-ins.

## Reviewing a case

1. Read the case-specific semantic, preservation and boundary criteria and the
   exact source/target prompts copied from generation reports.
2. Inspect original, baseline, candidate A and candidate B. Click gallery images
   to enlarge them and open the matched source-coordinate boundary accordion.
   Baseline scores are read-only; assess each candidate independently using
   [the registered rubric](acceptance.md). Pixel metrics do not vote on ratings.
3. Select **0 = fail, 1 = partial, 2 = pass** for each of the three dimensions,
   and enter a concrete reason for each dimension, in either language. Failure
   tags are optional. Blank remains unreviewed; zero is an actual failure.
4. Use **Save complete & next** when both candidates have all scores and reasons.
   **Save draft**, **Save draft & previous**, and selecting another case preserve
   incomplete work. Changes are not saved simply by typing or closing the tab.
5. Check the saved verdict and progress counter. A pass requires 2/2/2; a zero
   fails and other complete triples are partial. Incomplete reasons or scores
   leave a draft unreviewed. Download the current table from the page if needed.

CSV remains a compatible storage/export format, not a required input interface.
There are no reviewer identity or review-time fields. Each save preserves other
cases, run IDs, output hashes, metric references and verbatim reasons. It snapshots
the previous CSV under `rating_history/` and records changed fields with revision
hashes. Draft edits can revise a prior rating, but the previous raw table remains
available; use separate tables for independent assessments or adjudication.

The server verifies output hashes and shared source/target prompts before display
and saving. If another tab or Excel changes the table, saving rejects the stale
revision. Close Excel, then use **Reload CSV (discard unsaved form)** before
continuing. Copy unsaved notes elsewhere first if needed. Saves replace the table
atomically within one process; the revision check does not provide distributed
locking across independent server processes. A backup is retained before a write.

## Prompt-bearing exported comparisons

```powershell
.\.venv\Scripts\python.exe scripts/annotate_acceptance_comparisons.py --run-root outputs/acceptance/2026-11-16-side-validation --table outputs/acceptance/2026-11-16-side-validation/development-round-02/review.csv
```

Source and target are read literally from each generation report and checked to
match all three compared variants. Full prompts are wrapped below the full-frame
and matched boundary figures; the original image area stays pixel-identical.
The contact sheet includes a separate prompt block for each source. Unannotated
derived panels are retained as `*.without-prompts.png` for repeatable exports.
Hashed run-level `comparison.png` files stay intact; their new
`comparison-with-prompts.png` supplements contain the text. No raw output, input,
mask, execution report or numeric metric is recomputed by this command.

## Verification

Offline tests cover zero/blank distinction, complete/draft verdicts, notes,
versioned backups, unchanged unrelated rows, stale revisions, output integrity,
path containment, prompt mismatches and repeatable figure export. UI construction
requires `pytest tests/test_review.py --run-demo`. Browser verification uses an
isolated copy of the table to exercise scoring, save/navigation, required fields
and stale-write handling without adding test ratings to the real human table.

### Current promotion status (November 18, 2026)

The workspace review reran the offline suite (285 passed, 28 opt-in skips) and
the focused review suite with `--run-demo` (11 passed). The latter checks app
construction and save/navigation callbacks using temporary synthetic tables.
End-to-end browser save/navigation/stale-write validation on a copied table is
still pending; it is not claimed as passed. The app is retained as an optional
standalone development tool, not merged into the main editing demo. See
[workspace-integration-review.md](workspace-integration-review.md) for the full
retention and demo-eligibility decisions.
