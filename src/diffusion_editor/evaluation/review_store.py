"""Version-checked human rating persistence and read-only generated evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from typing import Any
from uuid import uuid4

from diffusion_editor.evaluation.acceptance import case_verdict

RATING_FIELDS = ("semantic_score", "preservation_score", "boundary_score",
                 "semantic_notes", "preservation_notes", "boundary_notes", "failure_tags")
SCORE_FIELDS = RATING_FIELDS[:3]
NOTE_FIELDS = RATING_FIELDS[3:6]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(row: dict[str, Any]) -> bool:
    return all(str(row.get(k, "")).strip() in ("0", "1", "2") for k in SCORE_FIELDS) and all(
        str(row.get(k, "")).strip() for k in NOTE_FIELDS)


class ReviewStore:
    """One local CSV, immutable output hashes, versioned snapshots and safe saves.

    No model is imported or executed. Only completed development rows are editable.
    Browser sessions must present the table revision they actually displayed.
    """

    def __init__(self, table: str | Path, run_root: str | Path, *, baseline: str = "development-pilot-01") -> None:
        self.table = Path(table).resolve()
        self.root = Path(run_root).resolve()
        if not self.table.is_relative_to(self.root):
            raise ValueError("Review table must be inside the selected run root")
        self.baseline = self.root / baseline
        self._lock = threading.RLock()
        rows, _, _ = self._read()
        self.case_ids = list(dict.fromkeys(r["case_id"] for r in rows if r["split"] == "development" and r["run_id"]))
        if not self.case_ids:
            raise ValueError("No completed development rows to review")
        for cid in self.case_ids:
            if len([r for r in rows if r["case_id"] == cid and r["run_id"]]) != 2:
                raise ValueError("This comparison UI requires two candidate rows per case")

    def _read(self) -> tuple[list[dict[str, str]], list[str], bytes]:
        raw = self.table.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        fields = reader.fieldnames or []
        required = {"case_id", "split", "run_id", "metrics_report", "output_sha256", "verdict", *RATING_FIELDS}
        if not required.issubset(fields) or {"reviewer", "reviewed_at"}.intersection(fields):
            raise ValueError("Unsupported review CSV schema; use the current table without reviewer/time columns")
        rows = list(reader)
        ids = [r["run_id"] for r in rows if r["run_id"]]
        if len(ids) != len(set(ids)):
            raise ValueError("run_id must uniquely identify each rated output")
        return rows, fields, raw

    def _inside(self, path: str | Path) -> Path:
        target = Path(path)
        target = (target if target.is_absolute() else self.root / target).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Evidence path escapes the selected run root")
        return target

    def case(self, case_id: str) -> dict[str, Any]:
        if case_id not in self.case_ids:
            raise ValueError("Choose a registered development case")
        with self._lock:
            rows, _, raw = self._read()
            selected = [r for r in rows if r["case_id"] == case_id and r["run_id"]]
            variants = []
            for row in selected:
                if row["split"] != "development":
                    raise ValueError("Unexecuted/holdout rows cannot be graded here")
                report_path = self._inside(row["metrics_report"])
                report = json.loads(report_path.read_text(encoding="utf-8"))
                image = report_path.parent / "output.png"
                if digest(image) != row["output_sha256"]:
                    raise ValueError(f"Output hash changed: {row['run_id']}")
                variants.append({"row": row, "report": report, "image": str(image),
                                 "directory": str(report_path.parent)})
            if len(variants) != 2:
                raise ValueError("Candidate rows changed; reload the application")
            baseline_dir = self._inside(self.baseline / case_id)
            baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
            prompts = variants[0]["report"]["prompts"]
            if any(v["report"]["prompts"] != prompts for v in variants) or baseline_report["prompts"] != prompts:
                raise ValueError("Prompts differ across variants; a shared-prompt comparison would be misleading")
            baseline_score = None
            if (self.baseline / "review.csv").exists():
                with (self.baseline / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
                    baseline_score = next((r for r in csv.DictReader(stream) if r["case_id"] == case_id), None)
            return {"case_id": case_id, "revision": hashlib.sha256(raw).hexdigest(),
                    "index": self.case_ids.index(case_id), "total_cases": len(self.case_ids),
                    "completed_outputs": sum(complete(r) for r in rows if r["split"] == "development"),
                    "total_outputs": len(self.case_ids) * 2, "prompts": prompts,
                    "case": variants[0]["report"]["case"], "variants": variants,
                    "original": str(baseline_dir / "original.png"), "baseline": str(baseline_dir / "output.png"),
                    "baseline_score": baseline_score,
                    "comparison": str(self.table.parent / f"{case_id}-comparison.png"),
                    "boundaries": str(self.table.parent / f"{case_id}-boundaries.png")}

    def save(self, case_id: str, ratings: list[dict[str, Any]], expected_revision: str,
             *, require_complete: bool = False) -> str:
        if case_id not in self.case_ids or len(ratings) != 2:
            raise ValueError("Save both candidate ratings for a registered case")
        with self._lock:
            rows, fields, raw = self._read()
            revision = hashlib.sha256(raw).hexdigest()
            if revision != expected_revision:
                raise ValueError("The CSV changed outside this page. Reload CSV before saving; no scores were overwritten.")
            selected = [r for r in rows if r["case_id"] == case_id and r["run_id"]]
            if len(selected) != 2 or any(r["split"] != "development" for r in selected):
                raise ValueError("Candidate rows changed; reload the application")
            # Recheck evidence before associating new ratings with an output.
            self.case(case_id)
            changes = []
            for row, rating in zip(selected, ratings):
                if set(rating) - set(RATING_FIELDS):
                    raise ValueError("Only rating fields can be changed")
                scores = tuple(rating.get(k) for k in SCORE_FIELDS)
                verdict = case_verdict(*scores)
                updated = {k: "" if v is None else str(v) for k, v in zip(SCORE_FIELDS, scores)}
                for key in RATING_FIELDS[3:]:
                    value = rating.get(key, "")
                    if not isinstance(value, str) or len(value) > 6000:
                        raise ValueError("Notes must be text, at most 6000 characters")
                    updated[key] = value
                if require_complete and not complete(updated):
                    raise ValueError("Choose all three scores and enter a reason for each dimension for both candidates, or Save draft.")
                updated["verdict"] = verdict if complete(updated) else "unreviewed"
                if any(row[k] != v for k, v in updated.items()):
                    changes.append({"run_id": row["run_id"], "before": {k: row[k] for k in updated}, "after": updated})
                    row.update(updated)
            if not changes:
                return revision
            history = self.table.parent / "rating_history"
            history.mkdir(exist_ok=True)
            snapshot = history / f"review-{revision}.csv"
            if not snapshot.exists():
                snapshot.write_bytes(raw)
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            new_raw = buffer.getvalue().encode("utf-8-sig")
            with tempfile.NamedTemporaryFile(dir=self.table.parent, prefix=".review-", suffix=".tmp", delete=False) as stream:
                stream.write(new_raw)
                temporary = Path(stream.name)
            try:
                if digest(self.table) != revision:
                    raise ValueError("The CSV changed while saving. Reload; no scores were overwritten.")
                temporary.replace(self.table)
            except PermissionError as exc:
                temporary.unlink(missing_ok=True)
                raise ValueError("Close the CSV in Excel, then reload and save again") from exc
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            new_revision = hashlib.sha256(new_raw).hexdigest()
            (history / f"change-{uuid4().hex}.json").write_text(json.dumps({
                "before_revision": revision, "after_revision": new_revision,
                "case_id": case_id, "changes": changes}, indent=2), encoding="utf-8")
            return new_revision
