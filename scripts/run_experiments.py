"""Run custom-pipeline ablations from a strict YAML plan (cached weights only)."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from diffusion_editor.config import load_config
from diffusion_editor.evaluation.experiment_config import input_hashes, load_suite
from diffusion_editor.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/experiments/phase7.yaml"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/phase7"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without ML imports")
    args = parser.parse_args()
    config, suite = load_config(args.config), load_suite(args.suite)
    if not config.local_files_only:
        parser.error("Experiments require local_files_only=true")
    if args.dry_run:
        print(json.dumps({"suite": suite.to_dict(), "inputs": input_hashes(suite, args.root),
                          "cases": [asdict(case) for case in suite.cases()]}, indent=2))
        return
    configure_logging(config.log_level)
    from diffusion_editor.evaluation.experiments import ExperimentRunner
    from diffusion_editor.evaluation.report import summarize
    reports = ExperimentRunner(config).run(suite, args.output_dir, root=args.root, resume=args.resume)
    summarize(args.output_dir)
    print(f"Completed {len(reports)} cases. Reports, grids and plots: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
