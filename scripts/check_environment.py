"""Report hardware/dependencies without loading model weights."""
import argparse
import json
import sys
from pathlib import Path

# Also works before editable installation, using only the standard library.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_editor.utils.environment import inspect_environment, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_environment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        write_report(args.output, report)


if __name__ == "__main__":
    main()
