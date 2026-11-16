"""Regenerate plots and tables from saved experiment reports, without model loading."""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args()
    from diffusion_editor.evaluation.report import summarize
    print(json.dumps(summarize(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
