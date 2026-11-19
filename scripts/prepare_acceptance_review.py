"""Create a blank case table or aligned boundary panels; never runs a model."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("template")
    table.add_argument("--plan", default="configs/acceptance/cases.yaml")
    table.add_argument("--output", type=Path, required=True)
    panel = commands.add_parser("panels")
    for name in ("original", "edited", "mask", "output_dir"):
        panel.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    panel.add_argument("--raw", type=Path)
    panel.add_argument("--crop-size", type=int, default=64)
    panel.add_argument("--zoom", type=int, default=4)
    panel.add_argument("--extra-center", type=int, nargs=2, action="append", default=[], metavar=("X", "Y"),
                       help="Pre-registered critical detail center; repeat as needed, fixed across methods")
    args = parser.parse_args()
    from diffusion_editor.evaluation.acceptance import load_acceptance_plan, save_boundary_review, write_review_template
    if args.command == "template":
        write_review_template(load_acceptance_plan(args.plan), args.output)
    else:
        from PIL import Image
        def read(path):
            with Image.open(path) as image:
                return image.copy()
        save_boundary_review(read(args.original), read(args.edited), read(args.mask), args.output_dir,
                             raw=read(args.raw) if args.raw else None, crop_size=args.crop_size, zoom=args.zoom,
                             extra_centers=tuple(tuple(center) for center in args.extra_center))


if __name__ == "__main__":
    main()
