"""Forward-noising Img2Img/inpainting/outpainting with cached pretrained components."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("img2img", "inpaint", "outpaint"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forward-editing"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    for side, default in (("left", 0), ("right", 128), ("top", 0), ("bottom", 0), ("overlap", 0)):
        parser.add_argument("--" + side, type=int, default=default)
    parser.add_argument("--fill", choices=("edge", "neutral"), default="edge")
    parser.add_argument("--feather", type=float, default=0)
    parser.add_argument("--soft-mask", action="store_true")
    parser.add_argument("--no-latent-blend", action="store_true")
    parser.add_argument("--no-pixel-composite", action="store_true")
    args = parser.parse_args()
    if args.mode == "inpaint" and args.mask is None:
        parser.error("inpaint requires --mask")
    if args.mode != "inpaint" and args.mask is not None:
        parser.error("--mask applies only to inpaint")
    from PIL import Image
    from diffusion_editor.config import load_config
    from diffusion_editor.demo.forward_service import ForwardRequest
    from diffusion_editor.demo.service import DemoService
    from diffusion_editor.utils.logging import configure_logging
    configure_logging("INFO")
    request = ForwardRequest(args.mode, args.prompt, negative_prompt=args.negative_prompt, steps=args.steps,
        strength=args.strength, guidance=args.guidance, seed=args.seed, soft_mask=args.soft_mask,
        feather_radius=args.feather, latent_blending=not args.no_latent_blend,
        preserve_background=not args.no_pixel_composite, left=args.left, right=args.right,
        top=args.top, bottom=args.bottom, overlap=args.overlap, fill=args.fill)
    with Image.open(args.image) as value:
        image = value.copy()
    mask = None
    if args.mask:
        with Image.open(args.mask) as value:
            mask = value.copy()
    result = DemoService(load_config(args.config), args.output_dir).forward_edit(image, mask, None, request)
    print(result.status)
    print(result.archive_file)


if __name__ == "__main__":
    main()
