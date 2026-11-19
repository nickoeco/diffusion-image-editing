"""Literal source/target prompt annotations on derived comparison figures."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            if line and font.getlength(line + char) > width:
                lines.append(line)
                line = ""
            line += char
        lines.append(line)
    return lines


def append_prompt_footer(image: Image.Image, source: str, target: str, *, font_size: int = 20) -> Image.Image:
    """Keep the comparison's existing pixels intact; never truncate prompt text."""
    if not isinstance(source, str) or not isinstance(target, str) or image.width < 80:
        raise ValueError("Provide source/target text and an image at least 80 pixels wide")
    font = ImageFont.load_default(size=font_size)
    line_height = font_size + 7
    lines = _wrap("Generation prompts (shared by baseline and compared candidates)", font, image.width - 32) + [""]
    lines += _wrap("Source prompt: " + source, font, image.width - 32)
    lines += [""] + _wrap("Target prompt: " + target, font, image.width - 32)
    canvas = Image.new("RGB", (image.width, image.height + len(lines) * line_height + 32), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line((0, image.height, image.width, image.height), fill="#888888", width=1)
    for index, line in enumerate(lines):
        draw.text((16, image.height + 16 + index * line_height), line, fill="black", font=font)
    return canvas


def annotate_prompt_file(path: str | Path, source: str, target: str) -> Path:
    """Retain the original derived figure so repeated exports cannot stack footers."""
    path = Path(path)
    original = path.with_name(path.stem + ".without-prompts.png")
    if not original.exists():
        original.write_bytes(path.read_bytes())
    with Image.open(original) as image:
        result = append_prompt_footer(image, source, target)
    result.save(path)
    return path
