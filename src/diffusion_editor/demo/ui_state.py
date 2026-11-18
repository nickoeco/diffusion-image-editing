"""Pure UI selection rules; no Gradio or model imports."""

EDIT = "Edited image"
VAE = "VAE reconstruction"
DDIM = "DDIM reconstruction"
CONTROL = "Source-prompt control"
MODES = ("Local edit", "Global edit", "Reconstruct source")


def output_choices(mode: str) -> list[str]:
    if mode not in MODES:
        raise ValueError("Choose Local edit, Global edit or Reconstruct source.")
    return [VAE, DDIM] if mode == "Reconstruct source" else [EDIT, VAE, DDIM, CONTROL]


def default_outputs(mode: str) -> list[str]:
    output_choices(mode)
    return [DDIM] if mode == "Reconstruct source" else [EDIT]


def validate_outputs(mode: str, outputs: list[str]) -> None:
    choices = output_choices(mode)
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Select at least one output to calculate.")
    if any(not isinstance(value, str) or value not in choices for value in outputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Choose valid, distinct outputs for this operation.")


def layout_css(mode: str, outputs: list[str]) -> str:
    """Hide inapplicable controls without unmounting the initialized paint canvas."""
    output_choices(mode)
    editing = mode != "Reconstruct source" and any(value in outputs for value in (EDIT, CONTROL))
    selectors = []
    if mode != "Local edit" or not editing:
        selectors.append("#local-mask-panel")
    if not editing:
        selectors.extend(("#edit-parameters", "#inversion-guidance"))
    if EDIT not in outputs or mode == "Reconstruct source":
        selectors.append("#target-prompt")
    if DDIM not in outputs:
        selectors.append("#reconstruction-settings")
    if not editing and DDIM not in outputs:
        selectors.append("#intermediate-settings")
    return "<style>" + ",".join(selectors) + "{display:none !important;}</style>" if selectors else ""
