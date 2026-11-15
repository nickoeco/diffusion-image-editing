"""Validated configuration; importing this module never imports PyTorch."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: str = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    device: str = "auto"
    precision: str = "auto"
    local_files_only: bool = True
    cache_dir: str = ".cache/huggingface"
    seed: int = 42
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        for key in ("model_id", "revision", "cache_dir"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if self.precision not in ("auto", "fp16", "fp32"):
            raise ValueError("precision must be auto, fp16 or fp32")
        if type(self.local_files_only) is not bool:
            raise ValueError("local_files_only must be boolean")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("invalid log_level")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("CPU inference requires fp32")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    """Read YAML; reject unknown keys to catch configuration typos."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    unknown = set(data) - {field.name for field in fields(ProjectConfig)}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown, key=str)}")
    return ProjectConfig(**data)
