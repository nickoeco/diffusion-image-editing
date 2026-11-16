"""Strict, serializable one-variable experiment plans; no ML imports."""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import yaml

MASK_MODES = ("global", "latent_only", "latent_feather", "pixel_only", "latent_pixel", "latent_feather_pixel")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("IDs must start with a lowercase letter and contain only a-z, digits, _ or - (max 64).")


def _finite(value: Any, name: str, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower},{upper}]")


@dataclass(frozen=True)
class SampleSpec:
    id: str
    image: str
    source_prompt: str
    target_prompt: str
    provenance: str
    mask: str | None = None

    def __post_init__(self):
        _identifier(self.id)
        for key in ("image", "source_prompt", "target_prompt", "provenance"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"Sample {key} must be nonempty text")
        if self.mask is not None and (not isinstance(self.mask, str) or not self.mask.strip()):
            raise ValueError("mask must be a path or null")


@dataclass(frozen=True)
class RunSettings:
    steps: int = 50
    strength: float = 0.75
    guidance: float = 7.5
    inversion_guidance: float = 1.0
    refinement_steps: int = 0
    refinement_tolerance: float = 1e-6
    seed: int = 42
    negative_prompt: str = ""
    mask_mode: str = "global"
    feather_radius: float = 4.0

    def __post_init__(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 200:
            raise ValueError("steps must be an integer in [1,200]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        if type(self.refinement_steps) is not int or not 0 <= self.refinement_steps <= 50:
            raise ValueError("refinement_steps must be an integer in [0,50]")
        _finite(self.strength, "strength", 0, 1)
        _finite(self.guidance, "guidance", 0, 30)
        _finite(self.inversion_guidance, "inversion_guidance", 0, 30)
        _finite(self.refinement_tolerance, "refinement_tolerance", 0, 1)
        _finite(self.feather_radius, "feather_radius", 0, 64)
        if self.mask_mode not in MASK_MODES or not isinstance(self.negative_prompt, str):
            raise ValueError("Choose a supported mask_mode and text negative_prompt")


@dataclass(frozen=True)
class StudySpec:
    id: str
    operation: str
    samples: tuple[str, ...]
    axis: str
    values: tuple[Any, ...]
    base: RunSettings = field(default_factory=RunSettings)

    def __post_init__(self):
        _identifier(self.id)
        if self.operation not in ("reconstruct", "edit"):
            raise ValueError("operation must be reconstruct or edit")
        allowed = ("steps", "guidance") if self.operation == "reconstruct" else ("steps", "strength", "guidance", "mask_mode")
        if self.axis not in allowed:
            raise ValueError(f"axis must be one of {allowed}")
        if len(self.values) < 2 or any(value in self.values[:i] for i, value in enumerate(self.values)):
            raise ValueError("Each study requires at least two distinct axis values")
        if not self.samples or len(set(self.samples)) != len(self.samples):
            raise ValueError("Study sample IDs must be nonempty and unique")
        if self.operation == "reconstruct" and self.base.mask_mode != "global":
            raise ValueError("Reconstruction cannot use a local mask mode")
        for value in self.values:
            replace(self.base, **{self.axis: value})


@dataclass(frozen=True)
class ExperimentCase:
    id: str
    study: str
    sample: str
    operation: str
    axis: str
    value: Any
    settings: RunSettings


@dataclass(frozen=True)
class ExperimentSuite:
    name: str
    samples: tuple[SampleSpec, ...]
    studies: tuple[StudySpec, ...]
    schema_version: int = 1

    def __post_init__(self):
        _identifier(self.name)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Only experiment schema_version 1 is supported")
        for values in (self.samples, self.studies):
            if not values or len({value.id for value in values}) != len(values):
                raise ValueError("Sample/study IDs must be nonempty and unique")
        samples = {s.id: s for s in self.samples}
        for study in self.studies:
            for sample in study.samples:
                if sample not in samples:
                    raise ValueError(f"Unknown sample: {sample}")
                for value in study.values:
                    settings = replace(study.base, **{study.axis: value})
                    if settings.mask_mode != "global" and not samples[sample].mask:
                        raise ValueError(f"Local study requires a mask: {sample}")

    def cases(self) -> tuple[ExperimentCase, ...]:
        return tuple(ExperimentCase(f"{study.id}--{sample}--{index:02d}", study.id, sample,
                                    study.operation, study.axis, value, replace(study.base, **{study.axis: value}))
                     for study in self.studies for sample in study.samples for index, value in enumerate(study.values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_suite(path: str | Path) -> ExperimentSuite:
    """Unknown keys fail through dataclass construction; paths are relative to --root."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    try:
        samples = tuple(SampleSpec(**sample) for sample in data.pop("samples"))
        studies = tuple(StudySpec(**(dict(study) | {"samples": tuple(study["samples"]),
                             "values": tuple(study["values"]), "base": RunSettings(**study.get("base", {}))}))
                        for study in data.pop("studies"))
        return ExperimentSuite(samples=samples, studies=studies, **data)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid experiment configuration: {exc}") from exc


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def input_hashes(suite: ExperimentSuite, root: Path) -> dict[str, dict[str, str | None]]:
    """Validate all referenced files before loading any weights."""
    return {s.id: {"image": file_hash(root / s.image), "mask": file_hash(root / s.mask) if s.mask else None}
            for s in suite.samples}
