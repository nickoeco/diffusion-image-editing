import pytest

from diffusion_editor.config import ProjectConfig, load_config


def test_default_config():
    config = load_config("configs/default.yaml")
    assert config == ProjectConfig()
    assert config.local_files_only
    assert len(config.revision) == 40
    assert config.to_dict()["seed"] == 42


@pytest.mark.parametrize("kwargs", [
    {"device": "typo"}, {"precision": "bf16"}, {"device": "cpu", "precision": "fp16"},
    {"local_files_only": "false"}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
    {"model_id": ""}, {"revision": None}, {"cache_dir": ""}, {"log_level": "TYPO"},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ProjectConfig(**kwargs)


@pytest.mark.parametrize("content", ["", "[]", "unknown: 1", "seed: nope"])
def test_invalid_yaml(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)
