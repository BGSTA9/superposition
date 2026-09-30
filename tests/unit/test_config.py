"""Unit tests for config.py and utils.py."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from superposition.config import (
    ConfigError,
    ExperimentConfig,
    TrainConfig,
    config_hash,
    deep_merge,
    from_dict,
    load_config,
    profile_names,
    save_config,
    to_dict,
)
from superposition.utils import derive_seed, resolve_device


@dataclass
class Inner:
    width: int = 2
    label: str = "a"


@dataclass
class DemoConfig(ExperimentConfig):
    sparsity: list[float] = field(default_factory=lambda: [0.0, 0.9])
    inner: Inner = field(default_factory=Inner)
    limit: float | None = None


def test_from_dict_builds_nested_and_coerces() -> None:
    cfg = from_dict(
        DemoConfig,
        {
            "seed": 3,
            "train": {"lr": "1e-3", "steps": 50},
            "sparsity": [0, 0.8],
            "inner": {"width": 5},
        },
    )
    assert cfg.seed == 3
    assert cfg.train.lr == pytest.approx(1e-3)  # PyYAML reads 1e-3 as a string
    assert cfg.train.steps == 50
    assert cfg.sparsity == [0.0, 0.8]
    assert cfg.inner == Inner(width=5, label="a")
    assert cfg.limit is None


@pytest.mark.parametrize(
    "bad",
    [
        {"unknown": 1},
        {"seed": "zero"},
        {"seed": True},
        {"train": {"schedule": "linear"}},
        {"train": {"steps": 1.5}},
        {"inner": {"extra": 1}},
        {"sparsity": 0.5},
    ],
)
def test_from_dict_rejects_bad_input(bad: dict) -> None:
    with pytest.raises(ConfigError):
        from_dict(DemoConfig, bad)


def test_deep_merge() -> None:
    merged = deep_merge({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"d": 4}, "e": 5})
    assert merged == {"a": 1, "b": {"c": 2, "d": 4}, "e": 5}


def test_load_config_profiles_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text(
        "experiment: demo\n"
        "base:\n  seed: 1\n  train: {steps: 100, lr: 1.0e-3}\n  sparsity: [0.0]\n"
        "profiles:\n  quick: {}\n  full:\n    train: {steps: 5000}\n    inner: {label: big}\n"
    )
    quick = load_config(DemoConfig, path, "quick")
    full = load_config(DemoConfig, path, "full", overrides={"seed": 9})
    assert (quick.experiment, quick.profile, quick.train.steps) == ("demo", "quick", 100)
    assert (full.profile, full.train.steps, full.train.lr, full.inner.label) == (
        "full",
        5000,
        1e-3,
        "big",
    )
    assert full.seed == 9
    with pytest.raises(ConfigError):
        load_config(DemoConfig, path, "medium")
    out = tmp_path / "resolved.yaml"
    save_config(full, out)
    import yaml

    assert from_dict(DemoConfig, yaml.safe_load(out.read_text())) == full


def test_config_hash_is_stable_and_content_sensitive() -> None:
    a, b = DemoConfig(), DemoConfig()
    assert config_hash(a) == config_hash(b)
    b.train = TrainConfig(steps=11)
    assert config_hash(a) != config_hash(b)
    assert to_dict(a)["train"]["steps"] == 10_000


def test_named_variant_profiles(tmp_path: Path) -> None:
    """A file may define documented variants beyond quick and full (D-309)."""
    path = tmp_path / "demo.yaml"
    path.write_text(
        "base:\n  train: {steps: 100, batch_size: 256}\n"
        "profiles:\n  quick: {}\n  full: {train: {steps: 500}}\n"
        "  batch1024: {train: {batch_size: 1024}}\n"
    )
    assert profile_names(path) == ["quick", "full", "batch1024"]
    variant = load_config(DemoConfig, path, "batch1024")
    assert (variant.profile, variant.train.steps, variant.train.batch_size) == (
        "batch1024",
        100,
        1024,
    )


def test_missing_profile_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("base: {}\nprofiles:\n  quick: {}\n")
    with pytest.raises(ConfigError):
        load_config(DemoConfig, path)


def test_derive_seed() -> None:
    assert derive_seed(0, 1, 2) == derive_seed(0, 1, 2)
    seeds = {derive_seed(0, 0, 0, k) for k in range(100)} | {derive_seed(1, 0, 0, 0)}
    assert len(seeds) == 101
    assert all(0 <= s < 2**63 for s in seeds)
    with pytest.raises(ValueError):
        derive_seed(-1)


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
    with pytest.raises(ValueError):
        resolve_device("tpu")
