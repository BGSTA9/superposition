"""Typed dataclass configs, strict YAML IO and ``quick``/``full`` profiles.

SPEC §5 (Code: "Configs are typed dataclasses serialized to YAML, with no magic numbers in
code") and §7 ("each experiment gets a config with quick and full profiles").

A config file ``configs/<experiment>.yaml`` has this shape::

    experiment: intro_pentagon
    base:            # every field shared by all profiles
      seed: 0
      train: {steps: 10000}
    profiles:
      quick: {}      # overrides merged on top of `base`
      full:  {n_restarts: 32}

:func:`load_config` deep-merges ``base`` with the chosen profile (and optional overrides) and
builds the experiment's dataclass with :func:`from_dict`, which rejects unknown keys and
wrong types. The resolved config is what a run stores in ``config.yaml``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

Schedule = Literal["constant", "cosine", "warmup_cosine"]
Profile = Literal["quick", "full"]
PROFILES: tuple[str, ...] = ("quick", "full")

C = TypeVar("C")


@dataclass
class TrainConfig:
    """Optimizer and training-loop settings (SPEC §6 ``train.py``).

    The defaults are the ones verified for E00: Adam, lr 1e-3, batch 1024, 10,000 steps,
    constant schedule, a fresh batch every step. Paper: Demonstrating Superposition, p. 10
    (the paper does not state these; see docs/DECISIONS.md).
    """

    steps: int = 10_000
    batch_size: int = 1024
    lr: float = 1e-3
    schedule: Schedule = "constant"
    warmup_frac: float = 0.05
    l1_hidden: float = 0.0
    log_every: int = 100
    snapshot_every: int | None = None
    eval_samples: int = 20_000


@dataclass
class ExperimentConfig:
    """Fields every experiment config shares. Experiments subclass this."""

    experiment: str = ""
    profile: str = "quick"
    seed: int = 0
    device: str = "auto"
    train: TrainConfig = field(default_factory=TrainConfig)


class ConfigError(ValueError):
    """A config file or dict does not match its dataclass."""


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", repr(tp))


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """Convert a YAML-loaded ``value`` to type ``tp``, or raise ConfigError."""
    origin = get_origin(tp)
    if tp is Any:
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}, got {value!r}")
        return from_dict(tp, value, _where=where)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: None is not allowed")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, where)
            except ConfigError as err:
                errors.append(str(err))
        raise ConfigError(f"{where}: {value!r} matches none of {args}: {errors}")
    if origin is Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{where}: {value!r} is not one of {get_args(tp)}")
        return value
    if origin in (list, tuple):
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        args = get_args(tp)
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
            return tuple(
                _coerce(a, v, f"{where}[{i}]")
                for i, (a, v) in enumerate(zip(args, value, strict=True))
            )
        item_tp = args[0] if args else Any
        items = [_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an int, got {value!r}")
        return value
    if tp is float:
        # PyYAML (YAML 1.1) reads `1e-3` as a string; accept numeric strings for floats.
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a float, got {value!r}")
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"{where}: expected a float, got {value!r}")
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported field type {_type_name(tp)}")


def from_dict(cls: type[C], data: Mapping[str, Any], *, _where: str = "") -> C:
    """Build dataclass ``cls`` from a mapping, strictly.

    Unknown keys and wrongly typed values raise :class:`ConfigError`; missing keys take the
    dataclass defaults. Nested dataclasses, ``X | None``, ``Literal``, lists and tuples are
    supported.
    """
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - names
    if unknown:
        prefix = f"{_where}: " if _where else ""
        raise ConfigError(f"{prefix}unknown key(s) {sorted(unknown)} for {cls.__name__}")
    kwargs = {
        key: _coerce(hints[key], value, f"{_where}.{key}" if _where else key)
        for key, value in data.items()
    }
    return cls(**kwargs)


def to_dict(config: Any) -> dict[str, Any]:
    """Convert a config dataclass to plain Python containers (tuples become lists)."""

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [plain(v) for v in value]
        return value

    return plain(dataclasses.asdict(config))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; mappings merge, everything else replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_profiles(path: Path) -> dict[str, Any]:
    """Read a config YAML file and check its top-level structure."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping) or not {"base", "profiles"} <= set(raw):
        raise ConfigError(f"{path}: expected top-level keys 'base' and 'profiles'")
    missing = [p for p in PROFILES if p not in raw["profiles"]]
    if missing:
        raise ConfigError(f"{path}: missing profile(s) {missing}")
    return dict(raw)


def load_config(
    cls: type[C],
    path: Path,
    profile: str = "quick",
    overrides: Mapping[str, Any] | None = None,
) -> C:
    """Load ``path``, merge ``base`` + ``profiles[profile]`` + ``overrides``, build ``cls``.

    The resolved config records the experiment name (from the file) and the profile.
    """
    raw = read_profiles(path)
    if profile not in raw["profiles"]:
        raise ConfigError(f"{path}: unknown profile {profile!r}; choose from {PROFILES}")
    merged = deep_merge(raw["base"] or {}, raw["profiles"][profile] or {})
    merged = deep_merge(merged, overrides or {})
    merged["experiment"] = raw.get("experiment", merged.get("experiment", ""))
    merged["profile"] = profile
    return from_dict(cls, merged)


def save_config(config: Any, path: Path) -> None:
    """Write a resolved config to YAML (keys in dataclass order)."""
    Path(path).write_text(yaml.safe_dump(to_dict(config), sort_keys=False))


def config_hash(config: Any, length: int = 8) -> str:
    """A short, stable hash of a config's content (used in run IDs)."""
    canonical = json.dumps(to_dict(config), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]
