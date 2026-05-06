"""YAML config loader with include composition and CLI key-value overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from agni_modern.utils.config_schema import DataConfig, ExperimentConfig, FeatureConfig, ModelConfig


class ConfigError(ValueError):
    """Raised when experiment config parsing fails."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected mapping at root in {path}")
    return data


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _resolve_include_path(base_dir: Path, include_path: str) -> Path:
    candidate = Path(include_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return (base_dir / candidate).resolve()


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `key=value` style overrides onto a config dictionary."""
    if not overrides:
        return config

    merged = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}'. Expected key=value format.")
        key, raw_val = item.split("=", 1)
        _set_nested(merged, key.strip(), _parse_scalar(raw_val.strip()))
    return merged


def load_experiment_config(config_path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load experiment config and compose includes for data/features/model."""
    path = Path(config_path).resolve()
    exp_cfg = _load_yaml(path)

    includes = exp_cfg.get("includes")
    if not isinstance(includes, dict):
        raise ConfigError("Experiment config must define includes: {data, features, model}")

    base_dir = path.parent
    data_path = _resolve_include_path(base_dir, includes["data"])
    features_path = _resolve_include_path(base_dir, includes["features"])
    model_path = _resolve_include_path(base_dir, includes["model"])

    data_cfg = DataConfig.model_validate(_load_yaml(data_path)).model_dump()
    features_cfg = FeatureConfig.model_validate(_load_yaml(features_path)).model_dump()
    model_cfg = ModelConfig.model_validate(_load_yaml(model_path)).model_dump()

    composed = _deep_update(exp_cfg, {"data": data_cfg, "features": features_cfg, "model": model_cfg})
    composed = apply_overrides(composed, overrides)
    return ExperimentConfig.model_validate(composed)
