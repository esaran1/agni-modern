"""Ablation runner helpers for feature-family and model-component studies."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_ablation_configs(base_cfg: dict[str, Any], toggles: list[str]) -> list[dict[str, Any]]:
    """Create configs that disable one feature family at a time."""
    ablations: list[dict[str, Any]] = []
    for toggle in toggles:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("features", {}).setdefault("feature_families", {})[toggle] = False
        cfg.setdefault("ablation", {})["disabled_family"] = toggle
        ablations.append(cfg)
    return ablations
