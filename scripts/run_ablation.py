"""Run feature-family ablation config generation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from agni_modern.evaluation.ablation import build_ablation_configs
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Generate ablation configs by disabling one feature family at a time."""
    cfg = load_experiment_config(config, overrides=set).model_dump()

    toggles = [k for k, v in cfg.get("features", {}).get("feature_families", {}).items() if v]
    ablations = build_ablation_configs(cfg, toggles=toggles)

    out = Path("outputs/metrics") / "ablation_config_preview.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    typer.echo(f"Saved ablation config preview: {out}")


if __name__ == "__main__":
    app()
