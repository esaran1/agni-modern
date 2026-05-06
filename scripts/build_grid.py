"""Build and persist patch grid for Indonesia or pilot sub-region."""

from __future__ import annotations

from pathlib import Path

import typer

from agni_modern.data.grid import BoundingBox, generate_indonesia_patch_grid, generate_patch_grid
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Generate patch grid using experiment data config."""
    cfg = load_experiment_config(config, overrides=set)

    pilot_bbox = cfg.data.aoi.get("pilot_bbox")
    if isinstance(pilot_bbox, dict) and {"min_lon", "min_lat", "max_lon", "max_lat"}.issubset(pilot_bbox):
        bounds = BoundingBox(
            min_lon=float(pilot_bbox["min_lon"]),
            min_lat=float(pilot_bbox["min_lat"]),
            max_lon=float(pilot_bbox["max_lon"]),
            max_lat=float(pilot_bbox["max_lat"]),
        )
        grid = generate_patch_grid(bounds=bounds, grid_km=cfg.data.spatial.grid_km, patch_prefix="pilot")
    else:
        grid = generate_indonesia_patch_grid(grid_km=cfg.data.spatial.grid_km)

    path = Path(cfg.data.io.patch_grid_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(path, index=False)
    typer.echo(f"Saved patch grid to: {path} | rows={len(grid)}")


if __name__ == "__main__":
    app()
