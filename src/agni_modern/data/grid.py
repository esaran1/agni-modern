"""Patch grid generation helpers for Indonesia modeling units."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


@dataclass(slots=True)
class BoundingBox:
    """Simple lon/lat bounds used for coarse grid construction."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


# Approximate Indonesia bounds for scaffolding only.
INDONESIA_BOUNDS = BoundingBox(min_lon=94.0, min_lat=-11.0, max_lon=141.0, max_lat=6.5)


def km_to_degree_lat(km: float) -> float:
    """Approximate latitude degree conversion for small-cell scaffolding."""
    return km / 111.0


def generate_patch_grid(bounds: BoundingBox, grid_km: int = 5, patch_prefix: str = "indo") -> gpd.GeoDataFrame:
    """Generate regular lon/lat grid for any bounding box.

    Notes:
    - Approximate conversion is used for a compact pilot pipeline.
    - TODO: migrate to equal-area tiling for production-scale experiments.
    """
    step = km_to_degree_lat(float(grid_km))
    rows: list[dict[str, object]] = []

    row_idx = 0
    lat = bounds.min_lat
    while lat < bounds.max_lat:
        col_idx = 0
        lon = bounds.min_lon
        while lon < bounds.max_lon:
            geom = box(lon, lat, min(lon + step, bounds.max_lon), min(lat + step, bounds.max_lat))
            centroid = geom.centroid
            rows.append(
                {
                    "patch_id": f"{patch_prefix}_{row_idx}_{col_idx}",
                    "row": row_idx,
                    "col": col_idx,
                    "min_lon": lon,
                    "min_lat": lat,
                    "max_lon": min(lon + step, bounds.max_lon),
                    "max_lat": min(lat + step, bounds.max_lat),
                    "centroid_lon": float(centroid.x),
                    "centroid_lat": float(centroid.y),
                    "geometry": geom,
                }
            )
            col_idx += 1
            lon += step
        row_idx += 1
        lat += step

    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")


def generate_indonesia_patch_grid(grid_km: int = 5) -> gpd.GeoDataFrame:
    """Generate a coarse regular lon/lat grid over Indonesia bounds."""
    return generate_patch_grid(INDONESIA_BOUNDS, grid_km=grid_km, patch_prefix="indo")
