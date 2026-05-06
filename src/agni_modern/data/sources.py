"""Source adapter interfaces and default v1+ dataset registry.

This module supports two modes:
- placeholder mode for scaffold/synthetic flows
- real-pilot mode for a tiny Earth Engine occurrence pilot
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

from agni_modern.utils.types import PatchRecord

FINAL_DATASETS: dict[str, str] = {
    "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel2_cloud_probability": "COPERNICUS/S2_CLOUD_PROBABILITY",
    "cloud_score_plus": "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED",
    "era5_land_daily": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "viirs_active_fire": "NASA/LANCE/SNPP_VIIRS/C2",
    "modis_burned_area": "MODIS/061/MCD64A1",
    "burn_severity": "MODIS/061/MCD64A1",
    "srtm": "CGIAR/SRTM90_V4",
    "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
    "worldpop": "WorldPop/GP/100m/pop",
    "ghsl_built_surface": "JRC/GHSL/P2023A/GHS_BUILT_S",
}

PRIMARY_CLOUD_SOURCE = "sentinel2_cloud_probability"
AUXILIARY_CLOUD_SOURCE = "cloud_score_plus"


class SourceAdapter(Protocol):
    """Protocol for patch-date source summaries used in feature building."""

    source_name: str
    dataset_id: str
    role: str

    def fetch_patch_timeseries(
        self,
        patch: PatchRecord,
        reference_date: date,
        lookback_days: int,
    ) -> dict[str, Any]:
        """Fetch raw source summaries up to reference date."""


def _safe_source_cfg(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _ee_geometry_from_patch(patch: PatchRecord):
    import ee  # type: ignore

    return ee.Geometry.Rectangle([patch.min_lon, patch.min_lat, patch.max_lon, patch.max_lat])


def _coerce_band_list(values: list[str], fallback: list[str]) -> list[str]:
    if not values:
        return fallback
    if any(v.startswith("TODO_") for v in values):
        return fallback
    return values


def _safe_get_number(info: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = info.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class BasePlaceholderSource:
    """Base class for placeholder adapters when real mode is not used."""

    source_name: str
    dataset_id: str
    role: str

    def fetch_patch_timeseries(
        self,
        patch: PatchRecord,
        reference_date: date,
        lookback_days: int,
    ) -> dict[str, Any]:
        return {
            "patch_id": patch.patch_id,
            "reference_date": reference_date.isoformat(),
            f"{self.source_name}_dataset_id": self.dataset_id,
            f"{self.source_name}_role": self.role,
            f"{self.source_name}_summary_placeholder": None,
            "lookback_days": lookback_days,
        }


@dataclass(slots=True)
class EERealSourceAdapter:
    """Earth Engine adapter base with common config metadata."""

    source_name: str
    dataset_id: str
    role: str
    required_bands: list[str] = field(default_factory=list)
    required_properties: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Sentinel2SRSource(EERealSourceAdapter):
    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        geom = _ee_geometry_from_patch(patch)
        start_date = reference_date - timedelta(days=lookback_days)
        bands = _coerce_band_list(self.required_bands, ["B2", "B3", "B4", "B8", "B11", "B12"])

        collection = (
            ee.ImageCollection(self.dataset_id)
            .filterBounds(geom)
            .filterDate(start_date.isoformat(), (reference_date + timedelta(days=1)).isoformat())
        )
        count = int(collection.size().getInfo() or 0)
        if count == 0:
            out = {f"optical_{b.lower()}_mean_l{lookback_days}d": float("nan") for b in bands}
            out["optical_s2_observation_count"] = 0.0
            return out

        reduced = (
            collection.select(bands)
            .median()
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1_000_000_000)
            .getInfo()
            or {}
        )

        out = {f"optical_{b.lower()}_mean_l{lookback_days}d": _safe_get_number(reduced, b) for b in bands}
        out["optical_s2_observation_count"] = float(count)
        return out


@dataclass(slots=True)
class Sentinel2CloudProbSource(EERealSourceAdapter):
    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        geom = _ee_geometry_from_patch(patch)
        start_date = reference_date - timedelta(days=lookback_days)
        cloud_band = self.required_properties[0] if self.required_properties and not self.required_properties[0].startswith("TODO_") else "probability"

        collection = (
            ee.ImageCollection(self.dataset_id)
            .filterBounds(geom)
            .filterDate(start_date.isoformat(), (reference_date + timedelta(days=1)).isoformat())
        )
        count = int(collection.size().getInfo() or 0)
        if count == 0:
            return {
                "optical_cloud_probability_mean_l180d": float("nan"),
                "optical_cloud_probability_obs_count": 0.0,
            }

        reduced = (
            collection.select([cloud_band])
            .mean()
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1_000_000_000)
            .getInfo()
            or {}
        )
        return {
            f"optical_cloud_probability_mean_l{lookback_days}d": _safe_get_number(reduced, cloud_band),
            "optical_cloud_probability_obs_count": float(count),
        }


@dataclass(slots=True)
class CloudScorePlusSource(EERealSourceAdapter):
    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        geom = _ee_geometry_from_patch(patch)
        start_date = reference_date - timedelta(days=lookback_days)
        prop_fields = [p for p in self.required_properties if not p.startswith("TODO_")] or ["cs", "cs_cdf"]

        collection = (
            ee.ImageCollection(self.dataset_id)
            .filterBounds(geom)
            .filterDate(start_date.isoformat(), (reference_date + timedelta(days=1)).isoformat())
        )
        count = int(collection.size().getInfo() or 0)
        if count == 0:
            return {
                f"optical_cloudscore_{f}_mean_l{lookback_days}d": float("nan") for f in prop_fields
            } | {"optical_cloudscore_obs_count": 0.0}

        out: dict[str, Any] = {"optical_cloudscore_obs_count": float(count)}
        for field_name in prop_fields:
            try:
                reduced = (
                    collection.select([field_name])
                    .mean()
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1_000_000_000)
                    .getInfo()
                    or {}
                )
                out[f"optical_cloudscore_{field_name}_mean_l{lookback_days}d"] = _safe_get_number(reduced, field_name)
            except Exception:
                out[f"optical_cloudscore_{field_name}_mean_l{lookback_days}d"] = float("nan")
        return out


@dataclass(slots=True)
class ERA5LandSource(EERealSourceAdapter):
    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        geom = _ee_geometry_from_patch(patch)
        start_date = reference_date - timedelta(days=lookback_days)
        vars_ = [v for v in self.variables if not v.startswith("TODO_")] or ["temperature_2m", "total_precipitation_sum"]

        collection = (
            ee.ImageCollection(self.dataset_id)
            .filterBounds(geom)
            .filterDate(start_date.isoformat(), (reference_date + timedelta(days=1)).isoformat())
        )
        count = int(collection.size().getInfo() or 0)
        out: dict[str, Any] = {"weather_era5_obs_count": float(count)}
        if count == 0:
            for var in vars_:
                out[f"weather_{var}_mean_l{lookback_days}d"] = float("nan")
            return out

        for var in vars_:
            try:
                reduced = (
                    collection.select([var])
                    .mean()
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=10_000, maxPixels=1_000_000_000)
                    .getInfo()
                    or {}
                )
                out[f"weather_{var}_mean_l{lookback_days}d"] = _safe_get_number(reduced, var)
            except Exception:
                out[f"weather_{var}_mean_l{lookback_days}d"] = float("nan")
        return out


@dataclass(slots=True)
class SRTMSource(EERealSourceAdapter):
    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        _ = reference_date
        _ = lookback_days
        geom = _ee_geometry_from_patch(patch)
        bands = _coerce_band_list(self.required_bands, ["elevation"])

        image = ee.Image(self.dataset_id)
        out: dict[str, Any] = {}
        for band in bands:
            try:
                reduced = (
                    image.select([band])
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=90, maxPixels=1_000_000_000)
                    .getInfo()
                    or {}
                )
                out[f"terrain_{band}_mean"] = _safe_get_number(reduced, band)
            except Exception:
                out[f"terrain_{band}_mean"] = float("nan")
        return out


@dataclass(slots=True)
class VIIRSActiveFireSource(EERealSourceAdapter):
    """Active fire source supporting MODIS MOD14A1 (FireMask) or image-count fallback."""

    fire_band: str = "FireMask"
    fire_threshold: int = 7

    def _count_fire_pixels(
        self, collection: Any, geom: Any, start_str: str, end_str: str, ee: Any,
    ) -> float:
        filtered = collection.filterDate(start_str, end_str)
        n_images = int(filtered.size().getInfo() or 0)
        if n_images == 0:
            return 0.0
        try:
            fire_sum = (
                filtered.select([self.fire_band])
                .map(lambda img: img.gte(self.fire_threshold).selfMask())
                .sum()
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geom,
                    scale=1000,
                    maxPixels=1_000_000_000,
                )
                .getInfo()
                or {}
            )
            return float(fire_sum.get(self.fire_band, 0) or 0)
        except Exception:
            return float(n_images)

    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee  # type: ignore

        geom = _ee_geometry_from_patch(patch)
        start_date = reference_date - timedelta(days=lookback_days)
        collection = ee.ImageCollection(self.dataset_id).filterBounds(geom)

        next_day = (reference_date + timedelta(days=1)).isoformat()
        past = self._count_fire_pixels(
            collection, geom, start_date.isoformat(), next_day, ee,
        )
        c7 = self._count_fire_pixels(
            collection, geom, next_day, (reference_date + timedelta(days=8)).isoformat(), ee,
        )
        c30 = self._count_fire_pixels(
            collection, geom, next_day, (reference_date + timedelta(days=31)).isoformat(), ee,
        )
        c60 = self._count_fire_pixels(
            collection, geom, next_day, (reference_date + timedelta(days=61)).isoformat(), ee,
        )

        return {
            f"temporal_fire_past_count_l{lookback_days}d": past,
            "viirs_future_fire_count_7d": c7,
            "viirs_future_fire_count_30d": c30,
            "viirs_future_fire_count_60d": c60,
        }


@dataclass(slots=True)
class BurnSeveritySource(EERealSourceAdapter):
    """Compute pre/post-fire NBR composites for severity label construction.

    Uses MODIS MCD64A1 to detect burn timing in the future label window,
    then builds Sentinel-2 median composites before and after the burn to
    derive dNBR.  Post-fire imagery is permitted here because these outputs
    are used exclusively for *label* construction, never as features.
    """

    prefire_days: int = 90
    postfire_start_days: int = 30
    postfire_end_days: int = 120

    def _estimate_burn_date(self, geom: Any, future_start: date, future_end: date, ee: Any) -> date | None:
        burn_col = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filterBounds(geom)
            .filterDate(
                future_start.isoformat(),
                (future_end + timedelta(days=31)).isoformat(),
            )
        )
        n = int(burn_col.size().getInfo() or 0)
        if n == 0:
            return None

        info = (
            burn_col.select(["BurnDate"])
            .max()
            .reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=geom,
                scale=500,
                maxPixels=1_000_000_000,
            )
            .getInfo()
            or {}
        )
        day_of_year = info.get("BurnDate", 0)
        if not day_of_year or int(day_of_year) <= 0:
            return None
        try:
            return date(future_start.year, 1, 1) + timedelta(days=int(day_of_year) - 1)
        except (ValueError, OverflowError):
            return None

    def _s2_nbr_composite(self, geom: Any, start: date, end: date, ee: Any) -> tuple[float, float]:
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
            .select(["B8", "B12"])
        )
        if int(col.size().getInfo() or 0) == 0:
            return float("nan"), float("nan")
        info = (
            col.median()
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1_000_000_000)
            .getInfo()
            or {}
        )
        return _safe_get_number(info, "B8"), _safe_get_number(info, "B12")

    def fetch_patch_timeseries(self, patch: PatchRecord, reference_date: date, lookback_days: int) -> dict[str, Any]:
        import ee as _ee  # type: ignore
        import math

        geom = _ee_geometry_from_patch(patch)
        future_start = reference_date + timedelta(days=1)
        future_end = reference_date + timedelta(days=60)

        burn_date = self._estimate_burn_date(geom, future_start, future_end, _ee)
        if burn_date is None:
            return {}

        pre_nir, pre_swir2 = self._s2_nbr_composite(
            geom, burn_date - timedelta(days=self.prefire_days), burn_date - timedelta(days=1), _ee,
        )
        post_nir, post_swir2 = self._s2_nbr_composite(
            geom, burn_date + timedelta(days=self.postfire_start_days),
            burn_date + timedelta(days=self.postfire_end_days), _ee,
        )

        def _nbr(nir: float, swir2: float) -> float | None:
            if math.isnan(nir) or math.isnan(swir2) or (nir + swir2) == 0:
                return None
            return (nir - swir2) / (nir + swir2)

        pre_nbr = _nbr(pre_nir, pre_swir2)
        post_nbr = _nbr(post_nir, post_swir2)

        out: dict[str, Any] = {
            "prefire_nir_mean": pre_nir,
            "prefire_swir2_mean": pre_swir2,
            "postfire_nir_mean": post_nir,
            "postfire_swir2_mean": post_swir2,
        }
        if pre_nbr is not None:
            out["prefire_nbr"] = pre_nbr
        if post_nbr is not None:
            out["postfire_nbr"] = post_nbr
        if pre_nbr is not None and post_nbr is not None:
            out["dnbr"] = pre_nbr - post_nbr
        return out


def cloud_qa_source_priority() -> tuple[str, str]:
    """Return primary and auxiliary cloud-quality source names."""
    return PRIMARY_CLOUD_SOURCE, AUXILIARY_CLOUD_SOURCE


def _placeholder_adapter(name: str, cfg: dict[str, Any]) -> BasePlaceholderSource:
    return BasePlaceholderSource(
        source_name=name,
        dataset_id=str(cfg.get("dataset_id", FINAL_DATASETS.get(name, "UNKNOWN"))),
        role=str(cfg.get("role", "placeholder")),
    )


def _real_adapter(name: str, cfg: dict[str, Any]) -> SourceAdapter:
    common = {
        "source_name": name,
        "dataset_id": str(cfg.get("dataset_id", FINAL_DATASETS.get(name, "UNKNOWN"))),
        "role": str(cfg.get("role", "")),
        "required_bands": list(cfg.get("required_bands", [])),
        "required_properties": list(cfg.get("required_properties", [])),
        "variables": list(cfg.get("variables", [])),
    }

    if name == "sentinel2_sr":
        return Sentinel2SRSource(**common)
    if name == "sentinel2_cloud_probability":
        return Sentinel2CloudProbSource(**common)
    if name == "cloud_score_plus":
        return CloudScorePlusSource(**common)
    if name == "era5_land_daily":
        return ERA5LandSource(**common)
    if name == "srtm":
        return SRTMSource(**common)
    if name == "viirs_active_fire":
        return VIIRSActiveFireSource(**common)
    if name == "burn_severity":
        return BurnSeveritySource(**common)
    return _placeholder_adapter(name, cfg)


def default_source_adapters(
    source_configs: dict[str, Any] | None = None,
    mode: str = "placeholder",
) -> list[SourceAdapter]:
    """Return enabled adapters for placeholder or real-pilot mode."""
    source_configs = source_configs or {}
    adapters: list[SourceAdapter] = []

    if not source_configs:
        # Backward fallback.
        for name in [
            "sentinel2_sr",
            "sentinel2_cloud_probability",
            "cloud_score_plus",
            "era5_land_daily",
            "viirs_active_fire",
            "srtm",
        ]:
            cfg = {
                "dataset_id": FINAL_DATASETS[name],
                "enabled": True,
                "role": "default",
            }
            adapters.append(_placeholder_adapter(name, cfg))
        return adapters

    for name, raw_cfg in source_configs.items():
        cfg = _safe_source_cfg(raw_cfg)
        if not bool(cfg.get("enabled", False)):
            continue
        if mode == "real_pilot":
            adapters.append(_real_adapter(name, cfg))
        else:
            adapters.append(_placeholder_adapter(name, cfg))

    return adapters
