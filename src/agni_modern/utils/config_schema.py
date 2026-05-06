"""Pydantic schemas for experiment configuration validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SpatialConfig(BaseModel):
    grid_km: int = Field(default=5, ge=1)
    crs: str = "EPSG:4326"
    patch_id_template: str = "indo_{row}_{col}"


class TemporalConfig(BaseModel):
    reference_start: str
    reference_end: str
    reference_frequency: str = "7D"
    lookback_days: int = Field(default=180, ge=1)
    horizons_days: list[int] = Field(default_factory=lambda: [7, 30, 60])


class CloudQualityConfig(BaseModel):
    """Primary/auxiliary cloud-quality source selection."""

    primary_source: str = "sentinel2_cloud_probability"
    auxiliary_source: str = "cloud_score_plus"
    strategy: str = "primary_with_auxiliary_qa"
    notes: str = ""


class SourceConfig(BaseModel):
    """Config contract for one dataset source entry.

    Notes:
    - `dataset_id` is the Earth Engine dataset identifier.
    - Band/property names remain configurable placeholders until finalized.
    """

    dataset_id: str
    enabled: bool
    role: str
    notes: str
    required_bands: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    variables: list[str] = Field(default_factory=list)


class IOConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    patch_grid_path: str = "data/interim/patch_grid.parquet"
    dataset_path: str = "data/processed/patch_date_table.parquet"
    partition_cols: list[str] = Field(default_factory=lambda: ["reference_year"])


class QualityConfig(BaseModel):
    enforce_unique_patch_date: bool = True
    enforce_feature_time_leakage_guard: bool = True


class DataConfig(BaseModel):
    version: int = 1
    aoi: dict[str, Any] = Field(default_factory=dict)
    spatial: SpatialConfig
    temporal: TemporalConfig
    cloud_quality: CloudQualityConfig = Field(default_factory=CloudQualityConfig)
    sources: dict[str, SourceConfig]
    io: IOConfig
    quality: QualityConfig = Field(default_factory=QualityConfig)


class FeatureConfig(BaseModel):
    version: int = 1
    feature_families: dict[str, bool]
    spectral: dict[str, Any] = Field(default_factory=dict)
    vegetation: dict[str, Any] = Field(default_factory=dict)
    burn_indices: dict[str, Any] = Field(default_factory=dict)
    weather: dict[str, Any] = Field(default_factory=dict)
    terrain: dict[str, Any] = Field(default_factory=dict)
    landcover: dict[str, Any] = Field(default_factory=dict)
    human_pressure: dict[str, Any] = Field(default_factory=dict)
    temporal_rolling: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    loss: dict[str, Any] = Field(default_factory=dict)


class ExperimentMeta(BaseModel):
    name: str
    seed: int = 42


class IncludeConfig(BaseModel):
    data: str
    features: str
    model: str


class SplitConfig(BaseModel):
    strategy: str
    train_end: str | None = None
    val_end: str | None = None
    test_end: str | None = None
    holdout_regions: list[str] = Field(default_factory=list)
    val_fraction: float | None = None
    test_fraction: float | None = None


class OutputConfig(BaseModel):
    model_dir: str = "outputs/models"
    metrics_dir: str = "outputs/metrics"
    predictions_dir: str = "outputs/predictions"
    plots_dir: str = "outputs/plots"
    maps_dir: str = "outputs/maps"


class TrainConfig(BaseModel):
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 64
    epochs: int = 20


class EvalConfig(BaseModel):
    calibration_bins: int = 10
    top_k_values: list[int] = Field(default_factory=lambda: [100, 500, 1000])
    #: When true, derive patch_id prefixes via :func:`auto_spatial_holdout_prefixes`
    #: from the full dataset and merge with ``split.holdout_regions`` / CLI flags.
    spatial_holdout_auto: bool = False
    spatial_holdout_fraction: float = Field(default=0.25, ge=0.05, le=0.5)


class ExperimentConfig(BaseModel):
    experiment: ExperimentMeta
    includes: IncludeConfig
    task: dict[str, Any]
    split: SplitConfig
    outputs: OutputConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
