# Agni Modern

Agni Modern is a research-grade successor repository for wildfire forecasting in Indonesia.
It is built from scratch for Python 3.11+ and designed for reproducible experimentation,
with two model tracks sharing one processed Parquet backbone:

1. Tabular models (logistic regression, random forest, XGBoost)
2. Temporal transformer over structured time-step tokens

## Goals

- Forecast wildfire occurrence at 7/30/60-day horizons
- Forecast pre-ignition fire severity (conditional on fire)
- Produce operational risk maps:
  - fire probability
  - conditional severity
  - expected severity risk = `P(fire) * severity`

## Design Principles

- No dependency on legacy Agni internals
- No claim of direct compatibility with legacy Agni code
- Config-driven pipelines (`YAML + Pydantic`)
- One canonical row per `(patch_id, reference_date)`
- Strict leakage control: model features must only use data at or before `reference_date`
- Reproducible seeds, partitioned outputs, explicit artifacts

## Repository Layout

```text
agni-modern/
  configs/
  data/
    raw/
    interim/
    processed/
  notebooks/
  scripts/
  src/agni_modern/
  tests/
  outputs/
    models/
    metrics/
    maps/
    figures/
```

## Default v1+ Dataset Stack

The default Earth Engine stack is configured in [configs/data/base.yaml](/Users/Evan/agni-modern/agni-modern/configs/data/base.yaml).

| Source Key | Dataset ID | Role |
|---|---|---|
| `sentinel2_sr` | `COPERNICUS/S2_SR_HARMONIZED` | Optical imagery for spectral and vegetation features |
| `sentinel2_cloud_probability` | `COPERNICUS/S2_CLOUD_PROBABILITY` | Primary cloud mask source |
| `cloud_score_plus` | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | Auxiliary cloud/clear-pixel QA source |
| `era5_land_daily` | `ECMWF/ERA5_LAND/DAILY_AGGR` | Weather and land-surface context |
| `viirs_active_fire` | `NASA/LANCE/SNPP_VIIRS/C2` | Fire-occurrence label source |
| `modis_burned_area` | `MODIS/061/MCD64A1` | Burn timing/extent support for severity labels |
| `srtm` | `CGIAR/SRTM90_V4` | Terrain/elevation context |
| `dynamic_world` | `GOOGLE/DYNAMICWORLD/V1` | Land-cover context |
| `worldpop` | `WorldPop/GP/100m/pop` | Population-based human-pressure features |
| `ghsl_built_surface` | `JRC/GHSL/P2023A/GHS_BUILT_S` | Built-environment human-pressure proxy |

### Cloud QA Strategy

- Primary cloud-quality source: `sentinel2_cloud_probability`
- Auxiliary cloud-quality source: `cloud_score_plus`
- Strategy: use the primary mask with auxiliary clear-pixel QA signals where beneficial.

### Human-Pressure Strategy

Human-pressure features are expected to include both:
- population features from WorldPop
- built-environment features from GHSL built surface

Provider-specific band/property names are intentionally configurable and should be finalized via YAML.

## Canonical Data Contract

Each processed row represents one `(patch_id, reference_date)` pair and contains:

- Keys: `patch_id`, `reference_date`
- Feature namespaces:
  - `optical_*`
  - `weather_*`
  - `static_*`
  - `landcover_*`
  - `human_*`
  - `temporal_*`
- Labels:
  - `y_occ_7d`, `y_occ_30d`, `y_occ_60d`
  - `y_sev_reg`, `y_sev_cls`, `y_sev_available`
- Leakage metadata:
  - `feature_max_timestamp`
  - `label_window_start`, `label_window_end`

## Leakage Rules

- Features must be derived from observations at or before `reference_date`
- Future windows are used only for labels
- Post-fire composites are used only for historical severity label generation
- Inference never depends on post-reference-date observations

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the main pipeline scripts:

```bash
python scripts/build_grid.py configs/experiments/baseline_occurrence_7d.yaml
python scripts/build_dataset.py configs/experiments/baseline_occurrence_7d.yaml
python scripts/train_baselines.py configs/experiments/baseline_occurrence_7d.yaml
python scripts/evaluate.py configs/experiments/baseline_occurrence_7d.yaml
```

## Config Model

- Base configs live in `configs/data`, `configs/features`, and `configs/models`
- Experiment configs in `configs/experiments` compose those bases
- Local override support:

```bash
python scripts/train_baselines.py \
  configs/experiments/baseline_occurrence_30d.yaml \
  --set train.seed=123 \
  --set model.params.max_depth=8
```

## Pipeline Stages

1. Build Indonesia patch grid (fixed 5km default)
2. Generate reference dates and lookback windows
3. Fetch source summaries per patch-date
4. Build feature table + labels
5. Validate schema and leakage constraints
6. Train tabular and transformer models
7. Evaluate temporal/spatial holdouts and rank-based utility
8. Run inference and export map-ready outputs

## Experiments and Outputs

- Trained artifacts: `outputs/models/`
- Metrics and reports: `outputs/metrics/`
- Raster/vector map outputs: `outputs/maps/`
- Figures and explainability plots: `outputs/figures/`

## Limitations / TODO

- Provider field names (bands/properties) remain configurable placeholders until finalized
- Earth Engine credential flow is scaffolded, not production-hardened
- Cloud orchestration and distributed training are future extensions
- Severity bin defaults are configurable and should be calibrated with domain experts

## Synthetic End-to-End Smoke Test

Use the synthetic path to validate repository plumbing without Earth Engine or real wildfire data.

```bash
python scripts/build_synthetic_dataset.py configs/experiments/synthetic_baseline.yaml
python scripts/train_baselines.py configs/experiments/synthetic_baseline.yaml
python scripts/evaluate.py configs/experiments/synthetic_baseline.yaml
```

Expected outputs:
- `data/processed/patch_date_table.parquet` (synthetic canonical dataset)
- `outputs/models/synthetic_baseline_<model>.pkl`
- `outputs/metrics/synthetic_baseline_<model>.json`
- `outputs/metrics/synthetic_baseline_<model>_evaluation.json`
- `outputs/metrics/synthetic_baseline_resolved_config.json`

## Tiny Real-Data Pilot (Occurrence Only)

This repository includes a minimal real-data pilot for occurrence prediction (`y_occ_30d`) over a small region in Indonesia.

Pilot experiment config:
- `configs/experiments/pilot_occurrence_30d.yaml`
- data config: `configs/data/pilot_occurrence_jambi_small.yaml`

Pilot uses only:
- `COPERNICUS/S2_SR_HARMONIZED`
- `COPERNICUS/S2_CLOUD_PROBABILITY`
- `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`
- `ECMWF/ERA5_LAND/DAILY_AGGR`
- `CGIAR/SRTM90_V4`
- `NASA/LANCE/SNPP_VIIRS/C2`

Run end-to-end:

```bash
python scripts/build_grid.py configs/experiments/pilot_occurrence_30d.yaml
python scripts/build_dataset.py configs/experiments/pilot_occurrence_30d.yaml
python scripts/train_baselines.py configs/experiments/pilot_occurrence_30d.yaml
python scripts/evaluate.py configs/experiments/pilot_occurrence_30d.yaml
```

Outputs include:
- processed Parquet dataset
- dataset validation summary JSON
- baseline model artifacts
- training metrics JSON
- per-model prediction Parquet files
- per-model evaluation JSON

Notes:
- Severity modeling is intentionally not implemented in this pilot step.
- Field names that may vary by provider remain configurable in YAML.
