"""Deterministic synthetic dataset generation for end-to-end plumbing validation.

This module is intentionally non-scientific and designed only to validate repository
architecture without external APIs or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from agni_modern.labels.severity import build_severity_labels


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class SyntheticDataConfig:
    """Configuration for synthetic patch-date dataset generation."""

    seed: int = 42
    num_patches: int = 48
    num_reference_dates: int = 80
    start_date: str = "2021-01-01"
    frequency_days: int = 7


def generate_synthetic_patch_date_table(cfg: SyntheticDataConfig) -> pd.DataFrame:
    """Generate canonical synthetic rows with learnable, imbalanced occurrence signal.

    Signal design (synthetic only):
    - Occurrence probability depends on dryness/weather + human pressure + terrain.
    - Severity is generated only when 30-day occurrence is positive.
    - 7/30/60-day labels are correlated but differ by horizon calibration.
    """
    rng = np.random.default_rng(cfg.seed)

    start = date.fromisoformat(cfg.start_date)
    reference_dates = [start + timedelta(days=cfg.frequency_days * i) for i in range(cfg.num_reference_dates)]

    patch_ids = [f"syn_patch_{i:03d}" for i in range(cfg.num_patches)]
    centroid_lats = rng.uniform(-7.5, 4.5, size=cfg.num_patches)
    centroid_lons = rng.uniform(96.0, 138.0, size=cfg.num_patches)

    # Patch-level latent factors reused across time for structured signal.
    terrain_factor = rng.normal(0.0, 1.0, size=cfg.num_patches)
    human_factor = rng.normal(0.0, 1.0, size=cfg.num_patches)

    rows: list[dict[str, object]] = []
    for p_idx, patch_id in enumerate(patch_ids):
        prev_occ = 0.0
        for ref in reference_dates:
            day = ref.timetuple().tm_yday
            seasonal = np.sin(2.0 * np.pi * day / 365.25)

            weather_temp = 27.0 + 3.0 * seasonal + rng.normal(0.0, 1.1)
            weather_precip = max(0.0, 7.5 - 2.8 * seasonal + rng.normal(0.0, 1.5))
            weather_wind = 2.5 + 0.7 * np.cos(2.0 * np.pi * day / 365.25) + rng.normal(0.0, 0.4)

            optical_dryness = 0.55 + 0.18 * seasonal + 0.10 * terrain_factor[p_idx] + rng.normal(0.0, 0.08)
            optical_green = 0.65 - 0.20 * seasonal + rng.normal(0.0, 0.07)
            optical_smoke_proxy = 0.20 + 0.18 * prev_occ + rng.normal(0.0, 0.05)

            terrain_elevation = max(0.0, 200.0 + 180.0 * terrain_factor[p_idx] + rng.normal(0.0, 35.0))
            terrain_slope = max(0.0, 6.0 + 3.0 * abs(terrain_factor[p_idx]) + rng.normal(0.0, 1.0))

            landcover_bare_fraction = np.clip(0.15 + 0.10 * terrain_factor[p_idx] + rng.normal(0.0, 0.04), 0.0, 1.0)
            landcover_tree_fraction = np.clip(0.55 - 0.12 * terrain_factor[p_idx] + rng.normal(0.0, 0.05), 0.0, 1.0)

            human_population_density = max(1.0, 220.0 + 150.0 * human_factor[p_idx] + rng.normal(0.0, 30.0))
            human_built_fraction = np.clip(0.12 + 0.14 * human_factor[p_idx] + rng.normal(0.0, 0.04), 0.0, 1.0)
            human_access_proxy = np.clip(0.35 + 0.20 * human_factor[p_idx] + rng.normal(0.0, 0.08), 0.0, 1.0)

            temporal_recent_dryness = np.clip(0.6 * optical_dryness + 0.4 * prev_occ + rng.normal(0.0, 0.04), 0.0, 1.0)
            temporal_recent_fire_pressure = np.clip(0.3 * prev_occ + 0.2 * human_access_proxy + rng.normal(0.0, 0.03), 0.0, 1.0)

            # Moderate imbalance: around ~20-35% positives depending on patch/time.
            logit = (
                -1.15
                + 2.10 * optical_dryness
                - 0.28 * (weather_precip / 10.0)
                + 0.35 * (weather_wind / 5.0)
                + 0.40 * human_access_proxy
                + 0.22 * (terrain_slope / 10.0)
                + 0.45 * temporal_recent_fire_pressure
                + rng.normal(0.0, 0.25)
            )
            p30 = float(_sigmoid(np.array([logit]))[0])
            p7 = float(_sigmoid(np.array([logit - 0.45]))[0])
            p60 = float(_sigmoid(np.array([logit + 0.30]))[0])

            y_occ_7d = int(rng.random() < p7)
            y_occ_30d = int(rng.random() < p30)
            y_occ_60d = int(rng.random() < p60)

            synth_dnbr = float(
                np.clip(
                    0.15
                    + 0.40 * optical_dryness
                    + 0.15 * human_built_fraction
                    + 0.08 * (terrain_slope / 10.0)
                    + rng.normal(0.0, 0.06),
                    0.0,
                    1.0,
                )
            )
            sev_ctx: dict[str, object] = {
                "y_occ_7d": y_occ_7d,
                "y_occ_30d": y_occ_30d,
                "y_occ_60d": y_occ_60d,
                "dnbr": synth_dnbr,
            }
            sev_labels = build_severity_labels(reference_date=ref, context=sev_ctx)
            y_sev_available = sev_labels["y_sev_available"]
            sev_reg = sev_labels["y_sev_reg"] if y_sev_available else np.nan
            sev_cls = sev_labels["y_sev_cls"] if y_sev_available else np.nan

            label_window_start = ref + timedelta(days=1)
            label_window_end = ref + timedelta(days=60)

            row = {
                "patch_id": patch_id,
                "reference_date": ref,
                "centroid_lat": float(centroid_lats[p_idx]),
                "centroid_lon": float(centroid_lons[p_idx]),
                "feature_max_timestamp": ref,
                "label_window_start": label_window_start,
                "label_window_end": label_window_end,
                "y_occ_7d": y_occ_7d,
                "y_occ_30d": y_occ_30d,
                "y_occ_60d": y_occ_60d,
                "y_sev_available": y_sev_available,
                "y_sev_reg": sev_reg,
                "y_sev_cls": sev_cls,
                # Canonical feature namespaces used by real training code.
                "optical_dryness_idx": float(np.clip(optical_dryness, 0.0, 1.0)),
                "optical_green_idx": float(np.clip(optical_green, 0.0, 1.0)),
                "optical_smoke_proxy": float(np.clip(optical_smoke_proxy, 0.0, 1.0)),
                "weather_temp_mean_30d": float(weather_temp),
                "weather_precip_sum_30d": float(weather_precip),
                "weather_wind_mean_14d": float(weather_wind),
                "terrain_elevation": float(terrain_elevation),
                "terrain_slope": float(terrain_slope),
                "landcover_bare_fraction": float(landcover_bare_fraction),
                "landcover_tree_fraction": float(landcover_tree_fraction),
                "human_population_density_worldpop": float(human_population_density),
                "human_built_surface_fraction_ghsl": float(human_built_fraction),
                "human_distance_to_settlement_proxy": float(1.0 - human_access_proxy),
                "temporal_recent_dryness": float(temporal_recent_dryness),
                "temporal_recent_fire_pressure": float(temporal_recent_fire_pressure),
            }
            rows.append(row)
            prev_occ = float(y_occ_30d)

    df = pd.DataFrame(rows)
    df = df.sort_values(["patch_id", "reference_date"]).reset_index(drop=True)
    df["reference_year"] = pd.to_datetime(df["reference_date"]).dt.year
    return df
