"""Google Earth Engine client bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EEAuthConfig:
    """Earth Engine authentication options."""

    service_account_email: str | None = None
    service_account_key_path: Path | None = None
    project: str | None = None


def initialize_earth_engine(auth: EEAuthConfig | None = None) -> None:
    """Initialize the Earth Engine SDK.

    TODO:
    - Add production-safe retries and explicit auth mode selection.
    - Validate credentials path and project settings.
    """
    try:
        import ee  # type: ignore
    except ImportError as exc:
        raise RuntimeError("earthengine-api is required to initialize EE") from exc

    if auth and auth.service_account_email and auth.service_account_key_path:
        # TODO: confirm service-account auth flow for deployment environment.
        credentials = ee.ServiceAccountCredentials(
            auth.service_account_email,
            str(auth.service_account_key_path),
        )
        ee.Initialize(credentials=credentials, project=auth.project)
        return

    # Default: user auth context (e.g. `earthengine authenticate` done locally).
    ee.Initialize(project=auth.project if auth else None)
