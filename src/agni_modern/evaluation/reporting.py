"""Experiment reporting utilities for JSON/CSV result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_json_report(metrics: dict[str, float], output_path: Path) -> None:
    """Persist metrics report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def append_csv_report(row: dict[str, object], csv_path: Path) -> None:
    """Append one experiment row into CSV summary."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)
