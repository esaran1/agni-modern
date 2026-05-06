import pandas as pd
import pytest

from agni_modern.data.parquet_io import enforce_leakage_guard, enforce_unique_patch_date


def test_enforce_unique_patch_date_raises() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["a", "a"],
            "reference_date": ["2021-01-01", "2021-01-01"],
        }
    )
    with pytest.raises(ValueError):
        enforce_unique_patch_date(df)


def test_enforce_leakage_guard_raises() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["a"],
            "reference_date": ["2021-01-01"],
            "feature_max_timestamp": ["2021-01-02"],
        }
    )
    with pytest.raises(ValueError):
        enforce_leakage_guard(df)
