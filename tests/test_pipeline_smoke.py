import pandas as pd

from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.training.train_tabular import train_tabular_occurrence


def test_tiny_tabular_pipeline_smoke(tmp_path) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "patch_id": f"p{i % 5}",
                "reference_date": f"2021-01-{(i % 28) + 1:02d}",
                "optical_a": float(i % 3),
                "weather_b": float(i % 5),
                "static_c": 1.0,
                "landcover_d": 0.2,
                "human_e": 0.1,
                "temporal_f": float(i),
                "y_occ_30d": int(i % 2 == 0),
                "y_sev_available": 0,
            }
        )
    df = pd.DataFrame(rows)

    train, val, test = temporal_holdout_split(df, "2021-01-15", "2021-01-22", "2021-01-31")
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_tabular_occurrence(
        train_df=train,
        val_df=val,
        test_df=test,
        model_name="logreg",
        model_params={"max_iter": 200},
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        seed=7,
        target_col="y_occ_30d",
    )
    assert model_path.exists()
    assert metrics_path.exists()
    assert "f1" in metrics
