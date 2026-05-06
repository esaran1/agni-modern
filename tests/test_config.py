from agni_modern.utils.config_loader import ConfigError, load_experiment_config


def test_load_experiment_config() -> None:
    cfg = load_experiment_config("configs/experiments/baseline_occurrence_7d.yaml")
    assert cfg.experiment.name == "baseline_occurrence_7d"
    assert cfg.data.spatial.grid_km == 5
    assert cfg.model.name == "xgb_occurrence"


def test_pilot_transformer_experiment_has_eval_spatial_flags() -> None:
    cfg = load_experiment_config("configs/experiments/pilot_kalimantan_transformer.yaml")
    assert cfg.eval.spatial_holdout_auto is True
    assert 0.0 < cfg.eval.spatial_holdout_fraction <= 0.5


def test_bad_override_raises() -> None:
    try:
        load_experiment_config(
            "configs/experiments/baseline_occurrence_7d.yaml",
            overrides=["bad_override"],
        )
    except ConfigError:
        assert True
    else:
        assert False
