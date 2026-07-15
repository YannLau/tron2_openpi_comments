import pytest

from scripts import compute_norm_stats


def test_resolve_config_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        compute_norm_stats.resolve_config(None, None)

    with pytest.raises(ValueError, match="exactly one"):
        compute_norm_stats.resolve_config("debug", "task.yaml")


def test_resolve_config_loads_task_yaml(monkeypatch: pytest.MonkeyPatch):
    expected = object()
    monkeypatch.setattr(compute_norm_stats.tron2_task_config, "create_train_config", lambda path: expected)

    assert compute_norm_stats.resolve_config(None, "task.yaml") is expected
