from __future__ import annotations

from types import SimpleNamespace

from scripts import serve_policy


def test_create_policy_publishes_configured_physical_dimensions(monkeypatch):
    train_config = SimpleNamespace(
        name="test_tron2",
        data=SimpleNamespace(state_dim=19, action_dim=19),
    )
    policy = SimpleNamespace(metadata={})
    monkeypatch.setattr(serve_policy._config, "get_config", lambda _name: train_config)  # noqa: SLF001
    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        lambda *_args, **_kwargs: policy,
    )

    result = serve_policy.create_policy(
        serve_policy.Args(),
        {
            "policy": {
                "config": "test_tron2",
                "checkpoint_dir": "/tmp/test-checkpoint",
            }
        },
    )

    assert result.metadata["state_dim"] == 19
    assert result.metadata["action_dim"] == 19
