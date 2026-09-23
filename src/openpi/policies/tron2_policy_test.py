from __future__ import annotations

import numpy as np
import pytest

from openpi.policies import tron2_policy


def _sample(dimension: int) -> dict:
    return {
        "state": np.arange(dimension, dtype=np.float32),
        "actions": np.arange(2 * dimension, dtype=np.float32).reshape(2, dimension),
        "images": {"cam_high": np.zeros((3, 8, 8), dtype=np.uint8)},
        "prompt": "test",
    }


@pytest.mark.parametrize("dimension", [16, 18, 19, 21, 26, 31, 32])
def test_tron2_transforms_support_modular_physical_dimensions(dimension):
    transformed = tron2_policy.Tron2Inputs(state_dim=dimension, action_dim=dimension)(_sample(dimension))
    assert transformed["state"].shape == (dimension,)
    assert transformed["actions"].shape == (2, dimension)

    model_actions = np.arange(2 * 32, dtype=np.float32).reshape(2, 32)
    outputs = tron2_policy.Tron2Outputs(action_dim=dimension)({"actions": model_actions})
    np.testing.assert_array_equal(outputs["actions"], model_actions[:, :dimension])


@pytest.mark.parametrize(
    ("state_dim", "action_dim"),
    [(16, 18), (18, 16), (0, 0), (33, 33)],
)
def test_tron2_transforms_reject_invalid_dimension_pairs(state_dim, action_dim):
    with pytest.raises(ValueError, match="matching pair between 1 and 32"):
        tron2_policy.Tron2Inputs(state_dim=state_dim, action_dim=action_dim)


def test_tron2_inputs_reject_sample_dimension_mismatch():
    transform = tron2_policy.Tron2Inputs(state_dim=19, action_dim=19)
    with pytest.raises(ValueError, match="state dimension.*expected 19.*got 16"):
        transform(_sample(16))
